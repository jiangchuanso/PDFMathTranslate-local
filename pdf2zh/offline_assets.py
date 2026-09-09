"""Local (offline) replacements for front-end assets that are otherwise
fetched from public CDNs.

In an intranet / air-gapped environment every request to a public CDN fails,
which breaks parts of the Gradio GUI (most visibly the built-in PDF preview,
whose pdf.js worker is loaded from jsDelivr).

This module makes the GUI fully self-contained:

* ``pdf.worker.min.mjs`` is shipped inside the ``pdf2zh/static`` directory
  (see ``script/fetch_static_assets.py`` to (re)download it) and copied into
  the directory that Gradio serves under ``/assets/...``.
* ``gradio_pdf``'s bundled JavaScript is rewritten so that
  ``GlobalWorkerOptions.workerSrc`` points at that local copy instead of the
  jsDelivr URL. Gradio serves custom component files straight from disk
  (``gradio.routes.custom_component_path``), so patching the installed file is
  what the browser receives.
* Remaining CDN references in Gradio's own HTML template (Google Fonts
  preconnects, the async iframe-resizer script from cdnjs) are stripped.
* Gradio telemetry is disabled, so no analytics request leaves the machine.

All operations are idempotent and best-effort: if the environment does not
allow writing to the installed packages, a warning is logged and the GUI still
starts (with the original CDN behaviour).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

PDFJS_WORKER_FILENAME = "pdf.worker.min.mjs"

# Upstream location of the pdf.js worker used by gradio_pdf.
PDFJS_WORKER_CDN_URL = (
    "https://cdn.jsdelivr.net/gh/freddyaboulton/gradio-pdf@main/"
    + PDFJS_WORKER_FILENAME
)

# Gradio serves ``<gradio>/templates/frontend/assets`` under /assets/<path>,
# without authentication, so this URL always stays on the same origin.
LOCAL_ASSET_URL_PREFIX = "/assets"
LOCAL_PDFJS_WORKER_URL = f"{LOCAL_ASSET_URL_PREFIX}/{PDFJS_WORKER_FILENAME}"

# ``X.workerSrc = "https://cdn.jsdelivr.net/..."`` as emitted in the bundled
# gradio_pdf JavaScript (the local variable name is minified, hence \w+).
_WORKER_SRC_PATTERN = re.compile(
    r"(\w+\.workerSrc\s*=\s*)([\"'])https://cdn\.jsdelivr\.net/[^\"']*\2"
)


def bundled_static_dir() -> Path:
    """Directory inside the package that holds pre-downloaded static assets."""
    return Path(__file__).parent / "static"


def gradio_frontend_assets_dir() -> Path | None:
    """Directory served by Gradio under ``/assets/...`` (None if unavailable)."""
    try:
        import gradio
    except ImportError:
        return None
    return Path(gradio.__file__).parent / "templates" / "frontend" / "assets"


def locate_pdfjs_worker() -> Path | None:
    """Return a local copy of the pdf.js worker, or None when not available."""
    candidates = [
        bundled_static_dir() / PDFJS_WORKER_FILENAME,
        (d := gradio_frontend_assets_dir()) and d / PDFJS_WORKER_FILENAME,
    ]
    for candidate in candidates:
        if candidate and candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def install_pdfjs_worker() -> bool:
    """Make the pdf.js worker reachable through Gradio's ``/assets`` route.

    Returns True when ``/assets/pdf.worker.min.mjs`` is served locally.
    """
    target_dir = gradio_frontend_assets_dir()
    if target_dir is None:
        logger.warning("Gradio is not installed; skip pdf.js worker localisation.")
        return False

    target = target_dir / PDFJS_WORKER_FILENAME
    if target.is_file() and target.stat().st_size > 0:
        return True

    source = locate_pdfjs_worker()
    if source is None:
        logger.warning(
            "pdf.js worker (%s) is not bundled; the PDF preview needs "
            "internet access. Run script/fetch_static_assets.py on a machine "
            "with internet access to bundle it.",
            PDFJS_WORKER_FILENAME,
        )
        return False

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    except OSError as e:
        logger.warning("Could not install local pdf.js worker: %s", e)
        return False
    return True


def patch_gradio_pdf_worker_url() -> int:
    """Point gradio_pdf's bundled JS at the local pdf.js worker.

    Returns the number of patched files.
    """
    try:
        import gradio_pdf
    except ImportError:
        return 0

    templates_dir = Path(gradio_pdf.__file__).parent / "templates"
    if not templates_dir.is_dir():
        return 0

    patched = 0
    for js_file in templates_dir.rglob("index.js"):
        try:
            source = js_file.read_text(encoding="utf-8")
        except OSError:
            continue
        if PDFJS_WORKER_CDN_URL not in source:
            continue  # already localised (or an unexpected build)
        new_source, count = _WORKER_SRC_PATTERN.subn(
            lambda m: f"{m.group(1)}{m.group(2)}{LOCAL_PDFJS_WORKER_URL}{m.group(2)}",
            source,
        )
        if not count:
            continue
        try:
            js_file.write_text(new_source, encoding="utf-8")
        except OSError as e:
            logger.warning("Could not localise %s: %s", js_file, e)
            continue
        patched += 1
    return patched


# ``<link rel="preconnect" href="https://fonts.googleapis.com" ... />`` and the
# (async) iframe-resizer script that Gradio ships in its HTML template.
_PRECONNECT_PATTERN = re.compile(
    r'<link\s+rel="preconnect"\s+href="https://fonts\.(?:googleapis|gstatic)\.com"'
    r"[^>]*>",
    re.IGNORECASE,
)
_CDN_SCRIPT_PATTERN = re.compile(
    r'<script\s+src="https://cdnjs\.cloudflare\.com/[^"]*"[^>]*>\s*</script>',
    re.IGNORECASE,
)


def patch_gradio_frontend_template() -> bool:
    """Strip CDN references from Gradio's own HTML template.

    Returns True when the template was rewritten (or was already clean).
    """
    assets_dir = gradio_frontend_assets_dir()
    if assets_dir is None:
        return False
    template = assets_dir.parent / "index.html"
    if not template.is_file():
        return False

    try:
        source = template.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning("Could not read %s: %s", template, e)
        return False

    cleaned = _CDN_SCRIPT_PATTERN.sub("", _PRECONNECT_PATTERN.sub("", source))
    if cleaned == source:
        return True
    try:
        template.write_text(cleaned, encoding="utf-8")
    except OSError as e:
        logger.warning("Could not localise %s: %s", template, e)
        return False
    return True


def disable_outbound_telemetry() -> None:
    """Stop Gradio (and friends) from calling home."""
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"


def prepare_offline_gui_assets() -> None:
    """Best-effort preparation for running the GUI without internet access."""
    disable_outbound_telemetry()
    patch_gradio_frontend_template()
    if install_pdfjs_worker():
        if patch_gradio_pdf_worker_url():
            logger.info("PDF preview worker served from %s", LOCAL_PDFJS_WORKER_URL)
