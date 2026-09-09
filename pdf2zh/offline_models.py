"""Offline model provisioning for the fully local translation engines.

In an intranet / air-gapped deployment no weight may be downloaded at run time.
All models are therefore shipped inside a single zip archive that is placed next
to the executable (or inside the installed package)::

    offline-models.zip
    |-- firefox/
    |   |-- en-zh/   model.bin  source.spm  target.spm
    |   |            shared_vocabulary.json  config.json
    |   `-- zh-en/   ...
    `-- argos/
        |-- translate-en_zh-1_9.argosmodel
        `-- translate-zh_en-1_9.argosmodel

The archive is extracted once into a writable directory and re-used afterwards,
so only the very first translation pays the extraction cost.

Generate the archive with::

    python script/fetch_offline_models.py
"""

import logging
import os
import shutil
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

MODELS_DIR_NAME = "offline-models"
MODELS_ZIP_NAME = f"{MODELS_DIR_NAME}.zip"

#: Sub directories inside the model root.
FIREFOX_SUBDIR = "firefox"
ARGOS_SUBDIR = "argos"

#: A CTranslate2 model directory is usable as soon as it holds the weights.
_CT2_REQUIRED_FILES = ("model.bin",)


class OfflineModelError(RuntimeError):
    """Raised when a required model is missing from the offline bundle."""


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def _is_writable(directory: Path) -> bool:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe = directory / ".pdf2zh-write-probe"
        probe.touch()
        probe.unlink()
        return True
    except OSError:
        return False


def _user_models_dir() -> Path:
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    else:
        base = Path(
            os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")
        )
    return base / "pdf2zh" / MODELS_DIR_NAME


def candidate_locations() -> list[Path]:
    """Places that may hold the model bundle, ordered by priority."""
    locations: list[Path] = []
    for env_var in ("PDF2ZH_MODELS_DIR", "PDF2ZH_MODELS_ZIP"):
        value = os.environ.get(env_var)
        if value:
            locations.append(Path(value))
    locations.extend(
        [
            _package_dir() / MODELS_ZIP_NAME,
            _package_dir() / MODELS_DIR_NAME,
            # PyStand / portable layout: the bundle sits next to pdf2zh.exe,
            # i.e. two levels above <...>/site-packages/pdf2zh.  This makes the
            # models findable no matter what the working directory is.
            _package_dir().parent / MODELS_ZIP_NAME,
            _package_dir().parent.parent / MODELS_ZIP_NAME,
            Path.cwd() / MODELS_ZIP_NAME,
            Path.cwd() / MODELS_DIR_NAME,
        ]
    )
    return locations


def _zip_signature(zip_path: Path) -> str:
    stat = zip_path.stat()
    return f"{stat.st_size}:{int(stat.st_mtime)}"


def _ensure_extracted(zip_path: Path) -> Path:
    """Extract *zip_path* once and return the resulting model root."""
    signature = _zip_signature(zip_path)
    sibling = zip_path.parent / MODELS_DIR_NAME
    destination = sibling if _is_writable(sibling) else _user_models_dir()

    marker = destination / ".extracted"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == signature:
        return _collapse_root(destination)

    logger.info("Extracting offline models from %s to %s ...", zip_path, destination)
    staging = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(staging)
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        staging.rename(destination)
        marker.write_text(signature, encoding="utf-8")
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return _collapse_root(destination)


def _collapse_root(root: Path) -> Path:
    """Tolerate archives that keep everything under a top level folder."""
    nested = root / MODELS_DIR_NAME
    if (nested / FIREFOX_SUBDIR).is_dir() or (nested / ARGOS_SUBDIR).is_dir():
        return nested
    return root


def models_root() -> Path | None:
    """Return the extracted model root, or ``None`` when no bundle is present."""
    for location in candidate_locations():
        if location.is_dir():
            return _collapse_root(location)
        if location.is_file() and zipfile.is_zipfile(location):
            return _ensure_extracted(location)
    return None


def has_offline_models() -> bool:
    return models_root() is not None


def ensure_firefox_model(src_lang: str, trg_lang: str) -> Path:
    """Return the directory holding the CTranslate2 model for a language pair."""
    root = models_root()
    if root is None:
        raise OfflineModelError(
            "No offline model bundle found. Place 'offline-models.zip' next to "
            "pdf2zh.exe (or point PDF2ZH_MODELS_ZIP / PDF2ZH_MODELS_DIR at it). "
            "Generate it with `python script/fetch_offline_models.py`."
        )

    base = root / FIREFOX_SUBDIR
    for name in (f"{src_lang}-{trg_lang}", f"{src_lang}_{trg_lang}"):
        model_dir = base / name
        if any((model_dir / f).is_file() for f in _CT2_REQUIRED_FILES):
            return model_dir

    available = sorted(p.name for p in base.iterdir()) if base.is_dir() else []
    raise OfflineModelError(
        f"The offline bundle has no firefox model for {src_lang}->{trg_lang}. "
        f"Available: {available or 'none'}"
    )


def ensure_argos_model(from_code: str, to_code: str) -> Path | None:
    """Return the bundled ``.argosmodel`` for a pair, or ``None`` when absent."""
    root = models_root()
    if root is None:
        return None
    directory = root / ARGOS_SUBDIR
    if not directory.is_dir():
        return None
    matches = sorted(directory.glob(f"translate-{from_code}_{to_code}-*.argosmodel"))
    return matches[0] if matches else None
