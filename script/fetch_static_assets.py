#!/usr/bin/env python3
"""Download the front-end assets that the GUI would otherwise load from a CDN.

Run this on a machine with internet access, then ship the resulting
``pdf2zh/static`` directory to the intranet host (or build the win64 package
with it already present).

Usage:
    python script/fetch_static_assets.py [output_dir]
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests

# Imported lazily so the script works even when optional deps are missing.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pdf2zh.offline_assets import (  # noqa: E402
    PDFJS_WORKER_CDN_URL,
    PDFJS_WORKER_FILENAME,
    bundled_static_dir,
)

TIMEOUT = 60


def fetch(url: str, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} -> {destination}")
    try:
        with requests.get(url, timeout=TIMEOUT, stream=True) as response:
            response.raise_for_status()
            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
    except Exception as e:  # noqa: BLE001 - report and keep going
        print(f"  FAILED: {e}")
        if destination.exists():
            destination.unlink()
        return False
    size = destination.stat().st_size
    print(f"  OK ({size / 1024:.0f} KiB)")
    return size > 0


def main() -> int:
    output_dir = (
        Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else bundled_static_dir()
    )
    ok = fetch(PDFJS_WORKER_CDN_URL, output_dir / PDFJS_WORKER_FILENAME)
    if not ok:
        print("Some assets could not be downloaded (see errors above).")
        return 1
    print(f"Static assets ready in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
