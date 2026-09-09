#!/usr/bin/env python
"""Download the local translation models and pack them into one offline bundle.

Two engines are covered:

* ``firefox`` - `firefox-translations` (CTranslate2).  We prefer the largest
  published variant (``opus-mt-tc-big-*``) and fall back to the standard
  ``opus-mt-*`` model when the big one does not exist for a pair.  For
  zh<->en only the standard model exists, so that one is used.
* ``argos``   - argos-translate ``.argosmodel`` archives.

The result is a single ``offline-models.zip`` that can be shipped to an
intranet machine; see :mod:`pdf2zh.offline_models` for the expected layout.

Usage::

    python script/fetch_offline_models.py                 # zh<->en, both engines
    python script/fetch_offline_models.py --pairs en-zh zh-en
    python script/fetch_offline_models.py --skip-argos --out D:/offline-models.zip
"""

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# HuggingFace mirrors of the Firefox/Bergamot models, already converted to
# CTranslate2.  Tried in order; the first repository that serves every file wins.
#
# jiangzhuo9357/opus-mt-*-ct2 is the INT8 build of the very same standard
# model: 80 MB instead of 155 MB and clearly faster on CPU, so it is the default
# for the local engines.  "tc-big" is the large Tatoeba-Challenge variant; it is
# missing for some pairs (e.g. zh<->en), hence the fallback chain.
FIREFOX_REPOS: dict[str, list[str]] = {
    # fastest on CPU / smallest download
    "int8": [
        "jiangzhuo9357/opus-mt-{src}-{trg}-ct2",
        "ooeoeo/opus-mt-tc-big-{src}-{trg}-ct2-float16",
        "ooeoeo/opus-mt-{src}-{trg}-ct2-float16",
    ],
    # largest model first, INT8 only as a CPU-speed fallback
    "quality": [
        "ooeoeo/opus-mt-tc-big-{src}-{trg}-ct2-float16",
        "jiangzhuo9357/opus-mt-{src}-{trg}-ct2",
        "ooeoeo/opus-mt-{src}-{trg}-ct2-float16",
    ],
}
FIREFOX_FILES = [
    "model.bin",
    "source.spm",
    "target.spm",
    "shared_vocabulary.json",
    "config.json",
]
ARGOS_BASE_URL = "https://argos-net.com/v1"

DEFAULT_PAIRS = ["en-zh", "zh-en"]
HEADERS = {"User-Agent": "Mozilla/5.0"}


def firefox_base_url(repo_template: str, src: str, trg: str) -> str:
    repo = repo_template.format(src=src, trg=trg)
    return f"https://huggingface.co/{repo}/resolve/main"


def download(url: str, destination: Path) -> bool:
    """Stream *url* to *destination*.  Returns False when the file is absent."""
    if destination.exists() and destination.stat().st_size > 0:
        print(f"  skip (already present): {destination.name}")
        return True
    try:
        with requests.get(url, stream=True, timeout=60, headers=HEADERS) as response:
            if response.status_code != 200:
                print(f"  unavailable [{response.status_code}]: {url}")
                return False
            total = int(response.headers.get("content-length", 0))
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            with (
                open(temporary, "wb") as handle,
                tqdm(
                    desc=destination.name,
                    total=total,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar,
            ):
                for chunk in response.iter_content(chunk_size=1 << 20):
                    handle.write(chunk)
                    bar.update(len(chunk))
            temporary.rename(destination)
            return True
    except requests.RequestException as error:
        print(f"  download failed: {error}")
        return False


def fetch_firefox(pair: str, work_dir: Path, prefer: str = "int8") -> bool:
    """Download a CTranslate2 model for *pair* into ``work_dir/firefox/<pair>``.

    Each candidate repository is staged separately, so a half finished or
    abandoned variant can never mix files into the final model directory.
    """
    src, trg = pair.split("-")
    final_dir = work_dir / "firefox" / pair
    for index, template in enumerate(FIREFOX_REPOS[prefer]):
        base = firefox_base_url(template, src, trg)
        staging = work_dir / ".staging" / f"{pair}__{index}"
        print(f"[firefox] {pair} <- {template.format(src=src, trg=trg)}")
        complete = True
        for file_name in FIREFOX_FILES:
            if not download(f"{base}/{file_name}", staging / file_name):
                complete = False
                break
        if not complete:
            shutil.rmtree(staging, ignore_errors=True)
            continue
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        if final_dir.exists():
            shutil.rmtree(final_dir, ignore_errors=True)
        staging.rename(final_dir)
        return True
    print(f"[firefox] {pair}: FAILED - no repository serves a complete model")
    return False


def fetch_argos(pair: str, work_dir: Path) -> bool:
    src, trg = pair.split("-")
    # argos names packages translate-<from>_<to>-<version>.argosmodel
    url = f"{ARGOS_BASE_URL}/translate-{src}_{trg}-1_9.argosmodel"
    destination = work_dir / "argos" / f"translate-{src}_{trg}-1_9.argosmodel"
    print(f"[argos] {pair}")
    return download(url, destination)


def build_zip(work_dir: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    # leftover staging trees from an interrupted run must not end up in the zip
    files = sorted(
        p
        for p in work_dir.rglob("*")
        if p.is_file() and ".staging" not in p.relative_to(work_dir).parts
    )
    print(f"Packing {len(files)} files into {output} ...")
    with zipfile.ZipFile(output, "w", zipfile.ZIP_STORED) as archive:
        for path in files:
            archive.write(path, path.relative_to(work_dir).as_posix())
    size_mb = output.stat().st_size / 1e6
    print(f"Done: {output} ({size_mb:.0f} MB)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=DEFAULT_PAIRS,
        help="language pairs such as en-zh zh-en (default: %(default)s)",
    )
    parser.add_argument(
        "--work-dir",
        default="offline-models",
        help="staging directory for the downloaded files",
    )
    parser.add_argument("--out", default="offline-models.zip", help="output zip path")
    parser.add_argument("--skip-firefox", action="store_true")
    parser.add_argument("--skip-argos", action="store_true")
    parser.add_argument("--no-zip", action="store_true", help="only download")
    parser.add_argument(
        "--prefer",
        choices=sorted(FIREFOX_REPOS),
        default="int8",
        help="model selection: int8 (fast on CPU) or quality (largest model)",
    )
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    failures = []

    for pair in args.pairs:
        if not args.skip_firefox and not fetch_firefox(pair, work_dir, args.prefer):
            failures.append(f"firefox {pair}")
        if not args.skip_argos and not fetch_argos(pair, work_dir):
            failures.append(f"argos {pair}")

    if args.no_zip:
        print(f"Files staged in {work_dir}")
    else:
        build_zip(work_dir, Path(args.out))

    if failures:
        print("Failed: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
