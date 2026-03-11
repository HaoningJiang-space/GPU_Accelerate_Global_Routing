#!/usr/bin/env python3
"""
scripts/download_ispd2024_benchmarks.py
Download ISPD2024 global routing benchmarks from Google Drive.

Usage:
    python scripts/download_ispd2024_benchmarks.py [--dest benchmarks/ispd2024]

Requires: gdown >= 5.0  (pip install gdown)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

FOLDER_URL = "https://drive.google.com/drive/folders/1bon65UEAx8cjSvVhYJ-lgC8QMDX0fvUm"
REPO_ROOT   = Path(__file__).resolve().parent.parent
DEFAULT_DEST = REPO_ROOT / "benchmarks" / "ispd2024"


def check_gdown():
    try:
        import gdown
        return gdown
    except ImportError:
        sys.exit("gdown not found. Install with: pip install gdown")


def download(dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    print(f"[download] Target folder: {dest}")
    print(f"[download] Source: {FOLDER_URL}")
    print("[download] Starting download (this may take a while)...\n")

    import gdown
    gdown.download_folder(
        url=FOLDER_URL,
        output=str(dest),
        quiet=False,
        use_cookies=False,
    )


def update_manifest(dest: Path):
    """Scan downloaded .cap/.net pairs and print manifest entries to add."""
    cap_files = sorted(dest.glob("*.cap"))
    if not cap_files:
        print("[manifest] No .cap files found — skipping manifest update hint.")
        return

    repo = REPO_ROOT
    print("\n[manifest] Add the following entries to benchmarks/local_suite_manifest.yaml:\n")
    for cap in cap_files:
        stem = cap.stem
        net  = cap.with_suffix(".net")
        if not net.exists():
            continue
        cap_rel = cap.relative_to(repo)
        net_rel = net.relative_to(repo)
        out_rel = Path("results") / "baseline" / stem / "route.out"
        # Guess tier by file size
        size_mb = cap.stat().st_size / 1e6
        tier = "small" if size_mb < 50 else "medium"
        print(f"  - name: {stem}")
        print(f"    tier: {tier}")
        print(f"    cap:  {cap_rel}")
        print(f"    net:  {net_rel}")
        print(f"    out:  {out_rel}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Download ISPD2024 benchmarks from Google Drive.")
    parser.add_argument("--dest", default=str(DEFAULT_DEST),
                        help=f"Download destination (default: {DEFAULT_DEST})")
    args = parser.parse_args()

    check_gdown()
    dest = Path(args.dest)
    download(dest)
    update_manifest(dest)
    print("[download] Done.")


if __name__ == "__main__":
    main()
