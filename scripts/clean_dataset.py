#!/usr/bin/env python3
"""
Dataset Cleaning Script
- Validates all images can be opened
- Removes truly corrupt images (copies to interim/cleaned)
- Reports what was removed
Usage: python scripts/clean_dataset.py [--dry-run]
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "raw" / "handloom_sarees"
CLEANED_DIR = ROOT / "data" / "interim" / "cleaned"


def clean(dry_run: bool = False) -> dict:
    from PIL import Image, UnidentifiedImageError

    print(f"Scanning: {DATASET}")
    imgs = sorted(DATASET.rglob("*.png")) + sorted(DATASET.rglob("*.jpg"))
    print(f"Found {len(imgs)} images\n")

    corrupt = []
    valid = []

    for img_path in imgs:
        try:
            img = Image.open(img_path)
            img.verify()          # full read
            valid.append(img_path)
        except (UnidentifiedImageError, Exception) as e:
            corrupt.append({"path": str(img_path), "error": str(e)})

    print(f"Valid  : {len(valid)}")
    print(f"Corrupt: {len(corrupt)}")

    if corrupt and not dry_run:
        CLEANED_DIR.mkdir(parents=True, exist_ok=True)
        for item in corrupt:
            src = Path(item["path"])
            dst = CLEANED_DIR / src.name
            print(f"  Moving corrupt → {dst.name}")
            shutil.move(str(src), str(dst))
        print(f"Corrupt images quarantined to: {CLEANED_DIR}")
    elif dry_run:
        print("[DRY RUN] No files were moved.")

    report = {
        "total_scanned": len(imgs),
        "valid": len(valid),
        "corrupt": len(corrupt),
        "corrupt_files": corrupt,
        "dry_run": dry_run,
    }

    report_path = ROOT / "results" / "metrics" / "cleaning_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Report only, do not move files")
    args = parser.parse_args()
    clean(dry_run=args.dry_run)
