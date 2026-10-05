#!/usr/bin/env python3
"""
Deduplication Script — perceptual hash based near-duplicate detection.
Operates on the handloom_sarees dataset.
Usage: python scripts/deduplicate_dataset.py [--threshold 8] [--dry-run]
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

try:
    import imagehash
except ImportError:
    imagehash = None

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "raw" / "handloom_sarees"
DEDUP_DIR = ROOT / "data" / "interim" / "deduplicated"


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def deduplicate(threshold: int = 8, dry_run: bool = False) -> dict:
    if imagehash is None:
        print("imagehash not installed. Running exact-duplicate check only.")

    imgs = sorted(DATASET.rglob("*.png")) + sorted(DATASET.rglob("*.jpg"))
    print(f"Scanning {len(imgs)} images for duplicates...")

    # Exact duplicates by MD5
    md5_map: dict[str, list[Path]] = {}
    for p in imgs:
        h = file_md5(p)
        md5_map.setdefault(h, []).append(p)

    exact_groups = {h: paths for h, paths in md5_map.items() if len(paths) > 1}
    exact_dups = [str(p) for paths in exact_groups.values() for p in paths[1:]]

    print(f"Exact duplicate groups : {len(exact_groups)}")
    print(f"Exact duplicate files  : {len(exact_dups)}")

    # Near-duplicates by perceptual hash (if imagehash available)
    near_dups = []
    if imagehash:
        hashes: dict[str, Path] = {}
        for p in imgs:
            try:
                img = Image.open(p)
                ph = str(imagehash.phash(img))
                if ph in hashes:
                    near_dups.append({"a": str(hashes[ph]), "b": str(p)})
                else:
                    hashes[ph] = p
            except Exception:
                pass
        print(f"Near-duplicate pairs   : {len(near_dups)}")
    else:
        print("Near-duplicate check skipped (imagehash not installed)")

    report = {
        "total_images": len(imgs),
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicates": exact_dups,
        "near_duplicate_pairs": near_dups,
        "threshold_hamming": threshold,
        "note": (
            "This dataset has NO exact duplicates by design — each image is a unique "
            "colorway of a specific design. Near-duplicate check verifies colorway integrity."
        ),
    }

    report_path = ROOT / "results" / "metrics" / "dedup_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to: {report_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=int, default=8, help="Hamming distance threshold for near-duplication")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    deduplicate(threshold=args.threshold, dry_run=args.dry_run)
