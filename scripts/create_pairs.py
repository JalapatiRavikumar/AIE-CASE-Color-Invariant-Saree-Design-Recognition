#!/usr/bin/env python3
"""Create structured verification and color-trap test pairs.

Generates:
1. data/processed/verification_pairs.csv
   - Positive pairs: Same design, DIFFERENT colorways (palette_i != palette_j)
   - Negative pairs (Hard / Color Trap): Different designs, SAME colorway (palette_k == palette_k)
   - Negative pairs (Easy / General): Different designs, different colorways
2. data/processed/color_trap_pairs.csv
   - Specifically evaluates the core failure case:
     * Same design, different color -> should match
     * Different design, same color -> should NOT match

Usage:
    python scripts/create_pairs.py [--split test] [--max-pairs 5000]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def create_pairs(split: str = "test", max_pairs: int = 5000, seed: int = 42) -> dict:
    random.seed(seed)
    csv_path = PROCESSED / f"{split}.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run scripts/create_splits.py first.")
        sys.exit(1)

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} images from {csv_path.name}")

    # Group by design_id and colorway
    by_design = defaultdict(list)
    by_colorway = defaultdict(list)
    for r in rows:
        by_design[r["design_id"]].append(r)
        by_colorway[r["colorway"]].append(r)

    unique_designs = sorted(by_design.keys())
    unique_colorways = sorted(by_colorway.keys())
    print(f"Unique designs: {len(unique_designs)}, Unique colorways: {len(unique_colorways)}")

    positive_pairs = []
    # 1. Positive pairs: Same design, different colorways
    for did, d_rows in by_design.items():
        for r_a, r_b in combinations(d_rows, 2):
            if r_a["colorway"] != r_b["colorway"]:
                positive_pairs.append({
                    "image_a": r_a["image_path"],
                    "image_b": r_b["image_path"],
                    "design_a": r_a["design_id"],
                    "design_b": r_b["design_id"],
                    "colorway_a": r_a["colorway"],
                    "colorway_b": r_b["colorway"],
                    "label": 1,
                    "pair_type": "same_design_diff_color",
                })

    # 2. Hard Negative pairs (Color Trap): Different design, EXACT SAME colorway
    color_trap_negatives = []
    for cw, cw_rows in by_colorway.items():
        for r_a, r_b in combinations(cw_rows, 2):
            if r_a["design_id"] != r_b["design_id"]:
                color_trap_negatives.append({
                    "image_a": r_a["image_path"],
                    "image_b": r_b["image_path"],
                    "design_a": r_a["design_id"],
                    "design_b": r_b["design_id"],
                    "colorway_a": r_a["colorway"],
                    "colorway_b": r_b["colorway"],
                    "label": 0,
                    "pair_type": "diff_design_same_color_trap",
                })

    # 3. General Negative pairs: Different design, different colorway
    general_negatives = []
    for _ in range(len(positive_pairs)):
        d_a, d_b = random.sample(unique_designs, 2)
        r_a = random.choice(by_design[d_a])
        r_b = random.choice(by_design[d_b])
        if r_a["colorway"] != r_b["colorway"]:
            general_negatives.append({
                "image_a": r_a["image_path"],
                "image_b": r_b["image_path"],
                "design_a": r_a["design_id"],
                "design_b": r_b["design_id"],
                "colorway_a": r_a["colorway"],
                "colorway_b": r_b["colorway"],
                "label": 0,
                "pair_type": "diff_design_diff_color",
            })

    random.shuffle(positive_pairs)
    random.shuffle(color_trap_negatives)
    random.shuffle(general_negatives)

    # Subsample if necessary while keeping balance
    n_pos = min(len(positive_pairs), max_pairs // 2)
    pos_sample = positive_pairs[:n_pos]
    n_trap = min(len(color_trap_negatives), n_pos // 2)
    n_gen = n_pos - n_trap

    neg_sample = color_trap_negatives[:n_trap] + general_negatives[:n_gen]
    all_verification_pairs = pos_sample + neg_sample
    random.shuffle(all_verification_pairs)

    # Color trap benchmark pairs: balanced (same design diff color vs diff design same color)
    trap_count = min(len(positive_pairs), len(color_trap_negatives), 1000)
    color_trap_pairs = positive_pairs[:trap_count] + color_trap_negatives[:trap_count]
    random.shuffle(color_trap_pairs)

    def write_csv(data: list[dict], path: Path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)
        print(f"Written: {path.name} ({len(data)} pairs)")

    verif_out = PROCESSED / "verification_pairs.csv"
    trap_out = PROCESSED / "color_trap_pairs.csv"
    write_csv(all_verification_pairs, verif_out)
    write_csv(color_trap_pairs, trap_out)

    meta = {
        "split": split,
        "total_verification_pairs": len(all_verification_pairs),
        "positive_pairs": len(pos_sample),
        "negative_pairs": len(neg_sample),
        "color_trap_negatives_in_verification": n_trap,
        "color_trap_benchmark_pairs": len(color_trap_pairs),
        "color_trap_positives": trap_count,
        "color_trap_negatives": trap_count,
    }
    meta_path = PROCESSED / "pairs_summary.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nPairs Summary saved to: {meta_path}")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-pairs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    create_pairs(split=args.split, max_pairs=args.max_pairs, seed=args.seed)
