#!/usr/bin/env python3
"""Controlled Colorway Benchmark for Color-Invariant Saree Design Recognition.

Evaluates whether the same saree DESIGN matches across controlled color perturbations:
1. Hue shift (±0.25, ±0.4)
2. Saturation scaling (0.3x desaturated, 1.8x saturated)
3. Brightness scaling (0.6x darker, 1.4x brighter)
4. Contrast scaling (0.6x lower contrast, 1.5x higher contrast)
5. Color Jitter (combined)
6. Grayscale (complete color removal, structure-only)
7. Cross-palette test (procedural colorways: query palette_04..07 -> gallery palette_00..03)

Outputs: results/metrics/colorway.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "app" / "backend"
sys.path.insert(0, str(BACKEND))

from core.config import get_settings
from ml.models.model import build_model
from ml.augmentations.preprocess import eval_transform, load_rgb_image

PROCESSED = ROOT / "data" / "processed"
OUTPUT_PATH = ROOT / "results" / "metrics" / "colorway.json"


def evaluate_retrieval(query_embs: np.ndarray, query_labels: list[str],
                       gallery_embs: np.ndarray, gallery_labels: list[str],
                       ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    """Calculate Recall@K and mAP between query and gallery embeddings."""
    # Cosine similarities: (N_query, N_gallery)
    sims = query_embs @ gallery_embs.T
    n_queries = len(query_labels)
    g_labels = np.array(gallery_labels)

    recalls = {k: 0 for k in ks}
    aps = []

    for i in range(n_queries):
        target = query_labels[i]
        ranked_indices = np.argsort(-sims[i])
        ranked_labels = g_labels[ranked_indices]
        matches = (ranked_labels == target)

        for k in ks:
            if matches[:min(k, len(matches))].any():
                recalls[k] += 1

        if matches.sum() > 0:
            rel = matches.astype(float)
            cumsum = np.cumsum(rel)
            prec_at_k = cumsum / (np.arange(len(rel)) + 1)
            ap = (prec_at_k * rel).sum() / rel.sum()
            aps.append(ap)
        else:
            aps.append(0.0)

    result = {f"recall@{k}": round(recalls[k] / max(n_queries, 1), 4) for k in ks}
    result["mAP"] = round(float(np.mean(aps)), 4) if aps else 0.0
    return result


def run_colorway_benchmark(image_size: int = 128, batch_size: int = 32) -> dict:
    settings = get_settings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    ckpt_path = ROOT / "models" / "checkpoints" / "best_model.pth"
    if not ckpt_path.exists():
        print(f"ERROR: No checkpoint found at {ckpt_path}")
        sys.exit(1)

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = payload.get("meta", {})
    backbone = meta.get("backbone", "resnet18")
    dim = int(meta.get("embedding_dim", 256))

    model = build_model(backbone=backbone, embedding_dim=dim, pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    print(f"Loaded: {ckpt_path.name} ({backbone}, dim={dim})")

    # Load gallery
    gallery_csv = PROCESSED / "gallery.csv"
    query_csv = PROCESSED / "query.csv"
    if not gallery_csv.exists() or not query_csv.exists():
        print("ERROR: gallery.csv or query.csv missing. Run scripts/create_splits.py")
        sys.exit(1)

    with open(gallery_csv, encoding="utf-8") as f:
        gallery_rows = list(csv.DictReader(f))
    with open(query_csv, encoding="utf-8") as f:
        query_rows = list(csv.DictReader(f))

    print(f"Gallery images: {len(gallery_rows)}")
    print(f"Query images  : {len(query_rows)}")

    base_tfm = eval_transform(image_size)

    # 1. Embed Gallery
    print("\nEmbedding gallery reference images...")
    gallery_embs = []
    gallery_labels = []
    for r in gallery_rows:
        img = load_rgb_image(ROOT / r["image_path"])
        tensor = base_tfm(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            emb = model(tensor).cpu().numpy()[0]
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        gallery_embs.append(emb)
        gallery_labels.append(r["design_id"])
    gallery_matrix = np.stack(gallery_embs, axis=0)

    # Define controlled color transformations
    perturbations = {
        "original_cross_palette": lambda img: img,
        "hue_shift_pos": lambda img: TF.adjust_hue(img, 0.3),
        "hue_shift_neg": lambda img: TF.adjust_hue(img, -0.3),
        "desaturated_0.3x": lambda img: TF.adjust_saturation(img, 0.3),
        "oversaturated_1.8x": lambda img: TF.adjust_saturation(img, 1.8),
        "darkened_0.6x": lambda img: TF.adjust_brightness(img, 0.6),
        "brightened_1.4x": lambda img: TF.adjust_brightness(img, 1.4),
        "low_contrast_0.6x": lambda img: TF.adjust_contrast(img, 0.6),
        "high_contrast_1.5x": lambda img: TF.adjust_contrast(img, 1.5),
        "color_jitter_combined": lambda img: TF.adjust_hue(TF.adjust_saturation(TF.adjust_brightness(img, 0.8), 1.5), 0.2),
        "grayscale": lambda img: img.convert("L").convert("RGB"),
    }

    results = {}
    print("\nRunning controlled color perturbations against reference gallery:")
    print("-" * 65)

    for p_name, p_fn in perturbations.items():
        q_embs = []
        q_labels = []
        for r in query_rows:
            raw_img = load_rgb_image(ROOT / r["image_path"])
            perturbed = p_fn(raw_img)
            tensor = base_tfm(perturbed).unsqueeze(0).to(device)
            with torch.inference_mode():
                emb = model(tensor).cpu().numpy()[0]
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            q_embs.append(emb)
            q_labels.append(r["design_id"])

        q_matrix = np.stack(q_embs, axis=0)
        metrics = evaluate_retrieval(q_matrix, q_labels, gallery_matrix, gallery_labels)
        results[p_name] = metrics
        print(f"  {p_name:<25}: R@1={metrics['recall@1']*100:.1f}%, R@5={metrics['recall@5']*100:.1f}%, R@10={metrics['recall@10']*100:.1f}%, mAP={metrics['mAP']:.4f}")

    # Summary aggregations
    r1_scores = [v["recall@1"] for k, v in results.items() if k != "original_cross_palette"]
    r5_scores = [v["recall@5"] for k, v in results.items() if k != "original_cross_palette"]
    r10_scores = [v["recall@10"] for k, v in results.items() if k != "original_cross_palette"]
    map_scores = [v["mAP"] for k, v in results.items() if k != "original_cross_palette"]

    benchmark_summary = {
        "cross_palette_baseline": results["original_cross_palette"],
        "mean_perturbed_recall@1": round(float(np.mean(r1_scores)), 4),
        "min_perturbed_recall@1": round(float(np.min(r1_scores)), 4),
        "mean_perturbed_recall@5": round(float(np.mean(r5_scores)), 4),
        "mean_perturbed_recall@10": round(float(np.mean(r10_scores)), 4),
        "mean_perturbed_mAP": round(float(np.mean(map_scores)), 4),
        "perturbation_breakdown": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(benchmark_summary, indent=2), encoding="utf-8")
    print("-" * 65)
    print(f"Benchmark summary saved to: {OUTPUT_PATH}")
    print(f"Mean Perturbed Recall@1 : {benchmark_summary['mean_perturbed_recall@1']*100:.1f}%")
    print(f"Mean Perturbed Recall@5 : {benchmark_summary['mean_perturbed_recall@5']*100:.1f}%")
    print(f"Mean Perturbed Recall@10: {benchmark_summary['mean_perturbed_recall@10']*100:.1f}%")
    print(f"Mean Perturbed mAP      : {benchmark_summary['mean_perturbed_mAP']:.4f}")
    return benchmark_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    run_colorway_benchmark(image_size=args.image_size, batch_size=args.batch_size)
