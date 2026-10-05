#!/usr/bin/env python3
"""Comprehensive Evaluation Script for Color-Invariant Saree Design Recognition.

Evaluates:
1. Identification (Retrieval):
   - Recall@1, Recall@5, Recall@10, mAP on unseen test set designs
   - Cross-palette retrieval: query colorways (04..07) -> gallery reference colorways (00..03)
   - Outputs: results/metrics/identification.json

2. Verification:
   - Evaluated on structured verification pairs (same design diff color vs diff design same color)
   - ROC-AUC, EER, EER Threshold, Best F1, Best F1 Threshold, Pairs Tested
   - Outputs: results/metrics/verification.json

3. Color Trap Benchmark:
   - Same-Design Different-Color Recall
   - Different-Design Same-Color Rejection Rate (anti-color distractor test)
   - Color Trap Accuracy
   - Outputs: results/metrics/color_trap.json

4. Consolidated Results:
   - results/metrics/latest.json (consumed by FastAPI /api/metrics and React frontend)

Usage:
    python scripts/evaluate.py [--checkpoint models/checkpoints/best_model.pth]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "app" / "backend"
sys.path.insert(0, str(BACKEND))

from core.config import get_settings
from ml.models.model import build_model, SareeEmbeddingNet
from ml.augmentations.preprocess import eval_transform, load_rgb_image

METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


class CsvDataset(Dataset):
    """Load images from a CSV produced by create_splits.py."""

    def __init__(self, csv_path: Path, root: Path, image_size: int = 128) -> None:
        with open(csv_path, encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))
        self.root = root
        self.transform = eval_transform(image_size)

        unique = sorted({r["design_id"] for r in self.rows if r.get("design_id")})
        self.label_map = {v: i for i, v in enumerate(unique)}
        self.labels = [self.label_map.get(r.get("design_id", ""), 0) for r in self.rows]
        self.num_classes = len(unique)
        self.design_ids = [r.get("design_id", "") for r in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = self.root / row.get("image_path", row.get("path", ""))
        image = load_rgb_image(path)
        return self.transform(image), self.labels[index]


def embed_dataset(
    model: SareeEmbeddingNet,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_emb, all_labels = [], []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            emb = model(images.to(device))
            all_emb.append(emb.cpu().numpy())
            all_labels.append(labels.numpy())
    Z = np.vstack(all_emb).astype(np.float32)
    norms = np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8
    Z = Z / norms
    L = np.concatenate(all_labels)
    return Z, L


def recall_at_k(Z: np.ndarray, L: np.ndarray, ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    sim = Z @ Z.T
    np.fill_diagonal(sim, -2.0)
    results = {}
    for k in ks:
        k_ = min(k, sim.shape[1] - 1)
        top_idx = np.argpartition(-sim, k_, axis=1)[:, :k_]
        hits = (L[top_idx] == L[:, None]).any(axis=1).mean()
        results[f"recall@{k}"] = round(float(hits), 4)
    return results


def cross_split_recall(
    Z_query: np.ndarray,
    L_query: np.ndarray,
    Z_gallery: np.ndarray,
    L_gallery: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Color-invariant retrieval: query → gallery (different colorways, same design)."""
    sim = Z_query @ Z_gallery.T  # (n_query, n_gallery)
    results = {}
    for k in ks:
        k_ = min(k, sim.shape[1])
        top_idx = np.argpartition(-sim, k_, axis=1)[:, :k_]
        hits = (L_gallery[top_idx] == L_query[:, None]).any(axis=1).mean()
        results[f"cross_recall@{k}"] = round(float(hits), 4)
    return results


def mean_average_precision(Z: np.ndarray, L: np.ndarray) -> float:
    sim = Z @ Z.T
    np.fill_diagonal(sim, -2.0)
    n = len(L)
    aps = []
    for i in range(n):
        order = np.argsort(-sim[i])
        rel = (L[order] == L[i]).astype(float)
        rel = rel[order != i]
        if rel.sum() == 0:
            continue
        cumsum = np.cumsum(rel)
        prec_at_k = cumsum / (np.arange(len(rel)) + 1)
        ap = (prec_at_k * rel).sum() / rel.sum()
        aps.append(ap)
    return round(float(np.mean(aps)), 4) if aps else 0.0


def evaluate_verification_pairs(
    model: SareeEmbeddingNet,
    pairs_csv: Path,
    device: torch.device,
    image_size: int = 128,
) -> dict:
    """Evaluate verification on structured pairs (calculating ROC-AUC, EER, Best F1)."""
    with open(pairs_csv, encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))

    tfm = eval_transform(image_size)
    cache = {}

    def get_emb(rel_path: str) -> np.ndarray:
        if rel_path in cache:
            return cache[rel_path]
        img = load_rgb_image(PROJECT_ROOT / rel_path)
        tensor = tfm(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            emb = model(tensor).cpu().numpy()[0]
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        cache[rel_path] = emb
        return emb

    sims = []
    labels = []
    pair_types = []

    for p in pairs:
        za = get_emb(p["image_a"])
        zb = get_emb(p["image_b"])
        sim = float(np.dot(za, zb))
        sims.append(sim)
        labels.append(int(p["label"]))
        pair_types.append(p.get("pair_type", "unknown"))

    sims_arr = np.array(sims, dtype=np.float32)
    labels_arr = np.array(labels, dtype=np.int32)

    # Threshold sweep for ROC-AUC, EER, and Best F1
    thresholds = np.linspace(-1.0, 1.0, 401)
    tprs, fprs = [], []
    f1_scores = []

    for thr in thresholds:
        pred = (sims_arr >= thr).astype(int)
        tp = int(((pred == 1) & (labels_arr == 1)).sum())
        fp = int(((pred == 1) & (labels_arr == 0)).sum())
        fn = int(((pred == 0) & (labels_arr == 1)).sum())
        tn = int(((pred == 0) & (labels_arr == 0)).sum())

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        tprs.append(tpr)
        fprs.append(fpr)

        prec = tp / max(tp + fp, 1e-8)
        rec = tp / max(tp + fn, 1e-8)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-8) if (prec + rec) > 0 else 0.0
        f1_scores.append(f1)

    tprs_arr = np.array(tprs)
    fprs_arr = np.array(fprs)
    f1_arr = np.array(f1_scores)

    # ROC-AUC
    try:
        trap_val = float(np.trapezoid(tprs_arr, fprs_arr))
    except AttributeError:
        trap_val = float(np.trapz(tprs_arr, fprs_arr))
    auc = max(0.0, min(1.0, -trap_val))

    # EER
    frrs = 1.0 - tprs_arr
    diffs = np.abs(fprs_arr - frrs)
    eer_idx = int(np.argmin(diffs))
    eer = float((fprs_arr[eer_idx] + frrs[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])

    # Best F1
    best_f1_idx = int(np.argmax(f1_arr))
    best_f1 = float(f1_arr[best_f1_idx])
    best_f1_threshold = float(thresholds[best_f1_idx])

    return {
        "roc_auc": round(auc, 4),
        "eer": round(eer, 4),
        "eer_threshold": round(eer_threshold, 4),
        "best_f1": round(best_f1, 4),
        "best_f1_threshold": round(best_f1_threshold, 4),
        "n_pairs": len(pairs),
        "n_same": int((labels_arr == 1).sum()),
        "n_diff": int((labels_arr == 0).sum()),
    }


def evaluate_color_trap(
    model: SareeEmbeddingNet,
    trap_pairs_csv: Path,
    threshold: float,
    device: torch.device,
    image_size: int = 128,
) -> dict:
    """Evaluate core failure mode: different design with same color vs same design with different color."""
    with open(trap_pairs_csv, encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))

    tfm = eval_transform(image_size)
    cache = {}

    def get_emb(rel_path: str) -> np.ndarray:
        if rel_path in cache:
            return cache[rel_path]
        img = load_rgb_image(PROJECT_ROOT / rel_path)
        tensor = tfm(img).unsqueeze(0).to(device)
        with torch.inference_mode():
            emb = model(tensor).cpu().numpy()[0]
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        cache[rel_path] = emb
        return emb

    pos_correct = 0
    pos_total = 0
    neg_correct = 0
    neg_total = 0

    for p in pairs:
        za = get_emb(p["image_a"])
        zb = get_emb(p["image_b"])
        sim = float(np.dot(za, zb))
        pred_same = (sim >= threshold)
        label = int(p["label"])

        if label == 1:  # Same design, different color
            pos_total += 1
            if pred_same:
                pos_correct += 1
        else:           # Different design, same color (Color Trap)
            neg_total += 1
            if not pred_same:
                neg_correct += 1

    same_design_diff_color_recall = round(pos_correct / max(pos_total, 1), 4)
    diff_design_same_color_rejection = round(neg_correct / max(neg_total, 1), 4)
    color_trap_accuracy = round((pos_correct + neg_correct) / max(pos_total + neg_total, 1), 4)

    return {
        "color_trap_accuracy": color_trap_accuracy,
        "same_design_diff_color_recall": same_design_diff_color_recall,
        "diff_design_same_color_rejection": diff_design_same_color_rejection,
        "threshold_used": round(threshold, 4),
        "total_trap_pairs": len(pairs),
        "positives_tested": pos_total,
        "negatives_tested": neg_total,
        "conclusion": (
            "PASSED: Model correctly relies on weave structure/motifs rather than superficial color matching. "
            "Rejection of same-color distractor motifs is robust."
        ) if color_trap_accuracy >= 0.85 else "Marginal or poor color invariance."
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate saree embedding model")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    settings = get_settings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    ckpt_path = Path(args.checkpoint) if args.checkpoint else settings.checkpoint_path
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        sys.exit(1)

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = payload.get("meta", {})
    backbone = meta.get("backbone", settings.backbone)
    dim = int(meta.get("embedding_dim", settings.embedding_dim))
    image_size = int(meta.get("image_size", args.image_size))

    model = build_model(backbone=backbone, embedding_dim=dim, pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    print(f"Loaded: {ckpt_path} ({backbone}, dim={dim})")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. IDENTIFICATION EVALUATION (Test Split)
    print("\n" + "=" * 60)
    print("1. EVALUATING IDENTIFICATION (RETRIEVAL)")
    print("=" * 60)

    test_csv = PROCESSED_DIR / "test.csv"
    gallery_csv = PROCESSED_DIR / "gallery.csv"
    query_csv = PROCESSED_DIR / "query.csv"

    test_ds = CsvDataset(test_csv, PROJECT_ROOT, image_size)
    print(f"Test split images: {len(test_ds)} ({test_ds.num_classes} designs)")
    Z_test, L_test = embed_dataset(model, test_ds, device, args.batch_size)

    id_recall = recall_at_k(Z_test, L_test)
    id_map = mean_average_precision(Z_test, L_test)

    # Cross-palette evaluation: query (colorways 4..7) -> gallery (colorways 0..3)
    gallery_ds = CsvDataset(gallery_csv, PROJECT_ROOT, image_size)
    query_ds = CsvDataset(query_csv, PROJECT_ROOT, image_size)
    Z_gal, L_gal = embed_dataset(model, gallery_ds, device, args.batch_size)
    Z_qry, L_qry = embed_dataset(model, query_ds, device, args.batch_size)

    cross_recall = cross_split_recall(Z_qry, L_qry, Z_gal, L_gal)

    identification_results = {
        "recall@1": id_recall["recall@1"],
        "recall@5": id_recall["recall@5"],
        "recall@10": id_recall["recall@10"],
        "mAP": id_map,
        "cross_palette_recall@1": cross_recall["cross_recall@1"],
        "cross_palette_recall@5": cross_recall["cross_recall@5"],
        "cross_palette_recall@10": cross_recall["cross_recall@10"],
        "gallery_size": len(gallery_ds),
        "query_size": len(query_ds),
        "test_images": len(test_ds),
        "test_designs": test_ds.num_classes,
    }
    (METRICS_DIR / "identification.json").write_text(json.dumps(identification_results, indent=2), encoding="utf-8")

    print(f"  Recall@1  : {identification_results['recall@1']*100:.1f}%")
    print(f"  Recall@5  : {identification_results['recall@5']*100:.1f}%")
    print(f"  Recall@10 : {identification_results['recall@10']*100:.1f}%")
    print(f"  mAP       : {identification_results['mAP']:.4f}")
    print(f"  Cross R@1 : {identification_results['cross_palette_recall@1']*100:.1f}%")

    # 2. VERIFICATION EVALUATION
    print("\n" + "=" * 60)
    print("2. EVALUATING PAIRWISE VERIFICATION")
    print("=" * 60)

    verif_csv = PROCESSED_DIR / "verification_pairs.csv"
    verification_results = evaluate_verification_pairs(model, verif_csv, device, image_size)
    (METRICS_DIR / "verification.json").write_text(json.dumps(verification_results, indent=2), encoding="utf-8")

    print(f"  ROC-AUC      : {verification_results['roc_auc']:.4f}")
    print(f"  EER          : {verification_results['eer']*100:.1f}%")
    print(f"  EER Threshold: {verification_results['eer_threshold']:.4f}")
    print(f"  Best F1      : {verification_results['best_f1']*100:.1f}%")
    print(f"  F1 Threshold : {verification_results['best_f1_threshold']:.4f}")
    print(f"  Pairs Tested : {verification_results['n_pairs']:,}")

    # 3. COLOR TRAP BENCHMARK
    print("\n" + "=" * 60)
    print("3. EVALUATING COLOR TRAP BENCHMARK")
    print("=" * 60)

    trap_csv = PROCESSED_DIR / "color_trap_pairs.csv"
    color_trap_results = evaluate_color_trap(model, trap_csv, verification_results["eer_threshold"], device, image_size)
    (METRICS_DIR / "color_trap.json").write_text(json.dumps(color_trap_results, indent=2), encoding="utf-8")

    print(f"  Color Trap Accuracy       : {color_trap_results['color_trap_accuracy']*100:.1f}%")
    print(f"  Same-Design Diff-Color Rec: {color_trap_results['same_design_diff_color_recall']*100:.1f}%")
    print(f"  Diff-Design Same-Color Rej: {color_trap_results['diff_design_same_color_rejection']*100:.1f}%")

    # 4. LOAD COLORWAY BENCHMARK RESULTS IF AVAILABLE
    colorway_json_path = METRICS_DIR / "colorway.json"
    colorway_data = {}
    if colorway_json_path.exists():
        colorway_data = json.loads(colorway_json_path.read_text(encoding="utf-8"))

    # 5. CONSOLIDATED LATEST.JSON FOR API / UI
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    latest_report = {
        "checkpoint": str(ckpt_path),
        "mode": meta.get("mode", "trained"),
        "backbone": backbone,
        "embedding_dim": dim,
        "color_mode": meta.get("color_mode", "full"),
        "split": "test",
        "n_images": len(test_ds),
        "n_classes": test_ds.num_classes,
        "identification": {
            "recall@1": identification_results["recall@1"],
            "recall@5": identification_results["recall@5"],
            "recall@10": identification_results["recall@10"],
            "mAP": identification_results["mAP"],
        },
        "verification": verification_results,
        "color_invariance": {
            "cross_recall@1": cross_recall["cross_recall@1"],
            "cross_recall@5": cross_recall["cross_recall@5"],
            "cross_recall@10": cross_recall["cross_recall@10"],
            "color_trap_accuracy": color_trap_results["color_trap_accuracy"],
            "same_design_diff_color_recall": color_trap_results["same_design_diff_color_recall"],
            "diff_design_same_color_rejection": color_trap_results["diff_design_same_color_rejection"],
            "mean_perturbed_recall@1": colorway_data.get("mean_perturbed_recall@1"),
            "mean_perturbed_recall@5": colorway_data.get("mean_perturbed_recall@5"),
            "mean_perturbed_recall@10": colorway_data.get("mean_perturbed_recall@10"),
            "mean_perturbed_mAP": colorway_data.get("mean_perturbed_mAP"),
        },
        "efficiency": {
            "n_params_total": n_params,
            "n_params_trainable": n_trainable,
            "embedding_size_bytes": dim * 4,
        }
    }

    latest_path = METRICS_DIR / "latest.json"
    latest_path.write_text(json.dumps(latest_report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("ALL EVALUATIONS COMPLETE")
    print(f"Consolidated metrics saved to: {latest_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
