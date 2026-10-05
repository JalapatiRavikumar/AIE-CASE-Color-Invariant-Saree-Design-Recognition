#!/usr/bin/env python3
"""
Generate gallery embeddings from trained model.
Embeds all gallery images and saves to models/embeddings/.

Usage:
    python scripts/generate_embeddings.py [--split gallery] [--batch-size 32]
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app" / "backend"))

from core.config import get_settings
from ml.models.model import build_model
from ml.augmentations.preprocess import eval_transform


def generate_embeddings(split: str = "gallery", batch_size: int = 32) -> None:
    settings = get_settings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    ckpt_path = ROOT / "models" / "checkpoints" / "best_model.pth"
    if not ckpt_path.exists():
        print(f"ERROR: No checkpoint at {ckpt_path}")
        print("Train the model first: python scripts/train.py")
        sys.exit(1)

    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = payload.get("meta", {})
    backbone = meta.get("backbone", "resnet18")
    dim = int(meta.get("embedding_dim", 256))

    model = build_model(backbone=backbone, embedding_dim=dim, pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    print(f"Loaded: {ckpt_path.name}  ({backbone}, dim={dim})")

    # Load split CSV
    csv_path = ROOT / "data" / "processed" / f"{split}.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run: python scripts/create_splits.py")
        sys.exit(1)

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Embedding {len(rows)} images from '{split}' split...")

    tfm = eval_transform(settings.image_size)
    embeddings = []
    metadata = []
    failed = []

    for i, row in enumerate(rows):
        img_path = ROOT / row["image_path"]
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = tfm(img).unsqueeze(0).to(device)
            with torch.inference_mode():
                emb = model(tensor).cpu().numpy()[0]
            # L2-normalize
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            embeddings.append(emb)
            metadata.append({
                "image_path": row["image_path"],
                "design_id": row["design_id"],
                "tradition": row["tradition"],
                "colorway": row["colorway"],
            })
        except Exception as e:
            failed.append({"path": str(img_path), "error": str(e)})

        if (i + 1) % 100 == 0 or (i + 1) == len(rows):
            print(f"  {i + 1}/{len(rows)}")

    emb_matrix = np.stack(embeddings, axis=0).astype(np.float32)
    out_dir = ROOT / "models" / "embeddings"
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = out_dir / f"{split}_embeddings.npy"
    meta_path = out_dir / f"{split}_metadata.json"

    np.save(str(emb_path), emb_matrix)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved embeddings : {emb_path}  shape={emb_matrix.shape}")
    print(f"Saved metadata   : {meta_path}")
    if failed:
        print(f"Failed           : {len(failed)} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="gallery", choices=["gallery", "query", "train", "validation", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    generate_embeddings(split=args.split, batch_size=args.batch_size)
