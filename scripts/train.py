"""Training script: supervised contrastive learning on saree images.

Usage (from project root):
    python scripts/train.py [OPTIONS]

Key options:
    --backbone      resnet18 | efficientnet_b0  (default: resnet18)
    --epochs        int  (default: from config)
    --batch-size    int
    --lr            float
    --image-size    int  (default: 128)
    --color-mode    none | standard | recolor | full  (default: full)
    --output-dir    path to save checkpoints
    --data-dir      path to image root (discovers all images recursively)
    --resume        path to checkpoint to resume from
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, Dataset

# Make sure app/backend is on the path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND = PROJECT_ROOT / "app" / "backend"
sys.path.insert(0, str(BACKEND))

from core.config import get_settings
from ml.models.model import build_model, SareeEmbeddingNet
from ml.losses.losses import SupervisedContrastiveLoss
from ml.augmentations.preprocess import train_transform, eval_transform, load_rgb_image
from ml.training.dataset import build_manifest, read_manifest, discover_images

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class SareeDataset(Dataset):
    """Dataset of (image, label) pairs from a manifest CSV.

    The label here is the design_id index (tradition-level grouping).
    This supports supervised contrastive learning where same design_id
    across different colorways are treated as positives.
    """

    def __init__(
        self,
        rows: list[dict],
        root: Path,
        transform=None,
        split: str | None = None,
    ) -> None:
        if split:
            rows = [r for r in rows if r["split"] == split]
        self.rows = rows
        self.root = root
        self.transform = transform

        # Use design_id as the label (the tradition+index, e.g. banarasi_0000)
        unique = sorted({r["design_id"] for r in rows if r.get("design_id")})
        if not unique:
            # Fallback to tradition as label
            unique = sorted({r.get("tradition", r["path"].split("/")[0]) for r in rows})
        self.label_map = {v: i for i, v in enumerate(unique)}

        self.labels = []
        for r in rows:
            did = r.get("design_id", "")
            if did and did in self.label_map:
                self.labels.append(self.label_map[did])
            else:
                trad = r.get("tradition", r["path"].split("/")[0])
                self.labels.append(self.label_map.get(trad, 0))

        self.num_classes = len(unique)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = self.root / row["path"]
        image = load_rgb_image(path)
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index]


class FlatFolderDataset(Dataset):
    """Fallback dataset: discovers images directly from raw folder structure.
    
    Folder structure: {tradition}_{design_id}/palette_XX.png
    Label = design folder name (e.g. 'banarasi_0000') for contrastive learning.
    """

    def __init__(self, root: Path, transform=None) -> None:
        paths = sorted(
            p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS
        )
        if not paths:
            raise RuntimeError(f"No images found in {root}")
        # design_id folder name = label (e.g. banarasi_0000)
        design_ids = sorted({p.parent.name for p in paths})
        design_to_id = {d: i for i, d in enumerate(design_ids)}
        self.paths = paths
        self.labels = [design_to_id[p.parent.name] for p in paths]
        self.transform = transform
        self.num_classes = len(design_ids)
        print(f"FlatFolderDataset: {len(paths)} images, {len(design_ids)} design classes")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        image = load_rgb_image(self.paths[index])
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index]


# ---------------------------------------------------------------------------
# Balanced sampler
# ---------------------------------------------------------------------------

class BalancedBatchSampler:
    """Yield batches with P classes × K images/class.

    Classes with fewer than K images are oversampled.
    For color-invariant learning: each design_id class contains 8 colorways.
    """

    def __init__(
        self,
        labels: list[int],
        p_classes: int = 8,
        k_per_class: int = 4,
        n_batches: int = 50,
    ) -> None:
        self.p = p_classes
        self.k = k_per_class
        self.n_batches = n_batches
        self.class_to_idx: dict[int, list[int]] = defaultdict(list)
        for idx, lbl in enumerate(labels):
            self.class_to_idx[lbl].append(idx)
        self.classes = list(self.class_to_idx.keys())

    def __iter__(self):
        for _ in range(self.n_batches):
            classes = random.sample(
                self.classes, min(self.p, len(self.classes))
            )
            batch = []
            for cls in classes:
                idxs = self.class_to_idx[cls]
                chosen = random.choices(idxs, k=self.k)
                batch.extend(chosen)
            yield batch

    def __len__(self) -> int:
        return self.n_batches


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate_recall(
    model: SareeEmbeddingNet,
    dataset: Dataset,
    device: torch.device,
    batch_size: int = 64,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    """Compute Recall@K on the given split."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    all_emb, all_labels = [], []
    model.eval()
    for images, labels in loader:
        emb = model(images.to(device))
        all_emb.append(emb.cpu())
        all_labels.append(labels)
    Z = F.normalize(torch.cat(all_emb), dim=1)
    L = torch.cat(all_labels)
    # cosine similarity matrix
    sim = Z @ Z.T
    # mask self
    sim.fill_diagonal_(-2.0)
    results: dict[str, float] = {}
    for k in ks:
        topk_idx = sim.topk(min(k, sim.size(1)), dim=1).indices
        hits = L[topk_idx].eq(L.unsqueeze(1)).any(dim=1).float().mean().item()
        results[f"recall@{k}"] = round(hits, 4)
    return results


def train_one_epoch(
    model: SareeEmbeddingNet,
    loader: DataLoader,
    criterion: SupervisedContrastiveLoss,
    optimizer: optim.Optimizer,
    device: torch.device,
    scaler,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=scaler is not None):
            emb = model(images, normalize=False)
            loss = criterion(emb, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        total_loss += loss.item()
        steps += 1
    return total_loss / max(steps, 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Color-Invariant Saree Embedding Model")
    p.add_argument("--backbone", default="resnet18", choices=["resnet18", "efficientnet_b0"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--color-mode", default="full", choices=["none", "standard", "recolor", "full"])
    p.add_argument("--data-dir", type=str, default=None, help="Image root directory")
    p.add_argument("--output-dir", type=str, default=None, help="Checkpoint output directory")
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    p.add_argument("--p-classes", type=int, default=8, help="Classes per batch")
    p.add_argument("--k-per-class", type=int, default=4, help="Images per class per batch")
    p.add_argument("--n-batches", type=int, default=100, help="Batches per epoch")
    p.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    set_seed(args.seed)

    # Resolve paths
    data_dir = Path(args.data_dir) if args.data_dir else settings.dataset_raw
    output_dir = Path(args.output_dir) if args.output_dir else settings.model_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = args.epochs or settings.epochs
    batch_size = args.batch_size or settings.batch_size
    lr = args.lr or settings.learning_rate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Backbone: {args.backbone}, dim={args.embedding_dim}, color_mode={args.color_mode}")

    # ---------------------------------------------------------------------------
    # Dataset — prefer create_splits.py CSVs, fall back to flat folder
    # ---------------------------------------------------------------------------
    train_csv = PROJECT_ROOT / "data" / "processed" / "train.csv"
    val_csv = PROJECT_ROOT / "data" / "processed" / "validation.csv"
    train_ds, val_ds = None, None

    if train_csv.exists() and val_csv.exists():
        print("Loading from train.csv / validation.csv ...")
        import csv
        with open(train_csv, encoding="utf-8") as f:
            train_rows = list(csv.DictReader(f))
        with open(val_csv, encoding="utf-8") as f:
            val_rows = list(csv.DictReader(f))
        # Remap path key: create_splits.py uses 'image_path'
        for r in train_rows + val_rows:
            if "image_path" in r and "path" not in r:
                r["path"] = r["image_path"]
        tfm_train = train_transform(args.image_size, color_mode=args.color_mode)
        tfm_eval = eval_transform(args.image_size)
        train_ds = SareeDataset(train_rows, PROJECT_ROOT, transform=tfm_train)
        val_ds = SareeDataset(val_rows, PROJECT_ROOT, transform=tfm_eval)
        print(f"  Train: {len(train_ds)} images, {train_ds.num_classes} design classes")
        print(f"  Val:   {len(val_ds)} images, {val_ds.num_classes} design classes")
    else:
        print("No split CSVs found — building from flat folder structure...")
        print("Run 'python scripts/create_splits.py' first for proper splits.")
        tfm_train = train_transform(args.image_size, color_mode=args.color_mode)
        train_ds = FlatFolderDataset(data_dir, transform=tfm_train)
        val_ds = None

    if train_ds is None or len(train_ds) == 0:
        raise RuntimeError("No training images found. Check data directory.")

    # Sampler
    sampler = BalancedBatchSampler(
        train_ds.labels,
        p_classes=args.p_classes,
        k_per_class=args.k_per_class,
        n_batches=args.n_batches,
    )
    loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=settings.num_workers,
        pin_memory=device.type == "cuda",
    )

    # ---------------------------------------------------------------------------
    # Model, loss, optimiser
    # ---------------------------------------------------------------------------
    model = build_model(
        backbone=args.backbone,
        embedding_dim=args.embedding_dim,
        pretrained=True,
        gem_p=settings.gem_p,
    ).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"Resumed from {args.resume}")

    # Count params
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,} total, {n_trainable:,} trainable")

    criterion = SupervisedContrastiveLoss(temperature=args.temperature)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=settings.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    use_amp = not args.no_amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    print(f"Mixed precision: {use_amp}")

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------
    best_recall = -1.0
    patience_counter = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, loader, criterion, optimizer, device, scaler)
        scheduler.step()

        row: dict = {"epoch": epoch, "train_loss": round(train_loss, 4)}

        if val_ds is not None and len(val_ds) > 0:
            val_recall = evaluate_recall(model, val_ds, device, batch_size=batch_size)
            row.update(val_recall)
            r1 = val_recall.get("recall@1", 0.0)
            print(
                f"Epoch {epoch:3d}/{epochs}  loss={train_loss:.4f}  "
                f"R@1={r1:.4f}  R@5={val_recall.get('recall@5',0):.4f}  "
                f"time={time.time()-t0:.1f}s"
            )
            if r1 > best_recall:
                best_recall = r1
                patience_counter = 0
                _save_checkpoint(model, args, epoch, output_dir, is_best=True)
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        else:
            print(f"Epoch {epoch:3d}/{epochs}  loss={train_loss:.4f}  time={time.time()-t0:.1f}s")
            _save_checkpoint(model, args, epoch, output_dir, is_best=(epoch == epochs))

        # Always save last
        _save_checkpoint(model, args, epoch, output_dir, is_best=False, suffix="last")
        history.append(row)

    # Save history
    hist_path = output_dir / "training_history.json"
    hist_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\nTraining complete. Best Recall@1: {best_recall:.4f}")
    print(f"Checkpoint: {output_dir / 'best_model.pth'}")
    print(f"History:    {hist_path}")


def _save_checkpoint(
    model: SareeEmbeddingNet,
    args: argparse.Namespace,
    epoch: int,
    output_dir: Path,
    is_best: bool,
    suffix: str = "best",
) -> None:
    payload = {
        "state_dict": model.state_dict(),
        "epoch": epoch,
        "meta": {
            "backbone": args.backbone,
            "embedding_dim": args.embedding_dim,
            "image_size": args.image_size,
            "temperature": args.temperature,
            "color_mode": args.color_mode,
            "mode": "trained",
        },
    }
    name = "best_model.pth" if is_best else f"model_{suffix}.pth"
    path = output_dir / name
    torch.save(payload, str(path))
    if is_best:
        meta = payload["meta"]
        meta_path = output_dir / "checkpoint_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
