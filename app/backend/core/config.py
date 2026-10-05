"""Centralized configuration — loads from split YAML files + environment."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


def _project_root() -> Path:
    env = os.getenv("PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    # app/backend/core/config.py -> project root is parents[3]
    here = Path(__file__).resolve()
    return here.parents[3]


def _load_yamls(root: Path) -> dict[str, Any]:
    """Merge all config/*.yaml files into one dict."""
    merged: dict[str, Any] = {}
    for fname in ["base.yaml", "data.yaml", "model.yaml", "training.yaml", "evaluation.yaml"]:
        path = root / "config" / fname
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            merged.update(data)
    return merged


class Settings(BaseModel):
    project_name: str = "Color-Invariant Saree Design Recognition"
    project_root: Path
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    cors_origins: list[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"]
    )
    max_upload_mb: int = 8
    dataset_raw: Path
    dataset_processed: Path
    gallery_dir: Path
    model_dir: Path
    results_dir: Path
    image_size: int = 224
    embedding_dim: int = 256
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    temperature: float = 0.07
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    backbone: str = "resnet18"
    gem_p: float = 3.0
    freeze_backbone: bool = False
    verification_threshold: float | None = None

    @property
    def checkpoint_path(self) -> Path:
        return self.model_dir / "best_model.pth"

    @property
    def embeddings_path(self) -> Path:
        return self.project_root / "models" / "embeddings" / "gallery_embeddings.npy"

    @property
    def metrics_path(self) -> Path:
        return self.results_dir / "metrics" / "latest.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    root = _project_root()
    env_file = root / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

    yml = _load_yamls(root)
    api = yml.get("api", {})
    paths = yml.get("paths", {})
    model = yml.get("model", {})
    train = yml.get("train", {})

    def pjoin(key: str, default: str) -> Path:
        rel = os.getenv(key.upper() + "_DIR", paths.get(key, default))
        path = Path(rel)
        return path if path.is_absolute() else (root / path)

    cors = os.getenv("API_CORS_ORIGINS")
    cors_list = (
        [c.strip() for c in cors.split(",") if c.strip()]
        if cors
        else list(api.get("cors_origins", ["http://localhost:5173", "http://127.0.0.1:5173"]))
    )
    threshold_raw = os.getenv("VERIFICATION_THRESHOLD", "")
    threshold = float(threshold_raw) if threshold_raw.strip() else None

    return Settings(
        project_name=yml.get("project", {}).get("name", "Color-Invariant Saree Design Recognition"),
        project_root=root,
        api_host=os.getenv("API_HOST", str(api.get("host", "0.0.0.0"))),
        api_port=int(os.getenv("API_PORT", api.get("port", 8000))),
        cors_origins=cors_list,
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", api.get("max_upload_mb", 8))),
        dataset_raw=pjoin("dataset_raw", "data/raw/handloom_sarees"),
        dataset_processed=pjoin("dataset_processed", "data/processed"),
        gallery_dir=pjoin("gallery", "models/embeddings"),
        model_dir=pjoin("models", "models/checkpoints"),
        results_dir=pjoin("results", "results"),
        image_size=int(os.getenv("IMAGE_SIZE", train.get("image_size", 224))),
        embedding_dim=int(os.getenv("EMBEDDING_DIM", model.get("embedding_dim", 256))),
        batch_size=int(os.getenv("BATCH_SIZE", train.get("batch_size", 32))),
        epochs=int(os.getenv("EPOCHS", train.get("epochs", 20))),
        learning_rate=float(os.getenv("LEARNING_RATE", train.get("learning_rate", 1e-4))),
        weight_decay=float(os.getenv("WEIGHT_DECAY", train.get("weight_decay", 1e-4))),
        temperature=float(os.getenv("TEMPERATURE", train.get("temperature", 0.07))),
        num_workers=int(os.getenv("NUM_WORKERS", train.get("num_workers", 0))),
        seed=int(os.getenv("SEED", train.get("seed", 42))),
        device=os.getenv("DEVICE", "auto"),
        backbone=os.getenv("BACKBONE", model.get("backbone", "resnet18")),
        gem_p=float(model.get("gem_p", 3.0)),
        freeze_backbone=bool(model.get("freeze_backbone", False)),
        verification_threshold=threshold,
    )
