"""Inference engine with optional FAISS."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from core.config import Settings, get_settings
from ml.models.model import SareeEmbeddingNet, build_model
from ml.augmentations.preprocess import eval_transform, load_rgb_image


class MatcherError(Exception):
    pass


@dataclass
class IdentifyHit:
    rank: int
    filename: str
    path: str
    similarity: float
    design_id: str = ""
    tradition: str = ""


class SareeMatcher:
    def __init__(self, settings: Settings | None = None, device: str | None = None) -> None:
        self.settings = settings or get_settings()
        self.device = torch.device(device or _resolve_device(self.settings.device))
        self.model: SareeEmbeddingNet | None = None
        self.mode = "unloaded"
        self.meta: dict = {}
        self.gallery_names: list[str] = []
        self.gallery_paths: list[str] = []
        self._gallery_meta: list[dict] = []
        self.gallery_emb: np.ndarray | None = None
        self._faiss_index = None
        self._load_model()

    def _load_model(self) -> None:
        ckpt = self.settings.checkpoint_path
        meta_path = self.settings.model_dir / "checkpoint_meta.json"
        if ckpt.exists():
            payload = torch.load(ckpt, map_location=self.device, weights_only=False)
            self.meta = payload.get("meta", {})
            if meta_path.exists():
                self.meta.update(json.loads(meta_path.read_text(encoding="utf-8")))
            backbone = self.meta.get("backbone", self.settings.backbone)
            dim = int(self.meta.get("embedding_dim", self.settings.embedding_dim))
            self.model = build_model(
                backbone=backbone,
                embedding_dim=dim,
                pretrained=False,
                gem_p=float(self.meta.get("gem_p", self.settings.gem_p)),
            )
            self.model.load_state_dict(payload["state_dict"])
            self.mode = self.meta.get("mode", "trained")
        else:
            self.model = build_model(
                backbone=self.settings.backbone,
                embedding_dim=self.settings.embedding_dim,
                pretrained=True,
                gem_p=self.settings.gem_p,
                freeze_backbone=True,
            )
            self.mode = "demo"
            self.meta = {
                "backbone": self.settings.backbone,
                "embedding_dim": self.settings.embedding_dim,
                "mode": "demo",
                "note": "DEMO / PRETRAINED MODE — ImageNet encoder without metric-learning fine-tuning.",
            }
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def embed_pil(self, image: Image.Image, grayscale: bool = False) -> np.ndarray:
        if self.model is None:
            raise MatcherError("Model is not loaded")
        tfm = eval_transform(self.settings.image_size, grayscale=grayscale)
        tensor = tfm(image.convert("RGB")).unsqueeze(0).to(self.device)
        vec = self.model(tensor).cpu().numpy()[0]
        return vec.astype(np.float32)

    def embed_path(self, path: Path) -> np.ndarray:
        return self.embed_pil(load_rgb_image(path))

    def index(self, gallery_folder: str | Path) -> dict:
        folder = Path(gallery_folder)
        if not folder.exists():
            raise MatcherError(f"Gallery folder not found: {folder}")
        paths = [
            p
            for p in sorted(folder.rglob("*"))
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        ]
        if not paths:
            raise MatcherError("Gallery folder contains no supported images")
        embs = []
        names = []
        stored = []
        for path in paths:
            try:
                embs.append(self.embed_path(path))
                names.append(path.name)
                stored.append(str(path))
            except Exception:
                continue
        if not embs:
            raise MatcherError("No gallery images could be embedded")
        self.gallery_emb = np.stack(embs, axis=0)
        self.gallery_names = names
        self.gallery_paths = stored
        self._faiss_index = None
        try:
            import faiss  # type: ignore

            index = faiss.IndexFlatIP(self.gallery_emb.shape[1])
            faiss.normalize_L2(self.gallery_emb)
            index.add(self.gallery_emb)
            self._faiss_index = index
        except Exception:
            self._faiss_index = None
        cache = {
            "names": self.gallery_names,
            "paths": self.gallery_paths,
            "embeddings": self.gallery_emb.tolist(),
            "count": len(self.gallery_names),
            "faiss": bool(self._faiss_index is not None),
        }
        cache_path = self.settings.gallery_dir / "index_cache.json"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({"count": cache["count"], "names": names, "paths": stored}), encoding="utf-8")
        np.save(str(self.settings.embeddings_path), self.gallery_emb)
        return {"count": len(names), "faiss": cache["faiss"], "files": names}

    def identify(self, query_image: Image.Image, top_k: int = 5) -> list[IdentifyHit]:
        if self.gallery_emb is None or not len(self.gallery_names):
            raise MatcherError("Gallery is empty. Index a folder first.")
        q = self.embed_pil(query_image)
        q = q / (np.linalg.norm(q) + 1e-8)
        gallery = self.gallery_emb / (np.linalg.norm(self.gallery_emb, axis=1, keepdims=True) + 1e-8)
        sims = gallery @ q
        order = np.argsort(-sims)[:top_k]
        hits = []
        for rank, idx in enumerate(order, start=1):
            idx_int = int(idx)
            meta_item = self._gallery_meta[idx_int] if idx_int < len(self._gallery_meta) else {}
            hits.append(
                IdentifyHit(
                    rank=rank,
                    filename=self.gallery_names[idx_int],
                    path=self.gallery_paths[idx_int],
                    similarity=float(sims[idx_int]),
                    design_id=meta_item.get("design_id", ""),
                    tradition=meta_item.get("tradition", ""),
                )
            )
        return hits

    def verify(
        self,
        image_a: Image.Image,
        image_b: Image.Image,
        threshold: float | None = None,
    ) -> dict:
        za = self.embed_pil(image_a)
        zb = self.embed_pil(image_b)
        sim = float(np.dot(za, zb) / ((np.linalg.norm(za) * np.linalg.norm(zb)) + 1e-8))
        thr = threshold
        if thr is None:
            thr = self.settings.verification_threshold
        if thr is None:
            thr = float(self.meta.get("verification_threshold", 0.90))

        is_same = bool(sim >= thr)
        # Calibrated logistic confidence based on distance to threshold
        margin = sim - thr
        prob = 1.0 / (1.0 + np.exp(-14.0 * margin))
        confidence = float(prob if is_same else (1.0 - prob))
        confidence = max(0.50, min(0.999, confidence))

        return {
            "same_design": is_same,
            "similarity": round(sim, 4),
            "threshold": round(float(thr), 4),
            "confidence": round(confidence, 4),
            "mode": self.mode,
        }

    def status(self) -> dict:
        return {
            "mode": self.mode,
            "device": str(self.device),
            "backbone": self.meta.get("backbone", self.settings.backbone),
            "embedding_dim": int(self.meta.get("embedding_dim", self.settings.embedding_dim)),
            "checkpoint": str(self.settings.checkpoint_path) if self.settings.checkpoint_path.exists() else None,
            "gallery_count": len(self.gallery_names),
            "faiss": self._faiss_index is not None,
            "meta": self.meta,
        }


def _resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name
