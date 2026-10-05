"""Dataset discovery, cleaning, hashing, and leak-safe splits."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = False

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class ManifestRow:
    image_id: str
    path: str
    source: str
    width: int
    height: int
    md5: str
    phash: str
    instance_id: str
    design_id: str
    style_label: str
    split: str
    license: str
    attribution: str
    notes: str


def file_md5(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            data = handle.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def perceptual_hash(image: Image.Image, hash_size: int = 8) -> str:
    """Simple DCT-free average hash (aHash), sufficient for near-duplicate screening."""
    gray = image.convert("L").resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    arr = np.asarray(gray, dtype=np.float32)
    bits = arr > arr.mean()
    value = 0
    for bit in bits.flatten():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def discover_images(root: Path) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            files.append(path)
    return sorted(files)


def inspect_image(path: Path, min_side: int) -> dict[str, Any] | None:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            rgb = im.convert("RGB")
            width, height = rgb.size
            if min(width, height) < min_side:
                return None
            phash = perceptual_hash(rgb)
            md5 = file_md5(path)
            return {
                "width": width,
                "height": height,
                "phash": phash,
                "md5": md5,
            }
    except (UnidentifiedImageError, OSError, ValueError):
        return None


def build_manifest(
    raw_root: Path,
    processed_dir: Path,
    min_side: int = 64,
    phash_threshold: int = 8,
    seed: int = 42,
    train: float = 0.7,
    val: float = 0.1,
) -> dict[str, Any]:
    processed_dir.mkdir(parents=True, exist_ok=True)
    discovered = discover_images(raw_root)
    rows: list[ManifestRow] = []
    removed: list[dict[str, str]] = []
    seen_md5: dict[str, str] = {}
    kept_meta: list[dict[str, Any]] = []

    for path in discovered:
        rel = str(path.relative_to(raw_root)).replace("\\", "/")
        source = "synthetic" if rel.startswith("synthetic/") else "wikimedia"
        if source == "synthetic":
            source = "synthetic"
        elif rel.startswith("local/"):
            source = "local_upload"
        else:
            source = "wikimedia"

        meta = inspect_image(path, min_side=min_side)
        if meta is None:
            removed.append({"path": rel, "reason": "corrupt_unreadable_or_too_small"})
            continue
        if meta["md5"] in seen_md5:
            removed.append({"path": rel, "reason": f"exact_duplicate_of_{seen_md5[meta['md5']]}"})
            continue
        seen_md5[meta["md5"]] = rel
        design_id = ""
        style_label = ""
        license_name = ""
        attribution = ""
        sidecar = path.with_suffix(path.suffix + ".json")
        if not sidecar.exists():
            sidecar = path.with_suffix(".json")
        if sidecar.exists():
            info = json.loads(sidecar.read_text(encoding="utf-8"))
            design_id = str(info.get("design_id", "") or "")
            style_label = str(info.get("style_label", "") or "")
            license_name = str(info.get("license", "") or "")
            attribution = str(info.get("attribution", "") or "")
            source = str(info.get("source", source) or source)
        instance_id = meta["md5"]
        if design_id:
            group_for_near = design_id
        else:
            group_for_near = instance_id
        kept_meta.append(
            {
                "rel": rel,
                "source": source,
                "design_id": design_id,
                "style_label": style_label,
                "license": license_name,
                "attribution": attribution,
                "instance_id": instance_id,
                "group_for_near": group_for_near,
                **meta,
            }
        )

    near_dup_count = 0
    drop_near: set[str] = set()
    for i, a in enumerate(kept_meta):
        if a["rel"] in drop_near:
            continue
        for b in kept_meta[i + 1 :]:
            if b["rel"] in drop_near:
                continue
            if a["group_for_near"] == b["group_for_near"] and a["design_id"]:
                continue
            dist = hamming(a["phash"], b["phash"])
            if dist <= phash_threshold:
                drop_near.add(b["rel"])
                near_dup_count += 1
                removed.append({"path": b["rel"], "reason": f"near_duplicate_of_{a['rel']}_hamming_{dist}"})

    kept_meta = [m for m in kept_meta if m["rel"] not in drop_near]

    # Group-level split: real design_id if present, else instance_id.
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in kept_meta:
        gid = item["design_id"] if item["design_id"] else item["instance_id"]
        groups[gid].append(item)

    rng = np.random.default_rng(seed)
    group_ids = list(groups.keys())
    rng.shuffle(group_ids)
    n = len(group_ids)
    n_train = int(round(train * n))
    n_val = int(round(val * n))
    split_map: dict[str, str] = {}
    for i, gid in enumerate(group_ids):
        if i < n_train:
            split_map[gid] = "train"
        elif i < n_train + n_val:
            split_map[gid] = "val"
        else:
            split_map[gid] = "test"

    for item in kept_meta:
        gid = item["design_id"] if item["design_id"] else item["instance_id"]
        rows.append(
            ManifestRow(
                image_id=item["md5"][:12],
                path=item["rel"],
                source=item["source"],
                width=item["width"],
                height=item["height"],
                md5=item["md5"],
                phash=item["phash"],
                instance_id=item["instance_id"],
                design_id=item["design_id"],
                style_label=item["style_label"],
                split=split_map[gid],
                license=item["license"],
                attribution=item["attribution"],
                notes="design_id from sidecar only; never inferred",
            )
        )

    manifest_path = processed_dir / "manifest.csv"
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(ManifestRow.__dataclass_fields__.keys())
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    widths = [r.width for r in rows]
    heights = [r.height for r in rows]
    design_ids = {r.design_id for r in rows if r.design_id}
    style_labels = {r.style_label for r in rows if r.style_label}
    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "discovered": len(discovered),
        "valid": len(rows),
        "removed": len(removed),
        "exact_duplicates": sum(1 for r in removed if r["reason"].startswith("exact_duplicate")),
        "near_duplicates": near_dup_count,
        "design_id_count": len(design_ids),
        "style_label_count": len(style_labels),
        "sources": _count([r.source for r in rows]),
        "splits": _count([r.split for r in rows]),
        "resolution": {
            "width_mean": float(np.mean(widths)) if widths else 0,
            "height_mean": float(np.mean(heights)) if heights else 0,
            "width_min": int(min(widths)) if widths else 0,
            "width_max": int(max(widths)) if widths else 0,
            "height_min": int(min(heights)) if heights else 0,
            "height_max": int(max(heights)) if heights else 0,
        },
        "disclosure": (
            "DeepLure did not provide a proprietary dataset. Images are public Wikimedia Commons "
            "photographs and/or project-generated synthetic layouts. design_id is present only when "
            "a sidecar explicitly provides it (synthetic set). Wikimedia photos use instance-level grouping."
        ),
    }
    (processed_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    (processed_dir / "removed.json").write_text(json.dumps(removed, indent=2), encoding="utf-8")
    return {"manifest": str(manifest_path), "stats": stats, "rows": [asdict(r) for r in rows]}


def _count(values: list[str]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for value in values:
        out[value] += 1
    return dict(out)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
