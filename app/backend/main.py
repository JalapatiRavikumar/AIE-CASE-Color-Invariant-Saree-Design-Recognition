"""FastAPI backend — Color-Invariant Saree Design Recognition."""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

# Ensure app/backend is on sys.path (uvicorn runs from this dir)
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from core.config import get_settings
from core.logging import get_logger
from ml.inference.matcher import SareeMatcher, MatcherError

settings = get_settings()
logger = get_logger("api")
_start_time = time.time()

app = FastAPI(
    title="Color-Invariant Saree Design Recognition",
    description=(
        "Identifies handloom saree weaving designs (Ikat, Jamdani, Banarasi, Patola, "
        "Kanjivaram, Chanderi) independent of colour palette using metric learning."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

matcher = SareeMatcher(settings=settings)
MAX_BYTES = settings.max_upload_mb * 1024 * 1024


def _read_image(upload: UploadFile) -> Image.Image:
    data = upload.file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB limit")
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, Exception) as exc:
        raise HTTPException(400, f"Cannot decode image: {exc}") from exc


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok", "uptime_seconds": round(time.time() - _start_time, 1)}


@app.get("/", tags=["Health"])
def root():
    return {
        "project": settings.project_name,
        "version": "1.0.0",
        "status": "running",
        "uptime_seconds": round(time.time() - _start_time, 1),
        "docs": "/docs",
    }


# ── System ────────────────────────────────────────────────────────────────────

# ── System ────────────────────────────────────────────────────────────────────

@app.get("/api/status", tags=["System"])
@app.get("/status", tags=["System"])
def api_status():
    """Model mode, device, gallery size, checkpoint info."""
    return matcher.status()


@app.get("/api/metrics", tags=["System"])
@app.get("/metrics", tags=["System"])
def api_metrics():
    """Latest evaluation metrics (if evaluate.py has been run)."""
    path = settings.metrics_path
    if not path.exists():
        return {"available": False, "message": "Run scripts/evaluate.py to generate metrics"}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"available": True, **data}


# ── Images & Downloads ────────────────────────────────────────────────────────

@app.get("/api/image/{image_path:path}", tags=["Images"])
def api_serve_image(image_path: str):
    """Serve dataset images for previews."""
    clean_path = image_path.replace("\\", "/")
    full_path = settings.project_root / clean_path
    if not full_path.exists() or not full_path.is_file():
        raise HTTPException(404, f"Image '{image_path}' not found")
    return FileResponse(full_path)


@app.get("/api/downloads", tags=["Downloads"])
def api_downloads():
    """List downloadable artifacts."""
    dl_dir = settings.project_root / "downloads"
    dl_dir.mkdir(parents=True, exist_ok=True)
    files = [
        {"name": p.name, "size_bytes": p.stat().st_size}
        for p in sorted(dl_dir.iterdir())
        if p.is_file()
    ]
    return {"files": files}


@app.get("/api/download/{filename}", tags=["Downloads"])
def api_download_file(filename: str):
    """Download a file from downloads directory."""
    dl_dir = settings.project_root / "downloads"
    file_path = dl_dir / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, f"File '{filename}' not found")
    return FileResponse(file_path, filename=filename)


# ── Gallery ───────────────────────────────────────────────────────────────────

@app.post("/api/index", tags=["Gallery"])
def api_index(gallery_path: str = Form(...)):
    """Index all images in a server-side folder as the active gallery."""
    try:
        return matcher.index(gallery_path)
    except MatcherError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("Indexing failed")
        raise HTTPException(500, str(exc)) from exc


# ── Inference ─────────────────────────────────────────────────────────────────

@app.post("/api/identify", tags=["Inference"])
@app.post("/identify", tags=["Inference"])
def api_identify(
    file: Annotated[UploadFile, File(description="Query saree image")],
    top_k: int = 5,
):
    """Return top-K gallery matches ranked by cosine similarity."""
    image = _read_image(file)
    try:
        hits = matcher.identify(image, top_k=top_k)
    except MatcherError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "top_k": top_k,
        "mode": matcher.mode,
        "gallery_total": len(matcher.gallery_names),
        "results": [
            {
                "rank": h.rank,
                "image": h.filename,
                "path": h.path,
                "image_url": f"/api/image/{h.path.replace(chr(92), '/')}",
                "similarity": round(h.similarity, 4),
                "design_id": h.design_id,
                "class": h.tradition or (h.filename.split("_")[0] if "_" in h.filename else "unknown"),
            }
            for h in hits
        ],
    }


@app.post("/api/search", tags=["Inference"])
@app.post("/search", tags=["Inference"])
def api_search(
    file: Annotated[UploadFile, File(description="Query saree image")],
    top_k: int = 5,
):
    """Alias for /api/identify — retrieval search interface."""
    return api_identify(file=file, top_k=top_k)


@app.post("/api/verify", tags=["Inference"])
@app.post("/verify", tags=["Inference"])
def api_verify(
    file_a: Annotated[UploadFile, File(description="First saree image")],
    file_b: Annotated[UploadFile, File(description="Second saree image")],
    threshold: Optional[float] = None,
):
    """Decide whether two images carry the same design."""
    img_a = _read_image(file_a)
    img_b = _read_image(file_b)
    return matcher.verify(img_a, img_b, threshold=threshold)


@app.post("/api/embed", tags=["Inference"])
@app.post("/embed", tags=["Inference"])
def api_embed(file: Annotated[UploadFile, File(description="Saree image to embed")]):
    """Return the L2-normalised 256-dim embedding vector."""
    image = _read_image(file)
    vec = matcher.embed_pil(image)
    return {"embedding": vec.tolist(), "dim": int(vec.shape[0])}


@app.post("/api/predict", tags=["Inference"])
@app.post("/predict", tags=["Inference"])
def api_predict(
    file: Annotated[UploadFile, File(description="Query saree image")],
    top_k: int = 5,
):
    """Primary prediction endpoint — returns top-K matching designs."""
    return api_identify(file=file, top_k=top_k)


# ── Startup ───────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Auto-load gallery embeddings from models/embeddings/ on startup."""
    project_root = settings.project_root
    # Try new-format split embeddings first
    emb_path = project_root / "models" / "embeddings" / "gallery_embeddings.npy"
    meta_path = project_root / "models" / "embeddings" / "gallery_metadata.json"

    # Fallback to legacy index_cache format
    legacy_emb = settings.embeddings_path
    legacy_cache = settings.gallery_dir / "index_cache.json"

    if emb_path.exists() and meta_path.exists():
        try:
            emb = np.load(str(emb_path))
            with open(meta_path) as f:
                metadata = json.load(f)
            matcher.gallery_emb = emb.astype(np.float32)
            matcher.gallery_names = [Path(m["image_path"]).name for m in metadata]
            matcher.gallery_paths = [m["image_path"].replace("\\", "/") for m in metadata]
            # Store design metadata for richer responses
            matcher._gallery_meta = metadata
            logger.info("Loaded gallery: %d images (from models/embeddings/)", len(matcher.gallery_names))
        except Exception as exc:
            logger.warning("Could not load gallery embeddings: %s", exc)
    elif legacy_emb.exists() and legacy_cache.exists():
        try:
            cache = json.loads(legacy_cache.read_text(encoding="utf-8"))
            emb = np.load(str(legacy_emb))
            matcher.gallery_emb = emb.astype(np.float32)
            matcher.gallery_names = cache.get("names", [])
            matcher.gallery_paths = cache.get("paths", [])
            logger.info("Loaded gallery (legacy): %d images", len(matcher.gallery_names))
        except Exception as exc:
            logger.warning("Could not load legacy gallery: %s", exc)
    else:
        logger.warning("No gallery embeddings found. Run: python scripts/generate_embeddings.py")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.api_host, port=settings.api_port, reload=True)
