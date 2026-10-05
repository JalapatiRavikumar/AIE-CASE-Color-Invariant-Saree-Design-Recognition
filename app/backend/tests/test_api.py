"""Tests for the FastAPI endpoints."""
from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Add backend to path
_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from fastapi.testclient import TestClient


def _make_png_bytes(color=(100, 150, 200), size=(128, 128)) -> bytes:
    img = Image.fromarray(np.full((*size, 3), color, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    from main import app
    return TestClient(app)


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "project" in data
    assert "status" in data


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_status(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert "mode" in data
    assert "backbone" in data
    assert "embedding_dim" in data


def test_embed_endpoint(client):
    png = _make_png_bytes()
    r = client.post("/api/embed", files={"file": ("test.png", png, "image/png")})
    assert r.status_code == 200
    data = r.json()
    assert "embedding" in data
    assert "dim" in data
    assert isinstance(data["embedding"], list)
    assert len(data["embedding"]) == data["dim"]


def test_verify_same_image(client):
    png = _make_png_bytes()
    r = client.post(
        "/api/verify",
        files={
            "file_a": ("a.png", png, "image/png"),
            "file_b": ("b.png", png, "image/png"),
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "same_design" in data
    assert "similarity" in data
    assert data["similarity"] > 0.99, "Same image should have near-1 cosine similarity"


def test_verify_different_images(client):
    png_a = _make_png_bytes(color=(0, 0, 0))
    png_b = _make_png_bytes(color=(255, 255, 255))
    r = client.post(
        "/api/verify",
        files={
            "file_a": ("a.png", png_a, "image/png"),
            "file_b": ("b.png", png_b, "image/png"),
        },
    )
    assert r.status_code == 200


def test_identify_no_gallery(client):
    """Should return 400 when gallery is empty."""
    png = _make_png_bytes()
    r = client.post("/api/identify", files={"file": ("test.png", png, "image/png")})
    assert r.status_code == 400


def test_metrics_endpoint(client):
    r = client.get("/api/metrics")
    assert r.status_code == 200
    data = r.json()
    assert "available" in data


def test_downloads_list(client):
    r = client.get("/api/downloads")
    assert r.status_code == 200
    assert "files" in r.json()


def test_bad_file_returns_400(client):
    r = client.post(
        "/api/embed",
        files={"file": ("bad.txt", b"not an image", "text/plain")},
    )
    assert r.status_code == 400
