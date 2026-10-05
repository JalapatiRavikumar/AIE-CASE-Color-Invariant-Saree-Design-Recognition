"""Tests for the ML core: model forward pass, losses, recoloring, metrics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

# Ensure backend is on path
_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from ml.models.model import build_model, SareeEmbeddingNet, cosine_similarity_matrix
from ml.losses.losses import SupervisedContrastiveLoss
from ml.models.gem import GeM
from ml.augmentations.recolor import RecolorEngine
from ml.augmentations.preprocess import eval_transform, train_transform, load_rgb_image


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def model() -> SareeEmbeddingNet:
    return build_model(backbone="resnet18", embedding_dim=64, pretrained=False)


@pytest.fixture()
def dummy_image() -> Image.Image:
    arr = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def test_model_forward_shape(model):
    x = torch.randn(4, 3, 128, 128)
    with torch.inference_mode():
        z = model(x)
    assert z.shape == (4, 64), f"Expected (4,64), got {z.shape}"


def test_model_l2_normalized(model):
    x = torch.randn(8, 3, 128, 128)
    with torch.inference_mode():
        z = model(x, normalize=True)
    norms = z.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "Embeddings not L2-normalised"


def test_model_unnormalized(model):
    x = torch.randn(4, 3, 128, 128)
    with torch.inference_mode():
        z = model(x, normalize=False)
    norms = z.norm(dim=1)
    # At least some should differ from 1
    assert not torch.allclose(norms, torch.ones_like(norms), atol=1e-2), \
        "Unnormalized embeddings should not all be unit length"


def test_cosine_similarity_matrix():
    a = torch.randn(5, 32)
    b = torch.randn(7, 32)
    sim = cosine_similarity_matrix(a, b)
    assert sim.shape == (5, 7)
    assert (sim.abs() <= 1.0 + 1e-5).all(), "Cosine similarity out of [-1,1]"


# ---------------------------------------------------------------------------
# GeM
# ---------------------------------------------------------------------------


def test_gem_output_shape():
    gem = GeM(p=3.0)
    x = torch.randn(4, 512, 8, 8)
    out = gem(x)
    assert out.shape == (4, 512)


def test_gem_greater_than_avg():
    gem = GeM(p=5.0)
    avg = torch.nn.AdaptiveAvgPool2d(1)
    x = torch.rand(2, 16, 4, 4) + 0.1
    gem_out = gem(x).mean().item()
    avg_out = avg(x).flatten(1).mean().item()
    # GeM with p>1 emphasises high activations → >= avg
    assert gem_out >= avg_out - 1e-4, "GeM(p=5) should be >= avg pooling"


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def test_supcon_loss_basic():
    loss_fn = SupervisedContrastiveLoss(temperature=0.07)
    # 8 samples, 4 classes (2 each) → should yield finite positive loss
    emb = torch.randn(8, 64)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = loss_fn(emb, labels)
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_supcon_loss_no_positives():
    """When every sample is its own class, loss should be zero."""
    loss_fn = SupervisedContrastiveLoss(temperature=0.07)
    emb = torch.randn(6, 32)
    labels = torch.arange(6)
    loss = loss_fn(emb, labels)
    assert loss.item() == 0.0, "No positives → loss must be 0"


def test_supcon_loss_all_same_class():
    loss_fn = SupervisedContrastiveLoss(temperature=0.07)
    emb = torch.randn(4, 32)
    labels = torch.zeros(4, dtype=torch.long)
    loss = loss_fn(emb, labels)
    assert not torch.isnan(loss)
    assert loss.item() >= 0


# ---------------------------------------------------------------------------
# Recolor
# ---------------------------------------------------------------------------


def test_recolor_grayscale(dummy_image):
    engine = RecolorEngine()
    out = engine.grayscale(dummy_image)
    arr = np.asarray(out)
    # All channels should be equal in grayscale
    assert np.allclose(arr[:, :, 0], arr[:, :, 1], atol=2), "R≠G in grayscale"
    assert np.allclose(arr[:, :, 1], arr[:, :, 2], atol=2), "G≠B in grayscale"


def test_recolor_hue_shift_preserves_shape(dummy_image):
    engine = RecolorEngine()
    out = engine.hue_shift(dummy_image, shift=0.5)
    assert out.size == dummy_image.size


def test_recolor_palette_map_preserves_shape(dummy_image):
    engine = RecolorEngine()
    out = engine.palette_map(dummy_image, "red")
    assert out.size == dummy_image.size


def test_recolor_random_colorway_preserves_shape(dummy_image):
    engine = RecolorEngine(rng=np.random.default_rng(0))
    for _ in range(5):
        out = engine.random_colorway(dummy_image)
        assert out.size == dummy_image.size


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------


def test_eval_transform_output_shape(dummy_image):
    tfm = eval_transform(128)
    t = tfm(dummy_image)
    assert t.shape == (3, 128, 128)


def test_train_transform_color_mode_full(dummy_image):
    tfm = train_transform(128, color_mode="full")
    t = tfm(dummy_image)
    assert t.shape == (3, 128, 128)


def test_train_transform_color_mode_none(dummy_image):
    tfm = train_transform(128, color_mode="none")
    t = tfm(dummy_image)
    assert t.shape == (3, 128, 128)


# ---------------------------------------------------------------------------
# End-to-end embedding consistency
# ---------------------------------------------------------------------------


def test_same_image_same_embedding(model, dummy_image):
    """Deterministic inference should produce identical embeddings."""
    tfm = eval_transform(128)
    t = tfm(dummy_image).unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        z1 = model(t)
        z2 = model(t)
    assert torch.allclose(z1, z2, atol=1e-6), "Embedding is not deterministic"


def test_different_images_different_embeddings(model):
    """Two unrelated images should not produce identical embeddings."""
    tfm = eval_transform(128)
    a = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
    b = Image.fromarray(np.full((128, 128, 3), 200, dtype=np.uint8))
    ta = tfm(a).unsqueeze(0)
    tb = tfm(b).unsqueeze(0)
    model.eval()
    with torch.inference_mode():
        za = model(ta)
        zb = model(tb)
    assert not torch.allclose(za, zb, atol=1e-4), "Black and white images should differ"
