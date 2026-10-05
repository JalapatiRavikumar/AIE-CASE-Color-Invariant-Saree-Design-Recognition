"""ML package for Color-Invariant Saree Design Recognition."""
from ml.models.model import build_model, SareeEmbeddingNet, cosine_similarity_matrix
from ml.models.gem import GeM
from ml.losses.losses import SupervisedContrastiveLoss
from ml.augmentations.recolor import RecolorEngine
from ml.augmentations.preprocess import eval_transform, train_transform, load_rgb_image

__all__ = [
    "build_model",
    "SareeEmbeddingNet",
    "cosine_similarity_matrix",
    "GeM",
    "SupervisedContrastiveLoss",
    "RecolorEngine",
    "eval_transform",
    "train_transform",
    "load_rgb_image",
]
