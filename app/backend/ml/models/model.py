"""Embedding network: pretrained backbone → GeM → projection → L2."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from ml.models.gem import GeM

# ImageNet normalisation constants (also imported by augment.py)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=1, keepdim=True) + eps)


class SareeEmbeddingNet(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        embedding_dim: int = 256,
        pretrained: bool = True,
        gem_p: float = 3.0,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.embedding_dim = embedding_dim
        features, feat_dim = _build_backbone(backbone, pretrained)
        self.features = features
        self.pool = GeM(p=gem_p)
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, embedding_dim),
        )
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        fmap = self.features(x)
        pooled = self.pool(fmap)
        z = self.projection(pooled)
        if normalize:
            z = F.normalize(z, p=2, dim=1)
        return z


def _build_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    weights_resnet = models.ResNet18_Weights.DEFAULT if pretrained else None
    if name == "resnet18":
        net = models.resnet18(weights=weights_resnet)
        feat_dim = net.fc.in_features
        features = nn.Sequential(*list(net.children())[:-2])
        return features, feat_dim
    if name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        net = models.efficientnet_b0(weights=weights)
        feat_dim = net.classifier[1].in_features
        features = net.features
        return features, feat_dim
    if name == "convnext_atto":
        try:
            import timm

            net = timm.create_model("convnext_atto", pretrained=pretrained, num_classes=0, global_pool="")
            feat_dim = int(getattr(net, "num_features"))
            return net, feat_dim
        except Exception as exc:  # pragma: no cover - optional dependency path
            raise RuntimeError(
                "convnext_atto requires the timm package and a successful weight download. "
                "Use BACKBONE=resnet18 as the default fallback."
            ) from exc
    raise ValueError(f"Unsupported backbone: {name}")


def build_model(**kwargs: Any) -> SareeEmbeddingNet:
    return SareeEmbeddingNet(**kwargs)


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    return a @ b.T
