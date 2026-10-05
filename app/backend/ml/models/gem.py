"""Generalized Mean (GeM) pooling.

Radenović, Tolias, Chum. Fine-tuning CNN Image Retrieval with No Human
Annotation. TPAMI 2018. Independent reimplementation of the pooling operator.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class GeM(nn.Module):
    """y = (1/|X| * sum_i x_i^p )^(1/p) with learnable p > 1."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learn_p: bool = True) -> None:
        super().__init__()
        self.eps = eps
        if learn_p:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer("p", torch.ones(1) * p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: B x C x H x W
        x = x.clamp(min=self.eps)
        gem = F.avg_pool2d(x.pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)
        return gem.flatten(1)
