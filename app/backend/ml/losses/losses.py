"""Supervised Contrastive Loss (Khosla et al., NeurIPS 2020). Independent implementation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SupervisedContrastiveLoss(nn.Module):
    """Multi-positive InfoNCE over L2-normalized embeddings.

    For an anchor i with positive set P(i):
        L_i = -1/|P(i)| sum_{p in P(i)} log(
            exp(z_i · z_p / tau) / sum_{a != i} exp(z_i · z_a / tau)
        )
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be [B, D]")
        z = F.normalize(embeddings, p=2, dim=1)
        labels = labels.view(-1)
        sim = torch.matmul(z, z.T) / self.temperature
        logits_mask = torch.ones_like(sim) - torch.eye(sim.size(0), device=sim.device)
        sim = sim - 1e9 * (1.0 - logits_mask)

        label_eq = labels.unsqueeze(0).eq(labels.unsqueeze(1)).float()
        positives = label_eq * logits_mask
        pos_count = positives.sum(dim=1)
        valid = pos_count > 0
        if valid.sum() == 0:
            return embeddings.new_zeros(())

        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        mean_log_prob_pos = (positives * log_prob).sum(dim=1) / pos_count.clamp(min=1.0)
        loss = -mean_log_prob_pos[valid].mean()
        return loss
