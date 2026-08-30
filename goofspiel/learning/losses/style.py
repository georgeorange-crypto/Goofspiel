"""Opponent style contrastive loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def opponent_style_infonce(
    embeddings: Tensor,
    opponent_ids: Tensor,
    regime_ids: Tensor,
    temperature: float = 0.1,
) -> Tensor:
    z = F.normalize(embeddings.float(), dim=-1)
    logits = z @ z.T / max(temperature, 1e-6)
    same = (opponent_ids[:, None] == opponent_ids[None, :]) & (regime_ids[:, None] == regime_ids[None, :])
    eye = torch.eye(z.shape[0], device=z.device, dtype=torch.bool)
    positives = same & ~eye
    if not positives.any():
        return embeddings.sum() * 0.0
    logits = logits.masked_fill(eye, -1e9)
    logp = F.log_softmax(logits, dim=-1)
    denom = positives.sum(dim=-1).clamp_min(1)
    per_row = -(logp * positives.float()).sum(dim=-1) / denom
    valid = positives.any(dim=-1)
    return per_row[valid].mean()
