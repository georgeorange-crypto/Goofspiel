"""Opponent-prediction supervised losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def _ce(logits: Tensor, actual_action: Tensor, legal_mask: Tensor) -> Tensor:
    idx = actual_action.long() - 1
    valid = legal_mask.bool().gather(1, idx[:, None]).squeeze(1)
    if not valid.any():
        return logits.sum() * 0.0
    masked = logits.masked_fill(~legal_mask.bool(), -1e9)
    return F.cross_entropy(masked[valid], idx[valid])


def opponent_prediction_loss(
    short_logits: Tensor,
    long_logits: Tensor,
    fused_logits: Tensor,
    actual_action: Tensor,
    legal_mask: Tensor,
) -> dict[str, Tensor]:
    return {
        "short_nll": _ce(short_logits, actual_action, legal_mask),
        "long_nll": _ce(long_logits, actual_action, legal_mask),
        "fused_nll": _ce(fused_logits, actual_action, legal_mask),
    }
