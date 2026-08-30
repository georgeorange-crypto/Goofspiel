"""Distributional outcome projection."""

from __future__ import annotations

import torch
from torch import Tensor


def project_score_difference_two_hot(final_score_diff_normalized: Tensor, num_bins: int = 201) -> Tensor:
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2")
    x = final_score_diff_normalized.float().clamp(-1.0, 1.0)
    pos = (x + 1.0) * (num_bins - 1) / 2.0
    lo = torch.floor(pos).long().clamp(0, num_bins - 1)
    hi = torch.ceil(pos).long().clamp(0, num_bins - 1)
    whi = (pos - lo.float()).clamp(0.0, 1.0)
    wlo = 1.0 - whi
    out = torch.zeros(*x.shape, num_bins, device=x.device, dtype=torch.float32)
    out.scatter_add_(-1, lo[..., None], wlo[..., None])
    out.scatter_add_(-1, hi[..., None], whi[..., None])
    return out
