"""Player-swap structural losses."""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def q_swap_symmetry_loss(q: Tensor, q_swapped: Tensor, joint_mask: Tensor) -> Tensor:
    err = (q + q_swapped.transpose(-1, -2)) * joint_mask.float()
    return (err.square().sum(dim=(-1, -2)) / joint_mask.float().sum(dim=(-1, -2)).clamp_min(1.0)).mean()


def distribution_symmetry_loss(logits: Tensor, swapped_logits: Tensor) -> Tensor:
    target = F.softmax(logits.detach().flip(-1), dim=-1)
    logp = F.log_softmax(swapped_logits, dim=-1)
    return F.kl_div(logp, target, reduction="batchmean")
