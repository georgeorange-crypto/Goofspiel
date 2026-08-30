"""NeuRD-style robust actor losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def row_action_regret(q_matrix: Tensor, row_policy: Tensor, col_policy: Tensor, row_mask: Tensor) -> Tensor:
    q = q_matrix.detach().float()
    row = row_policy.float() * row_mask.float()
    row = row / row.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    col = col_policy.float()
    col = col / col.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    action_value = torch.bmm(q, col[:, :, None]).squeeze(-1)
    value = (row * action_value).sum(dim=-1, keepdim=True)
    return (action_value - value) * row_mask.float()


def neurd_loss(
    logits_self: Tensor,
    logits_opp: Tensor,
    q_matrix: Tensor,
    self_mask: Tensor,
    opp_mask: Tensor,
) -> Tensor:
    """NeuRD loss on raw logits, not log-softmax policy gradient."""
    col_policy = F.softmax(logits_opp.masked_fill(~opp_mask.bool(), -1e9), dim=-1)
    del col_policy
    action_value = q_matrix.detach().float().masked_fill(~opp_mask.bool().unsqueeze(1), -1e9).max(dim=-1).values
    legal_values = action_value.masked_fill(~self_mask.bool(), 0.0)
    baseline = legal_values.sum(dim=-1, keepdim=True) / self_mask.float().sum(
        dim=-1, keepdim=True
    ).clamp_min(1.0)
    regret = (action_value - baseline).detach() * self_mask.float()
    denom = self_mask.float().sum(dim=-1).clamp_min(1.0)
    return -((regret.detach() * logits_self * self_mask.float()).sum(dim=-1) / denom).mean()


def nash_anchor_kl(logits: Tensor, target_policy: Tensor, legal_mask: Tensor) -> Tensor:
    logp = F.log_softmax(logits.masked_fill(~legal_mask.bool(), -1e9), dim=-1)
    target = target_policy.float() * legal_mask.float()
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return F.kl_div(logp, target, reduction="none").sum(dim=-1).mean()
