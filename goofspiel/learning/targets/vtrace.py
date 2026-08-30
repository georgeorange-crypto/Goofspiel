"""Joint-policy V-trace style targets."""

from __future__ import annotations

import torch
from torch import Tensor


def joint_vtrace_targets(
    rewards: Tensor,
    values: Tensor,
    target_prob_self: Tensor,
    target_prob_opp: Tensor,
    behavior_prob_self: Tensor,
    behavior_prob_opp: Tensor,
    done: Tensor,
    lambda_: float = 0.9,
    rho_clip: float = 1.0,
    c_clip: float = 1.0,
    gamma: float = 1.0,
) -> Tensor:
    if values.shape == rewards.shape:
        values_ext = torch.cat([values, torch.zeros_like(values[:1])], dim=0)
    else:
        values_ext = values
    ratio = (target_prob_self * target_prob_opp) / (behavior_prob_self * behavior_prob_opp).clamp_min(1e-12)
    rho = torch.clamp(ratio, max=rho_clip)
    c = lambda_ * torch.clamp(ratio, max=c_clip)
    out = torch.zeros_like(rewards)
    acc = torch.zeros_like(rewards[0])
    for t in reversed(range(rewards.shape[0])):
        nonterminal = (~done[t].bool()).float()
        delta = rho[t] * (rewards[t] + gamma * values_ext[t + 1] * nonterminal - values_ext[t])
        acc = delta + gamma * c[t] * nonterminal * acc
        out[t] = values_ext[t] + acc
    return out
