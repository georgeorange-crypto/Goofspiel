"""On-policy TD(lambda) targets."""

from __future__ import annotations

import torch
from torch import Tensor


def lambda_returns(
    rewards: Tensor,
    values: Tensor,
    done: Tensor,
    lambda_: float = 0.9,
    gamma: float = 1.0,
) -> Tensor:
    """Compute backward TD(lambda) returns for [T,B] tensors.

    `values` may be [T,B] or [T+1,B].  With [T,B], the bootstrap after the
    final row is treated as zero.
    """
    if rewards.ndim != 2:
        raise ValueError("rewards must be [T,B]")
    t_steps, batch = rewards.shape
    if values.shape == rewards.shape:
        values_ext = torch.cat([values, torch.zeros(1, batch, device=values.device, dtype=values.dtype)], dim=0)
    elif values.shape[0] == t_steps + 1:
        values_ext = values
    else:
        raise ValueError("values must be [T,B] or [T+1,B]")
    out = torch.zeros_like(rewards)
    g = values_ext[-1]
    for t in reversed(range(t_steps)):
        nonterminal = (~done[t].bool()).float()
        bootstrap = (1.0 - lambda_) * values_ext[t + 1] + lambda_ * g
        g = rewards[t] + gamma * nonterminal * bootstrap
        out[t] = g
    return out
