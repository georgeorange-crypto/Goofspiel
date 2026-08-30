"""Safe robust/adaptive policy mixture helpers."""

from __future__ import annotations

import torch
from torch import Tensor


def _normalize(policy: Tensor, mask: Tensor) -> Tensor:
    p = policy.float().clamp_min(0.0) * mask.float()
    return p / p.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def safe_exploit_mixture(
    robust_policy: Tensor,
    adaptive_policy: Tensor,
    robust_q: Tensor,
    opponent_belief: Tensor,
    self_mask: Tensor,
    *,
    robust_value: Tensor | None = None,
    epsilon: float = 0.0,
    max_grid: int = 101,
) -> tuple[Tensor, Tensor]:
    """Return the best robust-safe mixture of robust and adaptive policy.

    It grid-searches alpha in policy=(1-alpha)*robust+alpha*adaptive and keeps
    the highest expected value under the opponent belief subject to the robust
    floor against every pure opponent action:
        policy^T Q[:,b] >= V_R - epsilon.
    """
    r = _normalize(robust_policy, self_mask)
    a = _normalize(adaptive_policy, self_mask)
    belief = opponent_belief.float()
    belief = belief / belief.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    if robust_value is None:
        robust_value = torch.einsum("bi,bij,bj->b", r, robust_q, belief)
    alphas = torch.linspace(0.0, 1.0, max_grid, device=robust_q.device)
    candidates = r[:, None, :] * (1.0 - alphas[None, :, None]) + a[:, None, :] * alphas[None, :, None]
    pure_floor = torch.einsum("bki,bij->bkj", candidates, robust_q).min(dim=-1).values
    expected = torch.einsum("bki,bij,bj->bk", candidates, robust_q, belief)
    ok = pure_floor >= (robust_value[:, None] - float(epsilon))
    expected = expected.masked_fill(~ok, -1e9)
    best = expected.argmax(dim=-1)
    chosen = candidates[torch.arange(candidates.shape[0], device=candidates.device), best]
    alpha = alphas[best]
    return chosen, alpha
