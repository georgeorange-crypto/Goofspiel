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
    opponent_mask: Tensor | None = None,
    epsilon: float = 0.0,
    max_grid: int = 101,
) -> tuple[Tensor, Tensor]:
    """Return the best robust-safe mixture of robust and adaptive policy.

    It grid-searches ``alpha`` in ``policy = (1-alpha)*robust + alpha*adaptive``
    and keeps the highest expected value under the opponent belief subject to the
    robust floor against every *legal* pure opponent action:

        policy^T Q[:, b] >= V_R - epsilon    for every legal column b.

    ``V_R`` (``robust_value``) defaults to the **minimax worst case of the robust
    policy over legal columns** — a genuine opponent-agnostic guarantee, not a
    belief-weighted average.  With that default ``alpha = 0`` (pure robust) always
    satisfies the floor, so a safe mixture always exists and the adaptive tier is
    used exactly when it raises the belief-expected value without breaching the
    guarantee.

    ``opponent_mask`` marks the legal opponent columns; illegal (padded) columns
    carry ``Q = 0`` and must be excluded from the floor, otherwise a spurious
    ``0 >= V_R - epsilon`` constraint would suppress every exploit.  When omitted,
    all columns are treated as legal (back-compatible with dense matrices).
    """
    r = _normalize(robust_policy, self_mask)
    a = _normalize(adaptive_policy, self_mask)
    belief = opponent_belief.float()
    belief = belief / belief.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    # Legal opponent columns constrain the floor; illegal ones are ignored.
    if opponent_mask is None:
        col_mask = torch.ones(robust_q.shape[0], robust_q.shape[-1], dtype=torch.bool, device=robust_q.device)
    else:
        col_mask = opponent_mask.bool()
        if col_mask.ndim == 1:
            col_mask = col_mask.unsqueeze(0).expand(robust_q.shape[0], -1)

    if robust_value is None:
        # Worst-case value of the robust policy over legal columns = its
        # guaranteed floor against any opponent.
        r_col = torch.einsum("bi,bij->bj", r, robust_q)
        robust_value = r_col.masked_fill(~col_mask, float("inf")).min(dim=-1).values

    alphas = torch.linspace(0.0, 1.0, max_grid, device=robust_q.device)
    candidates = r[:, None, :] * (1.0 - alphas[None, :, None]) + a[:, None, :] * alphas[None, :, None]
    pure_cols = torch.einsum("bki,bij->bkj", candidates, robust_q)
    pure_floor = pure_cols.masked_fill(~col_mask[:, None, :], float("inf")).min(dim=-1).values
    expected = torch.einsum("bki,bij,bj->bk", candidates, robust_q, belief)
    ok = pure_floor >= (robust_value[:, None] - float(epsilon))
    expected = expected.masked_fill(~ok, -1e9)
    best = expected.argmax(dim=-1)
    chosen = candidates[torch.arange(candidates.shape[0], device=candidates.device), best]
    alpha = alphas[best]
    return chosen, alpha
