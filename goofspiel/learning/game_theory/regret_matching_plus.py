"""Batched regret-matching+ zero-sum matrix solver in PyTorch."""

from __future__ import annotations

import torch
from torch import Tensor

from .matrix_solver import MatrixGameSolution


def _rm_strategy(regret: Tensor, mask: Tensor) -> Tensor:
    pos = torch.relu(regret) * mask.to(regret.dtype)
    total = pos.sum(dim=-1, keepdim=True)
    uniform = mask.to(regret.dtype) / mask.sum(dim=-1, keepdim=True).clamp_min(1)
    return torch.where(total > 1e-12, pos / total.clamp_min(1e-12), uniform)


def solve_batch(
    q: Tensor,
    self_mask: Tensor,
    opponent_mask: Tensor,
    iterations: int = 64,
) -> MatrixGameSolution:
    """Approximate Nash policies for batched zero-sum matrices.

    This is the training/online approximate solver, not the FP64 reference LP.
    It never replaces the reference solver for validation.
    """
    if q.ndim != 3:
        raise ValueError(f"q must be [B,N,N], got {tuple(q.shape)}")
    batch, n, m = q.shape
    if n != m:
        raise ValueError("current training solver expects square matrices")
    row_mask = self_mask.bool()
    col_mask = opponent_mask.bool()
    if row_mask.shape != (batch, n) or col_mask.shape != (batch, n):
        raise ValueError("mask shapes must be [B,N]")

    q = q.float()
    row_regret = torch.zeros(batch, n, device=q.device, dtype=q.dtype)
    col_regret = torch.zeros_like(row_regret)
    row_sum = torch.zeros_like(row_regret)
    col_sum = torch.zeros_like(row_regret)

    for _ in range(int(iterations)):
        row = _rm_strategy(row_regret, row_mask)
        col = _rm_strategy(col_regret, col_mask)
        row_sum = row_sum + row
        col_sum = col_sum + col
        row_util = torch.bmm(q, col[:, :, None]).squeeze(-1)
        value = (row * row_util).sum(dim=-1, keepdim=True)
        col_util = torch.bmm((-q).transpose(1, 2), row[:, :, None]).squeeze(-1)
        col_value = (col * col_util).sum(dim=-1, keepdim=True)
        row_regret = torch.relu(row_regret + (row_util - value)) * row_mask.to(q.dtype)
        col_regret = torch.relu(col_regret + (col_util - col_value)) * col_mask.to(q.dtype)

    row_policy = row_sum / row_sum.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    col_policy = col_sum / col_sum.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    value = torch.einsum("bi,bij,bj->b", row_policy, q, col_policy)
    row_guarantee = (row_policy[:, None, :] @ q).squeeze(1).masked_fill(~col_mask, 1e9).min(dim=-1).values
    col_cap = (q @ col_policy[:, :, None]).squeeze(-1).masked_fill(~row_mask, -1e9).max(dim=-1).values
    duality_gap = (col_cap - row_guarantee).clamp_min(0.0)
    valid = torch.isfinite(value) & torch.isfinite(duality_gap)
    return MatrixGameSolution(row_policy, col_policy, value, duality_gap, int(iterations), valid)
