"""Matrix Nash tool wrapper for current-state joint Q estimates."""

from __future__ import annotations

import time
from typing import Any

import torch
from torch import Tensor

from goofspiel.learning.game_theory.regret_matching_plus import solve_batch
from goofspiel.reasoning.types import Exactness, GameToolResult, ToolMode


def solve_matrix_nash_tool(
    q_matrix: Tensor,
    self_mask: Tensor,
    opponent_mask: Tensor,
    *,
    iterations: int = 512,
    mode: ToolMode = ToolMode.PLAY,
    state_key: Any = None,
    model_version: str | None = None,
) -> GameToolResult:
    """Solve a batch of zero-sum matrix games from neural Q estimates."""
    started = time.perf_counter()
    solution = solve_batch(q_matrix, self_mask, opponent_mask, iterations=iterations)
    valid = torch.isfinite(solution.row_policy).all() and torch.isfinite(solution.column_policy).all()
    gap = solution.duality_gap
    quality = float((1.0 / (1.0 + gap.detach().float().mean())).cpu())
    return GameToolResult(
        source="MODEL_MATRIX_NASH",
        mode=mode.value,
        policy_self=solution.row_policy,
        policy_opponent=solution.column_policy,
        q_matrix=q_matrix,
        value=solution.value,
        valid_self_mask=self_mask,
        valid_opponent_mask=opponent_mask,
        quality_score=quality,
        duality_gap=gap,
        exactness=Exactness.APPROXIMATE.value,
        runtime_ms=(time.perf_counter() - started) * 1000.0,
        simulations=iterations,
        state_key=state_key,
        model_version=model_version,
        diagnostics={"solver": "regret_matching_plus", "iterations": iterations},
        valid=bool(valid.item()),
    )
