"""Exact Nash tool boundary backed by the project's existing solvers."""

from __future__ import annotations

import time
from typing import Any

import torch

from goofspiel.game import GameState
from goofspiel.reasoning.types import Exactness, GameToolResult, ToolMode
from goofspiel.solver import GoofspielCarrySolver
from goofspiel.training.teachers import exact_teacher_for_current_prize


def solve_exact_tool(
    state: GameState,
    *,
    max_remaining: int = 4,
    mode: ToolMode = ToolMode.PLAY,
    solver: GoofspielCarrySolver | None = None,
) -> GameToolResult:
    """Return exact current-prize matrix and policy when the state is feasible."""
    started = time.perf_counter()
    remaining = len(state.self_actions)
    if remaining > max_remaining:
        mask = torch.zeros(13, dtype=torch.bool)
        mask[[a - 1 for a in state.self_actions]] = True
        return GameToolResult(
            source="EXACT_NASH",
            mode=mode.value,
            policy_self=torch.zeros(13),
            valid_self_mask=mask,
            valid_opponent_mask=mask.clone(),
            valid=False,
            exactness=Exactness.NONE.value,
            diagnostics={"reason": "EXACT_BUDGET_EXCEEDED", "remaining_actions": remaining},
        )
    del solver
    sample = exact_teacher_for_current_prize(state)
    q = torch.zeros(13, 13)
    q_small = torch.tensor(sample.q_matrix, dtype=torch.float32)
    row = torch.zeros(13)
    col = torch.zeros(13)
    for idx, a in enumerate(state.self_actions):
        row[a - 1] = float(sample.row_policy[idx])
        for jdx, b in enumerate(state.opponent_actions):
            q[a - 1, b - 1] = q_small[idx, jdx]
    for idx, b in enumerate(state.opponent_actions):
        col[b - 1] = float(sample.column_policy[idx])
    self_mask = torch.zeros(13, dtype=torch.bool)
    opp_mask = torch.zeros(13, dtype=torch.bool)
    self_mask[[a - 1 for a in state.self_actions]] = True
    opp_mask[[a - 1 for a in state.opponent_actions]] = True
    # `exact_teacher_for_current_prize` solves the *immediate* Q-matrix only; it
    # ignores continuation value. That is the exact full-game matrix only on the
    # last round (no future prizes to reveal, `prize_mask == 0`). For any earlier
    # state it is an approximation, so we must not stamp NUMERICAL_EXACT there —
    # doing so was the dishonest label. Terminal round -> exact; otherwise
    # APPROXIMATE, with a diagnostic recording why.
    is_last_round = state.prize_mask == 0
    exactness = Exactness.NUMERICAL_EXACT if is_last_round else Exactness.APPROXIMATE
    diagnostics: dict[str, Any] = {"solver": sample.solver_precision, "is_last_round": bool(is_last_round)}
    if not is_last_round:
        diagnostics["approximation"] = "IMMEDIATE_MATRIX_IGNORES_CONTINUATION_VALUE"
    return GameToolResult(
        source="EXACT_NASH",
        mode=mode.value,
        policy_self=row,
        policy_opponent=col,
        q_matrix=q,
        value=float(sample.value),
        valid_self_mask=self_mask,
        valid_opponent_mask=opp_mask,
        quality_score=10.0 if is_last_round else 5.0,
        exactness=exactness.value,
        runtime_ms=(time.perf_counter() - started) * 1000.0,
        state_key=getattr(sample.state, "state_hash", None),
        diagnostics=diagnostics,
        valid=True,
    )
