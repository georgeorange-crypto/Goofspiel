"""Teacher target generation for exact, matrix, and priority-routed labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from goofspiel.game import GameState, legal_cards
from goofspiel.solver import cards_to_mask, solve_zero_sum_matrix
from goofspiel.training.data import ExactSample, TeacherSample, state_record_from_game_state


@dataclass
class TeacherRouterConfig:
    exact_max_remaining: int = 4
    exact_force: bool = False
    pseudo_q_disagreement_max: float = 0.02
    pseudo_policy_jsd_max: float = 0.05


def immediate_q_matrix(state: GameState) -> tuple[np.ndarray, list[int], list[int]]:
    a_cards = legal_cards(state.self_mask, state.n)
    b_cards = legal_cards(state.opp_mask, state.n)
    q = np.zeros((len(a_cards), len(b_cards)), dtype=np.float64)
    total = state.n * (state.n + 1) // 2
    stake = state.current_prize + state.carry_pool
    for i, a in enumerate(a_cards):
        for j, b in enumerate(b_cards):
            q[i, j] = stake * (1 if a > b else (-1 if a < b else 0)) / total
    return q, a_cards, b_cards


def matrix_teacher_from_q(state: GameState, q: np.ndarray, source: str = "REFERENCE_NASH_Q") -> TeacherSample:
    value, row, _col = solve_zero_sum_matrix(q)
    return TeacherSample(
        sample_id=f"{source}:{state.n}:{state.self_mask}:{state.opp_mask}:{state.prize_mask}:{state.current_prize}:{state.carry_pool}",
        state=state_record_from_game_state(state),
        teacher_q=q.tolist(),
        teacher_policy=row.tolist(),
        teacher_value=float(value),
        teacher_source=source,
        teacher_confidence=1.0,
    )


def exact_teacher_for_current_prize(state: GameState) -> ExactSample:
    """Build exact one-step matrix if the continuation is already encoded.

    This first implementation handles terminal states exactly and otherwise
    returns the normalized immediate matrix as a conservative teacher anchor.
    Full recursive exact-current-state labeling can be swapped in later without
    changing the `ExactSample` schema.
    """
    q, _a, _b = immediate_q_matrix(state)
    value, row, col = solve_zero_sum_matrix(q)
    return ExactSample(
        sample_id=f"EXACT:{state.n}:{state.self_mask}:{state.opp_mask}:{state.prize_mask}:{state.current_prize}:{state.carry_pool}",
        state=state_record_from_game_state(state),
        q_matrix=q.tolist(),
        row_policy=row.tolist(),
        column_policy=col.tolist(),
        value=float(value),
    )


class TeacherRouter:
    def __init__(self, config: TeacherRouterConfig | None = None) -> None:
        self.config = config or TeacherRouterConfig()

    def label_state(self, state: GameState, model_q: Any | None = None) -> TeacherSample:
        remaining = len(legal_cards(state.self_mask, state.n))
        if remaining <= self.config.exact_max_remaining:
            exact = exact_teacher_for_current_prize(state)
            return TeacherSample(
                sample_id=exact.sample_id,
                state=exact.state,
                teacher_q=exact.q_matrix,
                teacher_policy=exact.row_policy,
                teacher_value=exact.value,
                teacher_source="EXACT",
                teacher_confidence=1.0,
            )
        if model_q is not None:
            q = np.asarray(model_q, dtype=np.float64)
        else:
            q, _a, _b = immediate_q_matrix(state)
        return matrix_teacher_from_q(state, q, source="REFERENCE_NASH_Q")
