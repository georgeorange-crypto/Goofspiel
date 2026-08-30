"""Canonical reasoning state keys and privacy-separated inputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from goofspiel.game import GameState


@dataclass(frozen=True)
class CanonicalStateKey:
    n: int
    self_mask: int
    opponent_mask: int
    prize_mask: int
    current_prize: int
    carry_pool: int
    score_diff: int

    @classmethod
    def from_state(cls, state: GameState) -> "CanonicalStateKey":
        return cls(
            n=state.n,
            self_mask=state.self_mask,
            opponent_mask=state.opp_mask,
            prize_mask=state.prize_mask,
            current_prize=state.current_prize,
            carry_pool=state.carry_pool,
            score_diff=state.self_score - state.opp_score,
        )


@dataclass(frozen=True)
class ChanceStateKey:
    base: CanonicalStateKey
    remaining_prize_mask: int


@dataclass(frozen=True)
class ReasoningState:
    public_state: GameState
    model_version: str = "unversioned"
    opponent_history: tuple[Any, ...] = ()
    opponent_model_version: str | None = None

    @property
    def canonical_key(self) -> CanonicalStateKey:
        return CanonicalStateKey.from_state(self.public_state)

    def robust_view(self) -> "ReasoningState":
        return ReasoningState(public_state=self.public_state, model_version=self.model_version)

    def adaptive_view(self) -> "ReasoningState":
        return self
