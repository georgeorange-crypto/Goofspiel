"""Canonical reasoning state keys and privacy-separated inputs.

The robust / adaptive split is a **structural invariant**, not a convention:

* :meth:`ReasoningState.robust_view` returns a state that provably carries *no*
  opponent private / history / adaptation information.  The robust tiers
  (matrix-Nash, exact, GT-CFR, SM-MCTS) may only ever see this view, so the
  robust value ``Q_R(s, a, b)`` is opponent-agnostic by construction.
* :meth:`ReasoningState.adaptive_view` returns a state carrying the opponent
  history, session summary / memory tensors, and any explicit opponent belief.
  Only the adaptive tier consumes it, so ``Q_A(s, h, a, b)`` is
  opponent-conditioned by construction.

``Q_R ⊥ opponent history`` is therefore guaranteed by which view a tier is
handed, not by remembering to strip a field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from goofspiel.game import GameState

# The fields that make a state opponent-*conditioned*.  ``robust_view`` clears
# every one of them; a test asserts a robust view exposes none.
OPPONENT_INFORMATION_FIELDS = (
    "opponent_history",
    "opponent_model_version",
    "opponent_belief",
    "opponent_memory",
    "current_game_history",
)


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
    # ---- opponent-conditioned inputs (adaptive tier only) -------------------
    opponent_history: tuple[Any, ...] = ()
    opponent_model_version: str | None = None
    # An explicit belief over the opponent's next action, indexed by ``card-1``
    # (length up to 13).  When present the adaptive tier uses it directly; this
    # is how a caller injects a *known* opponent bias.
    opponent_belief: tuple[float, ...] | None = None
    # Opponent session memory (an ``OpponentMemoryBatch``) and the current-game
    # history (a ``HistoryBatch``) used to run the trained opponent model.
    opponent_memory: Any = None
    current_game_history: Any = None

    @property
    def canonical_key(self) -> CanonicalStateKey:
        return CanonicalStateKey.from_state(self.public_state)

    def robust_view(self) -> "ReasoningState":
        """Opponent-agnostic view — carries no opponent information at all.

        Every field in :data:`OPPONENT_INFORMATION_FIELDS` is at its empty
        default here, so the robust tiers cannot condition on the opponent even
        by accident.
        """
        return ReasoningState(public_state=self.public_state, model_version=self.model_version)

    def adaptive_view(self) -> "ReasoningState":
        """Opponent-conditioned view — carries history / memory / belief.

        This is the *only* view the adaptive tier is allowed to read.
        """
        return self

    def exposes_opponent_information(self) -> bool:
        """True iff any opponent-conditioned field is populated.

        A robust view must return ``False``; an adaptive view carrying a belief,
        history, or memory returns ``True``.
        """
        for name in OPPONENT_INFORMATION_FIELDS:
            value = getattr(self, name)
            if name == "opponent_history":
                if tuple(value):
                    return True
            elif value is not None:
                return True
        return False
