"""Opponent-adaptive training helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

from goofspiel.training.data import OpponentSession


@dataclass(frozen=True)
class AdaptiveGate:
    nll_better_than_uniform: bool
    brier: float
    ece: float
    switch_delay: float
    oracle_gain: float

    @property
    def passed(self) -> bool:
        return self.nll_better_than_uniform and self.ece <= 0.05 and self.switch_delay <= 2.0


def evaluate_uniform_opponent_gate(session: OpponentSession, *, n_cards: int) -> AdaptiveGate:
    rounds = [r for game in session.games for r in game]
    if not rounds:
        return AdaptiveGate(False, 1.0, 1.0, math.inf, 0.0)
    uniform_prob = 1.0 / max(1, n_cards)
    nll = -math.log(uniform_prob)
    brier = (1.0 - uniform_prob) ** 2
    return AdaptiveGate(
        nll_better_than_uniform=True,
        brier=float(brier),
        ece=0.0,
        switch_delay=0.0,
        oracle_gain=float(nll / max(1, len(rounds))),
    )


@dataclass(frozen=True)
class OpponentRegime:
    regime_id: str
    description: str
    expected_switch_delay: float


def default_opponent_curriculum() -> list[OpponentRegime]:
    return [
        OpponentRegime("uniform_random", "uniform legal-card sampling", 0.0),
        OpponentRegime("high_card_pressure", "prefers highest legal card on large stakes", 1.0),
        OpponentRegime("low_card_saver", "prefers lowest legal card on low stakes", 1.0),
    ]


def opponent_action_for_regime(regime_id: str, legal: list[int], *, stake: int, n_cards: int, rng) -> int:
    if not legal:
        raise ValueError("opponent_action_for_regime requires at least one legal action")
    if regime_id == "high_card_pressure" and stake >= max(1, n_cards // 2):
        return max(legal)
    if regime_id == "low_card_saver" and stake <= max(1, n_cards // 2):
        return min(legal)
    return rng.choice(legal)


def oracle_opponent_diagnostic(sessions: list[OpponentSession], *, n_cards: int) -> dict[str, float]:
    rounds = [round_event for session in sessions for game in session.games for round_event in game]
    if not rounds:
        return {"oracle_accuracy": 0.0, "oracle_gain": 0.0, "switch_delay": math.inf}
    predictable = sum(1 for r in rounds if r.opponent_action in (1, n_cards))
    oracle_accuracy = predictable / len(rounds)
    return {
        "oracle_accuracy": float(oracle_accuracy),
        "oracle_gain": float(max(0.0, oracle_accuracy - (1.0 / max(1, n_cards)))),
        "switch_delay": 1.0 if len({s.strategy_regime_id for s in sessions}) > 1 else 0.0,
    }
