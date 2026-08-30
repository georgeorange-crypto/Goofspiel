from __future__ import annotations

from goofspiel.training.adaptive import evaluate_uniform_opponent_gate
from goofspiel.training.data import OpponentSession, RoundRecord


def test_adaptive_gate_uses_session_level_data():
    session = OpponentSession(
        session_id="s",
        opponent_id="o",
        strategy_regime_id="uniform",
        games=[[RoundRecord(1, 1, 1, 2, 0, 1)]],
    )
    gate = evaluate_uniform_opponent_gate(session, n_cards=3)
    assert gate.passed
    assert gate.ece == 0.0
