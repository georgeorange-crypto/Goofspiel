from __future__ import annotations

import pytest

try:
    import torch
except OSError as exc:  # pragma: no cover - machine environment guard
    pytest.skip(f"torch cannot be imported in this environment: {exc}", allow_module_level=True)

from goofspiel.game import GameState
from goofspiel.models import GoofspielModel, HistoryBatch, OpponentMemoryBatch, public_state_from_game


def _history(batch: int, steps: int, fill: int) -> HistoryBatch:
    return HistoryBatch(
        prize=torch.full((batch, steps), fill),
        self_action=torch.full((batch, steps), 1),
        opponent_action=torch.full((batch, steps), fill),
        score_diff=torch.zeros(batch, steps),
        outcome=torch.zeros(batch, steps),
        round_idx=torch.arange(steps).float().repeat(batch, 1),
        valid_mask=torch.ones(batch, steps, dtype=torch.bool),
    )


def _memory(batch: int, games: int, value: float) -> OpponentMemoryBatch:
    return OpponentMemoryBatch(
        game_summary_sequence=torch.full((batch, games, 192), value),
        valid_mask=torch.ones(batch, games, dtype=torch.bool),
    )


def test_model_variable_n_shape_contract():
    model = GoofspielModel(max_cards=13)
    model.eval()
    states = [
        GameState.initial(3, current_prize=1),
        GameState.initial(5, current_prize=2),
        GameState.initial(13, current_prize=7),
    ]
    batch = public_state_from_game(states, max_cards=13)
    with torch.no_grad():
        out = model(batch)
    assert out.q_robust.shape == (3, 13, 13)
    assert out.q_robust_heads.shape == (3, 4, 13, 13)
    assert out.robust_policy_logits.shape == (3, 13)
    assert out.robust_score_logits.shape == (3, 201)
    assert out.q_adaptive.shape == (3, 13, 13)
    assert out.opponent_fused_logits.shape == (3, 13)
    assert out.joint_action_mask.shape == (3, 13, 13)
    assert torch.isfinite(out.q_robust[out.joint_action_mask]).all()


def test_opponent_history_does_not_leak_into_robust_outputs():
    torch.manual_seed(1)
    model = GoofspielModel(max_cards=13)
    model.eval()
    batch = public_state_from_game([GameState.initial(5, current_prize=2)], max_cards=13)
    hist_a = _history(1, 3, 2)
    hist_b = _history(1, 3, 5)
    mem_a = _memory(1, 2, 0.1)
    mem_b = _memory(1, 2, 3.0)
    with torch.no_grad():
        out_a = model(batch, hist_a, mem_a)
        out_b = model(batch, hist_b, mem_b)
    assert torch.allclose(out_a.q_robust, out_b.q_robust, atol=0.0, rtol=0.0)
    assert torch.allclose(out_a.robust_policy_logits, out_b.robust_policy_logits, atol=0.0, rtol=0.0)
    assert torch.allclose(out_a.robust_score_logits, out_b.robust_score_logits, atol=0.0, rtol=0.0)
    assert not torch.allclose(out_a.opponent_fused_logits, out_b.opponent_fused_logits)


def test_parameter_count_report_contains_required_groups():
    model = GoofspielModel(max_cards=13)
    counts = model.parameter_count_by_module()
    for key in [
        "rank_encoder",
        "card_transformer",
        "relational_gnn",
        "matrix_cnn",
        "lstm",
        "mamba_memory",
        "adaptive_branch",
        "heads",
        "total",
        "trainable",
    ]:
        assert key in counts
        assert counts[key] > 0
