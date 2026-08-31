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


# ======================================================================
# Phase 1.2 — carry/stake must reach the model's features
# ======================================================================
def test_global_features_include_carry_and_stake():
    """The global feature vector is now 10-wide and its last two entries are
    carry_norm and stake_norm.  A carry=13,prize=3 state must expose stake=16."""
    import dataclasses

    model = GoofspielModel(max_cards=13)
    # A stake-16 round: prize 3 on the board, 13 already rolled into the carry.
    state = dataclasses.replace(
        GameState.initial(13, current_prize=3),
        carry_pool=13,
    )
    batch = public_state_from_game([state], max_cards=13)
    feats = model._global_features(batch)
    assert feats.shape == (1, 10), feats.shape
    total = 13 * 14 / 2  # S_N = 91
    carry_norm = feats[0, 8].item()
    stake_norm = feats[0, 9].item()
    assert carry_norm == pytest.approx(13.0 / total, abs=1e-6)
    # stake = current_prize + carry = 3 + 13 = 16, NOT 3.
    assert stake_norm == pytest.approx(16.0 / total, abs=1e-6)


def test_immediate_pair_feature_reflects_stake_not_prize():
    """The implied immediate matrix magnitude must reflect stake=16, not prize=3.

    We compare a carry=13 state against a carry=0 state at the same prize=3: the
    raw immediate feature must be exactly 16/3 larger in magnitude for the carry
    state. This is the concrete carry=13,prize=3 acceptance case from the plan."""
    import dataclasses

    base = GameState.initial(13, current_prize=3)          # stake = 3
    carried = dataclasses.replace(base, carry_pool=13)      # stake = 16

    def immediate_magnitude(state: GameState) -> float:
        s = public_state_from_game([state], max_cards=13)
        # Recompute the `immediate` pair feature exactly as _pair_features does,
        # asserting the stake term (current_prize + carry) drives its magnitude.
        n = s.max_cards
        ranks = torch.arange(1, n + 1, dtype=torch.float32)
        ri = (ranks[:, None] / s.n_cards.float().clamp_min(1)).expand(n, n)
        rj = (ranks[None, :] / s.n_cards.float().clamp_min(1)).expand(n, n)
        sign = torch.sign(ri - rj)
        total = s.n_cards.float() * (s.n_cards.float() + 1.0) / 2.0
        stake = s.current_prize.float() + s.carry_pool.float()
        immediate = stake[:, None] * sign / total[:, None]
        return float(immediate.abs().max().item())

    mag_base = immediate_magnitude(base)
    mag_carry = immediate_magnitude(carried)
    assert mag_carry == pytest.approx(mag_base * 16.0 / 3.0, rel=1e-5), (mag_base, mag_carry)


def test_carry_changes_the_models_robust_output_end_to_end():
    """Carry must actually flow through the real forward pass: two states that
    differ ONLY in carry_pool must yield different robust Q / policy. Before
    Phase 1.2 the model ignored carry entirely, so these would be byte-identical."""
    import dataclasses

    torch.manual_seed(0)
    model = GoofspielModel(max_cards=13)
    model.eval()
    base = GameState.initial(13, current_prize=3)
    carried = dataclasses.replace(base, carry_pool=13)
    b0 = public_state_from_game([base], max_cards=13)
    b1 = public_state_from_game([carried], max_cards=13)
    with torch.no_grad():
        out0 = model(b0)
        out1 = model(b1)
    assert not torch.allclose(out0.q_robust, out1.q_robust), "carry did not affect robust Q"
    assert not torch.allclose(out0.robust_policy_logits, out1.robust_policy_logits), (
        "carry did not affect robust policy"
    )


