"""Stage7 held-out split kills the memorization hole.

Two complementary checks, both RE-EXECUTING the fact:

1. **Detector semantics (the flag==1.0 path).**  ``_memorization_flag`` is the
   pure decision the runner delegates to.  We drive it with
   ``_attack_state_regression``-shaped bucket dicts to prove it raises the flag
   exactly when TRAIN improved but NO held-out bucket did, and lowers it when a
   held-out bucket also improved.  This is the memorization signal the old
   train==test design could never express.

2. **Runner reproduction.**  We run Stage7 with a real held-out split, reload the
   BEFORE and AFTER checkpoints the run saved, rebuild the exact train / held-out
   states, recompute every bucket's match-rate with ``_attack_state_regression``,
   feed them back through ``_memorization_flag``, and assert the recomputed flag
   AND per-bucket match-rates reproduce what the report recorded — so the flag is
   a re-run fact, not a stored literal.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from goofspiel.game import GameState


def _bucket(match_rate: float) -> dict:
    """A minimal ``_attack_state_regression``-shaped result carrying only what
    ``_memorization_flag`` reads (``match_rate``)."""
    return {"match_rate": match_rate, "matched": 0, "total": 1, "per_state": [], "passed": False}


def test_memorization_flag_semantics():
    from goofspiel.training.stages import _memorization_flag

    train_before = _bucket(0.0)
    train_after = _bucket(1.0)  # TRAIN improved

    # Held-out did NOT improve (flat) → memorization detected.
    heldout_flat = [(_bucket(0.25), _bucket(0.25)), (_bucket(0.5), _bucket(0.5))]
    assert _memorization_flag(train_before, train_after, heldout_flat) is True

    # A held-out bucket ALSO improved → genuine generalization, no flag.
    heldout_up = [(_bucket(0.25), _bucket(0.25)), (_bucket(0.5), _bucket(0.75))]
    assert _memorization_flag(train_before, train_after, heldout_up) is False

    # No held-out buckets at all (SMOKE) → cannot claim memorization.
    assert _memorization_flag(train_before, train_after, [(None, None), (None, None)]) is False

    # TRAIN did not improve → not a memorization case regardless of held-out.
    assert _memorization_flag(_bucket(1.0), _bucket(1.0), heldout_flat) is False


def test_stage7_heldout_split_reproduces_reported_flag(tmp_path: Path):
    try:
        __import__("torch")
    except Exception as exc:  # pragma: no cover - depends on local torch install
        pytest.skip(f"torch cannot be imported: {exc}")

    from goofspiel.models import GoofspielModel
    from goofspiel.training.budgets import Stage7Budget
    from goofspiel.training.checkpoint import load_checkpoint
    from goofspiel.training.stages import (
        _attack_state_regression,
        _generate_attacks,
        _memorization_flag,
        run_stage7_redteam,
    )
    from goofspiel.training.teachers import TeacherRouter

    seed, n_cards, steps = 43, 3, 30
    train_cases, heldout = 2, 4
    budget = Stage7Budget(
        attack_cases=3,
        correction_steps=steps,
        correction_train_cases=train_cases,
        heldout_attack_cases=heldout,
        arena_games=0,
        arena_seeds=0,
    )
    run_stage7_redteam(
        out_dir=tmp_path / "s7",
        correction_steps=steps,
        n_cards=n_cards,
        seed=seed,
        budget=budget,
    )
    report = json.loads((tmp_path / "s7" / "redteam" / "focused_correction_report.json").read_text(encoding="utf-8"))

    # The held-out buckets are present and non-empty (the split actually ran).
    assert report["heldout_same_family"] is not None
    assert report["heldout_other_family"] is not None
    reported_flag = bool(report["memorization_flag"])

    # ---- Rebuild the EXACT train / held-out states the runner used ------------
    router = TeacherRouter()

    def teacher_card(state: GameState) -> int:
        sample = router.label_state(state)
        pol = sample.teacher_policy or [1.0] * len(state.self_actions)
        best = max(range(len(state.self_actions)), key=lambda i: pol[i])
        return state.self_actions[best]

    discovered = _generate_attacks(n_cards=n_cards, seed=seed, count=3)
    train_states = [c.state for c in discovered[:train_cases]]
    train_cards = [teacher_card(s) for s in train_states]
    hs_states = [c.state for c in discovered[train_cases:]]
    hs_cards = [teacher_card(s) for s in hs_states]
    ho_cases = _generate_attacks(
        n_cards=n_cards, seed=seed * 3 + 1, count=heldout,
        families=("curriculum_regimes", "carry_and_asymmetric_masks"),
        include_legacy_prefix=False,
    )
    ho_states = [c.state for c in ho_cases]
    ho_cards = [teacher_card(s) for s in ho_states]

    tp = report["training_plan"]

    def reload_model(path: str):
        model = GoofspielModel(max_cards=13)
        model.load_state_dict(load_checkpoint(path)["model_state"])
        model.eval()
        return model

    before = reload_model(tp["init_checkpoint"])
    after = reload_model(tp["corrected_checkpoint"])

    train_b = _attack_state_regression(before, train_states, train_cards)
    train_a = _attack_state_regression(after, train_states, train_cards)
    hs_b = _attack_state_regression(before, hs_states, hs_cards)
    hs_a = _attack_state_regression(after, hs_states, hs_cards)
    ho_b = _attack_state_regression(before, ho_states, ho_cards)
    ho_a = _attack_state_regression(after, ho_states, ho_cards)

    # Per-bucket match-rates reproduce the report exactly (re-run, not stored).
    assert train_b["match_rate"] == pytest.approx(report["regression"]["match_rate_before"])
    assert train_a["match_rate"] == pytest.approx(report["regression"]["match_rate_after"])
    assert hs_b["match_rate"] == pytest.approx(report["heldout_same_family"]["match_rate_before"])
    assert hs_a["match_rate"] == pytest.approx(report["heldout_same_family"]["match_rate_after"])
    assert ho_b["match_rate"] == pytest.approx(report["heldout_other_family"]["match_rate_before"])
    assert ho_a["match_rate"] == pytest.approx(report["heldout_other_family"]["match_rate_after"])

    # The memorization flag itself is reproducible from the reloaded checkpoints.
    recomputed = _memorization_flag(train_b, train_a, [(hs_b, hs_a), (ho_b, ho_a)])
    assert recomputed == reported_flag, "reported memorization_flag is not a reproducible re-run fact"
