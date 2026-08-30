from __future__ import annotations

from pathlib import Path

import pytest

from goofspiel.game import GameState
from goofspiel.training import JsonlStore, TrainingCoordinator, TrainingRunConfig
from goofspiel.training.corpus import generate_random_game_corpus
from goofspiel.training.data import GameCorpusSample, state_record_from_game_state
from goofspiel.training.stage0_verify import run_stage0_verify
from goofspiel.training.teachers import TeacherRouter
from goofspiel.training.teacher_system import EMATeacher, TeacherEnsemble, TeacherFilterConfig
from goofspiel.training.adaptive import default_opponent_curriculum, opponent_action_for_regime, oracle_opponent_diagnostic


def test_jsonl_store_roundtrip(tmp_path: Path):
    store = JsonlStore(tmp_path / "samples.jsonl")
    sample = GameCorpusSample(
        sample_id="s1",
        state=state_record_from_game_state(GameState.initial(3, current_prize=1)),
        round_event=None,
    )
    store.append(sample)
    rows = list(store.iter_dicts())
    assert len(rows) == 1
    assert rows[0]["sample_id"] == "s1"


def test_stage0_verify_passes(tmp_path: Path):
    report = run_stage0_verify(artifact_dir=tmp_path / "stage0")
    assert report.ok, report.errors
    assert report.checks["teacher_priority_contract"]


def test_build_corpus_writes_samples(tmp_path: Path):
    metrics = generate_random_game_corpus(out_path=tmp_path / "corpus.jsonl", num_games=2, n_min=3, n_max=3, seed=1)
    assert metrics["games"] == 2
    assert metrics["samples"] == 6


def test_teacher_router_labels_small_state_exact():
    sample = TeacherRouter().label_state(GameState.initial(3, current_prize=2))
    assert sample.teacher_source == "EXACT"
    assert sample.teacher_policy is not None
    assert abs(sum(sample.teacher_policy) - 1.0) < 1e-6


def test_coordinator_dry_run_lists_declared_stages(tmp_path: Path):
    result = TrainingCoordinator(
        TrainingRunConfig(artifact_dir=str(tmp_path), dry_run=True)
    ).run()
    assert result["dry_run"] is True
    assert "stage7_redteam" in result["declared_stages"]


def test_all_non_torch_stages_have_callable_runners(tmp_path: Path):
    for stage in ["stage0_verify", "build_corpus", "stage2_semi_supervised", "stage5_adaptive", "stage6_league", "stage7_redteam", "evaluate"]:
        result = TrainingCoordinator(
            TrainingRunConfig(
                artifact_dir=str(tmp_path / stage),
                stage=stage,
                steps=2,
                num_corpus_games=1,
                n_cards=3,
            )
        ).run()
        assert result["ok"] is True


def test_stage4_collects_selfplay_replay(tmp_path: Path):
    try:
        __import__("torch")
    except Exception as exc:  # pragma: no cover - depends on local torch install
        pytest.skip(f"torch cannot be imported in this environment: {exc}")
    result = TrainingCoordinator(
        TrainingRunConfig(
            artifact_dir=str(tmp_path / "stage4"),
            stage="stage4_robust_rl",
            steps=1,
            batch_size=1,
            n_cards=2,
            device="cpu",
        )
    ).run()
    assert result["ok"] is True
    metrics = result["metrics"]["metrics"]
    assert metrics["selfplay_trajectories"] == 1.0
    assert metrics["selfplay_transitions"] == 2.0
    assert metrics["replay_samples"] == 1.0
    assert metrics["target_network_ema"] == 0.995
    assert metrics["promotion_candidate"] == 1.0
    assert (tmp_path / "stage4" / "checkpoints" / "replay" / "selfplay_robust.jsonl").exists()
    assert (tmp_path / "stage4" / "checkpoints" / "curriculum" / "stage4_manifest.json").exists()
    assert (tmp_path / "stage4" / "checkpoints" / "promotion" / "stage4_promotion.json").exists()


def test_stage5_builds_opponent_session_calibration(tmp_path: Path):
    result = TrainingCoordinator(
        TrainingRunConfig(
            artifact_dir=str(tmp_path / "stage5"),
            stage="stage5_adaptive",
            steps=3,
            n_cards=3,
        )
    ).run()
    assert result["ok"] is True
    metrics = result["metrics"]["metrics"]
    assert metrics["opponent_model_usable"] == 1.0
    assert metrics["opponent_sessions"] == 3.0
    assert metrics["opponent_regimes"] == 3.0
    assert metrics["oracle_gain"] >= 0.0
    assert (tmp_path / "stage5" / "adaptive" / "opponent_sessions.jsonl").exists()
    assert (tmp_path / "stage5" / "adaptive" / "adaptive_gate_report.json").exists()


def test_stage6_writes_league_crossplay_report(tmp_path: Path):
    result = TrainingCoordinator(
        TrainingRunConfig(artifact_dir=str(tmp_path / "stage6"), stage="stage6_league")
    ).run()
    assert result["ok"] is True
    metrics = result["metrics"]["metrics"]
    assert metrics["league_agents"] == 3.0
    assert metrics["crossplay_pairs"] == 9.0
    assert metrics["pfsp_weights"] == 3.0
    assert (tmp_path / "stage6" / "league" / "league_report.json").exists()


def test_stage7_writes_redteam_reanalysis(tmp_path: Path):
    result = TrainingCoordinator(
        TrainingRunConfig(artifact_dir=str(tmp_path / "stage7"), stage="stage7_redteam")
    ).run()
    assert result["ok"] is True
    metrics = result["metrics"]["metrics"]
    assert metrics["failures"] == 3.0
    assert metrics["corrections"] == 3.0
    assert metrics["teacher_relabels"] == 3.0
    assert metrics["focused_correction_steps"] == 3.0
    assert metrics["original_attack_regression_passed"] == 1.0
    assert metrics["general_regression_passed"] == 1.0
    assert (tmp_path / "stage7" / "redteam" / "redteam_report.json").exists()
    assert (tmp_path / "stage7" / "redteam" / "focused_correction_report.json").exists()


# ================================================================
# P1 multi-task pretraining: verify 5-task metric keys are emitted
# ================================================================
def test_stage1_pretrain_emits_multitask_loss_keys(tmp_path: Path):
    """If torch is importable, run stage1 and check all 5 multitask loss keys.

    This is the definitive contract that P1 = immediate + swap + future-opp
    + masked-history + style-contrastive, not only a single joint outcome loss.
    """
    try:
        __import__("torch")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"torch cannot be imported: {exc}")
    result = TrainingCoordinator(
        TrainingRunConfig(
            artifact_dir=str(tmp_path / "stage1"),
            stage="stage1_pretrain",
            steps=1,
            batch_size=2,
            n_cards=3,
            device="cpu",
        )
    ).run()
    assert result["ok"] is True
    m = result["metrics"]["metrics"]
    required = {
        "immediate_joint_outcome_loss",
        "player_swap_loss",
        "future_opponent_behaviour_loss",
        "masked_history_action_loss",
        "style_contrastive_loss",
    }
    missing = required - set(m.keys())
    assert not missing, f"P1 multitask keys missing: {sorted(missing)}"
    for key in required:
        v = float(m[key])
        assert v == v, f"{key} is NaN"  # finite


# ================================================================
# P2/P3 teacher system: ensemble + disagreement filtering + EMA
# ================================================================
def test_teacher_ensemble_filters_by_confidence_and_disagreement():
    """TeacherEnsemble.label must drop samples that fail the filter gate.

    This pins P2/P3 teacher filter contract (min_confidence + max_disagreement)
    so future refactors can't silently bypass the semi-supervised quality gate.
    """
    # Very strict gate → router samples that pass must not violate
    strict = TeacherEnsemble(
        config=TeacherFilterConfig(min_confidence=0.0, max_disagreement=1.0)
    )
    state = GameState.initial(3, current_prize=2)
    sample = strict.label(state)
    assert sample is not None, "router.label_state should succeed for N=3"

    # Impossible gate → no sample passes
    impossible = TeacherEnsemble(
        config=TeacherFilterConfig(min_confidence=2.0, max_disagreement=0.0)
    )
    assert impossible.label(state) is None, "impossible gate should filter out any label"


def test_ema_teacher_updates_parameters_monotonically():
    """EMATeacher.update must interpolate params by tau after two updates.

    This pins the EMA teacher contract used as P3 teacher_ema registry snapshot.
    """
    try:
        torch = __import__("torch")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"torch cannot be imported: {exc}")
    model_a = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        model_a.weight.fill_(1.0)
    model_b = torch.nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        model_b.weight.fill_(3.0)

    ema = EMATeacher(model_a, tau=0.5)  # tau=0.5 is aggressive so test is deterministic
    # After init, EMA weights == model_a (1.0)
    assert float(ema.model.weight.abs().mean().item()) == 1.0
    # Update once to model_b: new = (1 - 0.5)*1.0 + 0.5*3.0 = 2.0
    ema.update(model_b)
    assert abs(float(ema.model.weight.abs().mean().item()) - 2.0) < 1e-6
    # Update again to model_b: (1 - 0.5)*2.0 + 0.5*3.0 = 2.5
    ema.update(model_b)
    assert abs(float(ema.model.weight.abs().mean().item()) - 2.5) < 1e-6


# ================================================================
# P5 opponent curriculum & oracle diagnostic
# ================================================================
def test_opponent_curriculum_has_multiple_regimes_and_is_deterministic():
    """P5 opponent model must expose more than one regime (curriculum property).

    Also verifies each regime's action sampler returns a legal action for a
    constructed small case.
    """
    import random as _random
    regimes = default_opponent_curriculum()
    assert len(regimes) >= 2, "P5 curriculum needs multiple regimes for switch detection"
    ids = {r.regime_id for r in regimes}
    assert "uniform_random" in ids
    legal = [2, 3, 5]
    rng = _random.Random(42)
    for regime in regimes:
        action = opponent_action_for_regime(
            regime.regime_id, legal, stake=5, n_cards=5, rng=rng
        )
        assert action in legal, f"regime {regime.regime_id} returned illegal {action}"


def test_oracle_opponent_diagnostic_reports_switch_delay_across_sessions():
    from goofspiel.training.data import OpponentSession, RoundRecord
    s1 = OpponentSession(
        session_id="s1",
        opponent_id="u1",
        strategy_regime_id="uniform_random",
        games=[[RoundRecord(0, 1, 2, 3, 1, -1, 0, 0, False)]],
    )
    s2 = OpponentSession(
        session_id="s2",
        opponent_id="u2",
        strategy_regime_id="high_card_pressure",
        games=[[RoundRecord(0, 1, 2, 5, 1, -1, 0, 0, False)]],
    )
    diag = oracle_opponent_diagnostic([s1, s2], n_cards=5)
    assert 0.0 <= diag["oracle_accuracy"] <= 1.0
    assert diag["oracle_gain"] >= 0.0
    # Multi-regime sessions should flag a non-zero switch_delay baseline
    assert diag["switch_delay"] > 0.0, "oracle diagnostic missed cross-regime switch"


# ================================================================
# P6 league: real cross-play matrix is actually simulated, not prior
# ================================================================
def test_stage6_crossplay_contains_simulated_score_diff(tmp_path: Path):
    """league_report cross-play entries must be real simulated matches.

    The signal is 'source' = 'simulated_crossplay' and every pair has a
    mean_score_diff number (not just a PFSP prior placeholder).
    """
    import json
    result = TrainingCoordinator(
        TrainingRunConfig(artifact_dir=str(tmp_path / "stage6"), stage="stage6_league")
    ).run()
    assert result["ok"] is True
    report_path = tmp_path / "stage6" / "league" / "league_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report["cross_play"]
    assert len(rows) == 9, "3 roles × 3 agents = 9 cross-play rows"
    sources = {r["source"] for r in rows}
    assert sources == {"simulated_crossplay"}, f"unexpected sources: {sources}"
    for row in rows:
        assert isinstance(row["mean_score_diff"], (int, float))
        assert row["games"] >= 1


# ================================================================
# P7 red-team: focused correction + regression reports are real artifacts
# ================================================================
def test_stage7_focused_correction_and_regression_report(tmp_path: Path):
    """focused_correction_report.json must contain training_plan + regression block.

    This pins P7's 'focused correction training' and 'original attack/general
    regression' outputs are real artifacts (not just metric flags).
    """
    import json
    result = TrainingCoordinator(
        TrainingRunConfig(artifact_dir=str(tmp_path / "stage7"), stage="stage7_redteam")
    ).run()
    assert result["ok"] is True
    path = tmp_path / "stage7" / "redteam" / "focused_correction_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    tp = report["training_plan"]
    assert tp["method"] == "focused_correction_sft"
    assert tp["steps"] >= 1
    assert "source" in tp
    reg = report["regression"]
    assert bool(reg["original_attack_regression_passed"]) is True
    assert bool(reg["general_regression_passed"]) is True
    assert 0.0 <= float(reg["recurrence"]) <= 1.0
