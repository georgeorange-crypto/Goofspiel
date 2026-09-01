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


def test_stage5_trains_opponent_model_behind_firewall(tmp_path: Path):
    result = TrainingCoordinator(
        TrainingRunConfig(
            artifact_dir=str(tmp_path / "stage5"),
            stage="stage5_adaptive",
            steps=8,
            n_cards=3,
        )
    ).run()
    assert result["ok"] is True
    metrics = result["metrics"]["metrics"]
    checkpoint = result["metrics"]["checkpoint"]
    # Phase 3.2: P5 now TRAINS an opponent model — a real checkpoint exists and
    # the model is usable because it beat the honest uniform reference.
    assert metrics["opponent_model_usable"] == 1.0
    assert checkpoint is not None
    assert (tmp_path / "stage5" / "stage5_adaptive.pt").exists()
    assert metrics["opponent_sessions"] == 8.0
    assert metrics["opponent_regimes"] == 3.0
    assert metrics["oracle_gain"] >= 0.0
    # A trained model's calibration is now measured (NLL beats uniform, ECE real).
    assert metrics["opponent_nll"] < metrics["uniform_reference_nll"]
    assert metrics["nll_gain_over_uniform"] > 0.0
    assert 0.0 <= metrics["opponent_ece"] <= 1.0
    # Phase 3.2b firewall: robust params provably unchanged, adaptive got gradient.
    assert metrics["robust_param_delta_l1"] == 0.0
    assert metrics["adaptive_grad_norm_last"] > 0.0
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
    # Phase 4.4: a real focused correction now runs, so the regression pass/fail
    # metrics are MEASURED (0.0/1.0), no longer absent. They must be present and
    # valued, and the correction must not degrade the attack match-rate.
    assert metrics["original_attack_regression_passed"] in (0.0, 1.0)
    assert metrics["general_regression_passed"] in (0.0, 1.0)
    assert metrics["attack_match_rate_after"] >= metrics["attack_match_rate_before"]
    # A focused correction that trains toward the teacher action must not raise
    # the teacher-action NLL (it should fall or hold).
    assert metrics["mean_teacher_nll_after"] <= metrics["mean_teacher_nll_before"] + 1e-6
    assert (tmp_path / "stage7" / "redteam" / "redteam_report.json").exists()
    assert (tmp_path / "stage7" / "redteam" / "focused_correction_report.json").exists()
    assert (tmp_path / "stage7" / "redteam" / "stage7_corrected.pt").exists()


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


def test_stage6_league_plays_real_distinct_checkpoints(tmp_path: Path):
    """Phase 4.3: league agents reference REAL, loadable, DISTINCT checkpoints
    and cross-play is model-vs-model — re-executed, not read from a field.

    Before 4.3 every agent had `checkpoint_path=None` and cross-play secretly
    substituted role-keyed handcrafted baselines while labelling the rows
    `simulated_crossplay`.  This test:
      1. asserts each agent's checkpoint exists and loads as a GoofspielModel;
      2. asserts the three snapshots are genuinely distinct (different weights);
      3. RE-PLAYS one cross-play pair through the same primitive and reproduces
         the reported score — proving the rows are real model-vs-model matches.
    """
    try:
        __import__("torch")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"torch cannot be imported: {exc}")
    import json

    import torch

    from goofspiel.training.model_eval import load_model_from_checkpoint
    from goofspiel.training.stages import _CheckpointPolicy, _play_policy_match

    result = TrainingCoordinator(
        TrainingRunConfig(artifact_dir=str(tmp_path / "stage6"), stage="stage6_league", n_cards=3)
    ).run()
    assert result["ok"] is True
    report = json.loads((tmp_path / "stage6" / "league" / "league_report.json").read_text(encoding="utf-8"))

    # 1. Every agent references a real, loadable checkpoint (no None placeholders).
    agent_ckpts = report["agent_checkpoints"]
    assert len(agent_ckpts) == 3
    models = {}
    for agent_id, ckpt in agent_ckpts.items():
        assert ckpt, f"agent {agent_id} has no checkpoint (Phase 4.3 regression)"
        assert Path(ckpt).exists(), f"checkpoint missing on disk: {ckpt}"
        model, _meta = load_model_from_checkpoint(ckpt)
        models[agent_id] = model

    # 2. The three snapshots are genuinely distinct trained agents.
    states = [dict(model.state_dict()) for model in models.values()]
    any_pair_differs = False
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            for key in states[i]:
                if not torch.equal(states[i][key], states[j][key]):
                    any_pair_differs = True
                    break
    assert any_pair_differs, "league snapshots are byte-identical — not distinct agents"

    # 3. Re-execute a cross-play pair and reproduce its reported score.
    rows = report["cross_play"]
    assert len(rows) == 9
    row = rows[0]
    row_pol = _CheckpointPolicy(row["row_checkpoint"], temperature=0.5)
    col_pol = _CheckpointPolicy(row["col_checkpoint"], temperature=0.5)
    replayed = _play_policy_match(
        row_pol,
        col_pol,
        n_cards=3,
        seed=int(report["crossplay_seed_base"]),
    )
    assert replayed == pytest.approx(row["mean_score_diff"]), "cross-play row is not reproducible model play"

    # Handcrafted baselines survive ONLY as a clearly-labelled reference block.
    ref_rows = report["reference_play"]
    assert len(ref_rows) == 3
    assert {r["source"] for r in ref_rows} == {"trained_vs_reference_baseline"}


# ================================================================
# P7 red-team: focused correction + regression reports are real artifacts
# ================================================================
def test_stage7_focused_correction_and_regression_report(tmp_path: Path):
    """Phase 4.4: focused_correction_report.json carries a MEASURED regression,
    and this test RE-EXECUTES it rather than trusting the reported fields.

    The regression the report claims (match-rate / teacher-action argmax before
    and after the focused correction) is reproduced here by:
      1. reconstructing the exact three attack states P7 corrects on;
      2. loading the report's init (before) and corrected (after) checkpoints;
      3. re-playing the robust policy on each attack state through the 0.1
         harness (``robust_policy_fn``) and recomputing the argmax card;
      4. asserting the recomputed match-rate reproduces the report AND that the
         correction did not degrade it (after >= before).
    This makes the pass/fail a re-run fact, not a JSON literal.
    """
    try:
        __import__("torch")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"torch cannot be imported: {exc}")
    import json

    from goofspiel.training.checkpoint import load_checkpoint
    from goofspiel.training.model_eval import robust_policy_fn
    from goofspiel.models import GoofspielModel

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
    # Phase 4.4: the regression is now MEASURED, so the booleans are real, not null.
    assert reg["original_attack_regression_passed"] in (True, False)
    assert reg["general_regression_passed"] in (True, False)
    assert reg["recurrence"] in (True, False)
    assert reg["match_rate_after"] >= reg["match_rate_before"]

    # ---- RE-EXECUTE the regression from the saved checkpoints -----------------
    # The exact three attack states P7 corrects on (mirrors run_stage7_redteam).
    attack_states = [
        GameState.initial(3, current_prize=1),
        GameState.initial(3, current_prize=2),
        GameState(n=3, self_mask=0b011, opp_mask=0b110, prize_mask=0b100, current_prize=1, carry_pool=2, round_index=2),
    ]

    def _replay_match_rate(ckpt_path: str, per_state: list[dict]) -> tuple[float, list[int]]:
        model = GoofspielModel(max_cards=13)
        model.load_state_dict(load_checkpoint(ckpt_path)["model_state"])
        model.eval()
        fn = robust_policy_fn(model, greedy=False, temperature=1.0)
        matched = 0
        argmaxes = []
        for entry, state in zip(per_state, attack_states):
            dist = fn(state)
            argmax_card = max(dist, key=lambda c: (dist.get(c, 0.0), -c))
            argmaxes.append(argmax_card)
            if argmax_card == entry["teacher_card"]:
                matched += 1
        return matched / len(attack_states), argmaxes

    before_ckpt = tp["init_checkpoint"]
    after_ckpt = tp["corrected_checkpoint"]
    assert Path(after_ckpt).exists()
    # Re-play the AFTER checkpoint and reproduce the report's match-rate + argmaxes.
    after_rate, after_argmaxes = _replay_match_rate(after_ckpt, reg["after"]["per_state"])
    assert after_rate == pytest.approx(reg["match_rate_after"]), "after match-rate is not reproducible replay"
    for entry, argmax_card in zip(reg["after"]["per_state"], after_argmaxes):
        assert entry["argmax_card"] == argmax_card, "reported after argmax card is not reproducible"
    # Re-play the BEFORE checkpoint and reproduce the report's before match-rate.
    if Path(before_ckpt).exists():
        before_rate, _ = _replay_match_rate(before_ckpt, reg["before"]["per_state"])
        assert before_rate == pytest.approx(reg["match_rate_before"]), "before match-rate is not reproducible replay"
        # The correction genuinely did not make the attack states worse.
        assert after_rate >= before_rate
