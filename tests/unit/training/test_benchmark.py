from __future__ import annotations

from pathlib import Path

import pytest


def test_unified_benchmark_writes_required_reports(tmp_path):
    from goofspiel.training.benchmark import EvaluationProfile, run_unified_benchmark, write_benchmark_report

    report = run_unified_benchmark(EvaluationProfile(name="QUICK", seeds=[1], num_games=1))
    paths = write_benchmark_report(report, tmp_path / "reports" / "candidate")
    assert report.benchmark_version
    assert "E0_MATHEMATICAL_CORRECTNESS" in report.arenas
    assert "E5_SEARCH_COMPUTE" in report.arenas
    assert report.arenas["E0_MATHEMATICAL_CORRECTNESS"]["passed"] is True
    assert report.arenas["E5_SEARCH_COMPUTE"]["rows"]
    assert (tmp_path / "reports" / "candidate" / "summary.json").exists()
    assert paths["summary_md"].endswith("summary.md")
    for key in ("main_table", "search_table", "adaptive_table", "opponent_table", "generalization_table"):
        assert Path(paths[key]).exists()


def test_benchmark_without_model_marks_e2_reference_and_g2_unrun():
    """Phase 5: with no checkpoint, E2 is the Heuristic-vs-Random REFERENCE (so
    labelled) and G2 is unrun (None) — never a literal derived from a reference
    row. This is the honesty half of Phase 5's benchmark fix."""
    from goofspiel.training.benchmark import EvaluationProfile, run_unified_benchmark

    report = run_unified_benchmark(EvaluationProfile(name="QUICK", seeds=[1], num_games=2))
    assert report.arenas["E2_N13_ROBUST"]["source"] == "heuristic_vs_random_reference"
    assert report.hard_gates["G2_exploitability"] is None
    # A None gate cannot carry promotion.
    assert report.promotion_decision == "REJECT_CANDIDATE"


def test_benchmark_with_model_makes_e2_real_play_and_g2_computed(tmp_path):
    """Phase 5: with a trained checkpoint, E2/E6 are the model's REAL play vs
    Random (not the reference), and G2 is a computed robustness verdict about the
    checkpoint. This test RE-PLAYS the same policy vs Random and reproduces the
    reported E2 mean score-diff — the arena number is real play, not a literal."""
    try:
        __import__("torch")
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"torch cannot be imported: {exc}")
    from statistics import mean

    from goofspiel.training.benchmark import EvaluationProfile, run_unified_benchmark
    from goofspiel.training.model_eval import (
        load_model_from_checkpoint,
        play_policy_vs_bot,
        robust_policy_fn,
    )
    from goofspiel.training.stages import run_stage1_pretrain

    ckpt = run_stage1_pretrain(steps=1, batch_size=4, out_dir=tmp_path / "ck", n_cards=5).checkpoint
    profile = EvaluationProfile(name="QUICK", seeds=[1, 2], num_games=6)
    report = run_unified_benchmark(profile, checkpoint=ckpt)

    e2 = report.arenas["E2_N13_ROBUST"]
    assert e2["source"] == "trained_model_vs_random"
    assert e2["checkpoint"] == str(ckpt)
    # G2 is now a real boolean about the trained checkpoint (not None, not a
    # reference-derived literal): it equals "beat Random on the computed diff".
    assert report.hard_gates["G2_exploitability"] in (True, False)
    assert report.hard_gates["G2_exploitability"] == bool(e2["mean_score_diff"] > 0.0)

    # RE-EXECUTE E2: replay the same robust policy vs Random over the same seeds
    # and reproduce the arena's mean score-diff.
    model, _meta = load_model_from_checkpoint(ckpt)
    policy = robust_policy_fn(model, greedy=True)
    replay = mean(
        play_policy_vs_bot(policy, "random", n_cards=13, num_games=profile.num_games, seed=s)["mean_score_diff"]
        for s in profile.seeds
    )
    assert e2["mean_score_diff"] == pytest.approx(replay), "E2 is not reproducible real model play"
    # E6 rows are also real model play now.
    for row in report.arenas["E6_GENERALIZATION"]["rows"]:
        assert isinstance(row["robust_score"], (int, float))
