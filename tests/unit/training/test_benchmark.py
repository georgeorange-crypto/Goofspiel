from __future__ import annotations

from pathlib import Path

import pytest


def test_default_profile_is_smoke_and_non_binding():
    """Arena v1 delta #12: the DEFAULT EvaluationProfile is SMOKE (not QUICK), it
    keeps the legacy 16-games/3-seeds workload, and it never certifies. Re-executed
    against the real harness so the default is proven, not read off a constant."""
    from goofspiel.training.benchmark import (
        QUICK_TO_SMOKE_MIGRATION_NOTE,
        EvaluationProfile,
        run_unified_benchmark,
    )

    default = EvaluationProfile()
    assert default.name == "SMOKE"
    # Legacy workload preserved by the migration.
    assert default.num_games == 16
    assert default.seeds == [1, 2, 3]
    assert default.emits_binding_promotion is False
    # The migration note exists verbatim as the plan requires.
    assert "intentionally reclassified from QUICK to SMOKE" in QUICK_TO_SMOKE_MIGRATION_NOTE

    report = run_unified_benchmark(EvaluationProfile(seeds=[1], num_games=1))
    assert report.profile.name == "SMOKE"
    assert report.promotion_decision == "NOT_EVALUATED"


def test_unified_benchmark_writes_required_reports(tmp_path):
    from goofspiel.training.benchmark import EvaluationProfile, run_unified_benchmark, write_benchmark_report

    report = run_unified_benchmark(EvaluationProfile(name="SMOKE", seeds=[1], num_games=1))
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

    report = run_unified_benchmark(EvaluationProfile(name="SMOKE", seeds=[1], num_games=2))
    assert report.arenas["E2_N13_ROBUST"]["source"] == "heuristic_vs_random_reference"
    assert report.hard_gates["G2_exploitability"] is None
    # Phase 0 / delta #13: SMOKE is a smoke profile and NEVER certifies. Its
    # decision is NOT_EVALUATED regardless of gate states — a binding
    # PROMOTE/REJECT is only emitted by the FULL profile. (A None gate cannot
    # carry promotion either.)
    assert report.promotion_decision == "NOT_EVALUATED"


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
    profile = EvaluationProfile(name="SMOKE", seeds=[1, 2], num_games=6)
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


# --------------------------------------------------------------------------
# Phase 0: tri-state gate semantics (PASS / FAIL / NOT_RUN) and the profile
# gating of binding promotions. These re-execute the mapping/decision logic
# directly with constructed inputs — they never read a stored decision field.
# --------------------------------------------------------------------------


def test_gate_label_maps_tristate_and_never_collapses_none_to_fail():
    """The original bug was ``'PASS' if v else 'FAIL'``: None (unrun) is falsy and
    silently rendered as FAIL. gate_label must map None -> NOT_RUN distinctly."""
    from goofspiel.training.benchmark import GATE_FAIL, GATE_NOT_RUN, GATE_PASS, gate_label

    assert gate_label(True) == GATE_PASS
    assert gate_label(False) == GATE_FAIL
    assert gate_label(None) == GATE_NOT_RUN
    # The three labels are distinct — NOT_RUN is neither PASS nor FAIL.
    assert len({GATE_PASS, GATE_FAIL, GATE_NOT_RUN}) == 3
    assert gate_label(None) != GATE_FAIL


@pytest.mark.parametrize(
    "profile_name, gates, expected",
    [
        # QUICK never certifies, whatever the gates say.
        ("QUICK", {"g": True}, "NOT_EVALUATED"),
        ("QUICK", {"g": False}, "NOT_EVALUATED"),
        ("QUICK", {"g": None}, "NOT_EVALUATED"),
        ("quick", {"a": True, "b": True}, "NOT_EVALUATED"),
        # FULL with any unrun gate cannot certify -> INCOMPLETE (NOT a REJECT).
        ("FULL", {"a": True, "b": None}, "INCOMPLETE"),
        ("FULL", {"a": None}, "INCOMPLETE"),
        # FULL, every gate ran: all-pass promotes, any real fail rejects.
        ("FULL", {"a": True, "b": True}, "PROMOTE_CANDIDATE"),
        ("FULL", {"a": True, "b": False}, "REJECT_CANDIDATE"),
        # A real FAIL rejects even if another gate is unrun? No — unrun dominates:
        # you cannot reject on an incomplete evaluation. NOT_RUN -> INCOMPLETE.
        ("FULL", {"a": False, "b": None}, "INCOMPLETE"),
        # Whitespace/case-insensitive FULL detection.
        (" full ", {"a": True}, "PROMOTE_CANDIDATE"),
    ],
)
def test_decide_promotion_honours_profile_and_tristate(profile_name, gates, expected):
    from goofspiel.training.benchmark import EvaluationProfile, decide_promotion

    profile = EvaluationProfile(name=profile_name)
    assert decide_promotion(profile, gates) == expected


def test_not_run_gate_blocks_but_does_not_reject():
    """A NOT_RUN (None) gate must never be treated as a FAIL: on a FULL run it
    blocks certification (INCOMPLETE) but is not itself a rejection."""
    from goofspiel.training.benchmark import EvaluationProfile, decide_promotion

    full = EvaluationProfile(name="FULL")
    # All real gates pass, one gate unrun -> INCOMPLETE, never PROMOTE, never REJECT.
    decision = decide_promotion(full, {"g0": True, "g1": True, "g2": None})
    assert decision == "INCOMPLETE"
    assert decision not in ("PROMOTE_CANDIDATE", "REJECT_CANDIDATE")


def test_only_full_profile_emits_binding_promotion():
    from goofspiel.training.benchmark import EvaluationProfile

    assert EvaluationProfile(name="FULL").emits_binding_promotion is True
    assert EvaluationProfile(name="SMOKE").emits_binding_promotion is False
    assert EvaluationProfile(name="QUICK").emits_binding_promotion is False
    assert EvaluationProfile(name="smoke").emits_binding_promotion is False
    # The default profile (delta #12: SMOKE) is non-binding.
    assert EvaluationProfile().emits_binding_promotion is False


def test_summary_md_renders_not_run_never_fail(tmp_path):
    """Re-execute the report writer and read back the rendered summary.md: a None
    gate appears as NOT_RUN with the explanatory footnote, and never as FAIL."""
    from goofspiel.training.benchmark import (
        BENCHMARK_VERSION,
        BenchmarkReport,
        EvaluationProfile,
        write_benchmark_report,
    )
    from goofspiel.training.schema import SCHEMA_VERSION

    report = BenchmarkReport(
        benchmark_version=BENCHMARK_VERSION,
        schema_version=SCHEMA_VERSION,
        profile=EvaluationProfile(name="FULL", seeds=[1], num_games=1),
        arenas={},
        hard_gates={"G0": True, "G1": False, "G2": None},
        promotion_decision="INCOMPLETE",
    )
    paths = write_benchmark_report(report, tmp_path / "r")
    md = Path(paths["summary_md"]).read_text(encoding="utf-8")
    assert "- G0: PASS" in md
    assert "- G1: FAIL" in md
    assert "- G2: NOT_RUN" in md
    # The unrun gate did NOT get rendered as FAIL.
    assert "- G2: FAIL" not in md
    assert "NOT_RUN means no evaluator computed that gate yet" in md


# --------------------------------------------------------------------------
# Arena v1 Workstream C — evaluate profile migration + tri-state promotion.
# Acceptance items #17-#21 (plan §12) and delta #4 (fixed budget). Every test
# re-executes the real decision/harness; none reads a stored decision field.
# --------------------------------------------------------------------------


def test_acc17_smoke_profile_renders_not_evaluated():
    """#17: SMOKE -> NOT_EVALUATED, whatever the gates would say."""
    from goofspiel.training.benchmark import EvaluationProfile, decide_promotion

    smoke = EvaluationProfile(name="SMOKE")
    assert decide_promotion(smoke, {"g0": True, "g1": True}) == "NOT_EVALUATED"
    assert decide_promotion(smoke, {"g0": False}) == "NOT_EVALUATED"
    assert decide_promotion(smoke, {"g0": None}) == "NOT_EVALUATED"


def test_acc18_quick_profile_still_renders_not_evaluated():
    """#18: QUICK is retained as a valid name and is also non-binding."""
    from goofspiel.training.benchmark import EvaluationProfile, decide_promotion

    quick = EvaluationProfile(name="QUICK")
    assert decide_promotion(quick, {"g0": True, "g1": True}) == "NOT_EVALUATED"
    assert decide_promotion(quick, {"g0": None}) == "NOT_EVALUATED"


def test_acc19_full_with_all_required_gates_run_certifies_pass_or_reject():
    """#19: FULL + every REQUIRED gate ran -> a binding PROMOTE or REJECT.

    Re-executes ``decide_promotion`` over the real REQUIRED_HARD_GATES set with
    fully-populated (non-None) verdicts: all-pass promotes, one real fail rejects.
    """
    from goofspiel.training.benchmark import (
        REQUIRED_HARD_GATES,
        EvaluationProfile,
        decide_promotion,
    )

    full = EvaluationProfile(name="FULL")
    all_pass = {g: True for g in REQUIRED_HARD_GATES}
    assert decide_promotion(full, all_pass) == "PROMOTE_CANDIDATE"

    one_fail = dict(all_pass)
    one_fail["G4_regression_suite"] = False
    assert decide_promotion(full, one_fail) == "REJECT_CANDIDATE"


def test_acc20_full_with_any_required_gate_not_run_is_incomplete():
    """#20: FULL + any required gate NOT_RUN -> INCOMPLETE (never REJECT).

    Two re-executions: (a) the abstract decision over the required set with one
    None, and (b) the LIVE harness with no checkpoint, which is honestly
    INCOMPLETE today because G1/G3..G7 have no evaluator yet — that is correct,
    not a bug, and the test pins it so a future 'shortcut' to PROMOTE is caught.
    """
    from goofspiel.training.benchmark import (
        REQUIRED_HARD_GATES,
        EvaluationProfile,
        run_unified_benchmark,
        decide_promotion,
    )

    full = EvaluationProfile(name="FULL")
    gates = {g: True for g in REQUIRED_HARD_GATES}
    gates["G3_historical"] = None
    decision = decide_promotion(full, gates)
    assert decision == "INCOMPLETE"
    assert decision != "REJECT_CANDIDATE"

    # Live FULL run, no model: several required gates are genuinely NOT_RUN.
    live = run_unified_benchmark(EvaluationProfile(name="FULL", seeds=[1], num_games=1))
    assert set(live.hard_gates) == set(REQUIRED_HARD_GATES)
    assert any(v is None for v in live.hard_gates.values())
    assert live.promotion_decision == "INCOMPLETE"


def test_acc21_not_run_gate_never_renders_as_fail_in_summary(tmp_path):
    """#21: a NOT_RUN gate is never rendered FAIL. Re-execute the LIVE FULL
    harness and read back summary.md — the genuinely-unrun gates show NOT_RUN."""
    from goofspiel.training.benchmark import (
        EvaluationProfile,
        run_unified_benchmark,
        write_benchmark_report,
    )

    report = run_unified_benchmark(EvaluationProfile(name="FULL", seeds=[1], num_games=1))
    paths = write_benchmark_report(report, tmp_path / "full")
    md = Path(paths["summary_md"]).read_text(encoding="utf-8")
    for name, value in report.hard_gates.items():
        if value is None:
            assert f"- {name}: NOT_RUN" in md
            assert f"- {name}: FAIL" not in md
    # And the INCOMPLETE reason names the unrun required gates.
    assert "required gates did not run" in md


def test_smoke_summary_records_the_verbatim_migration_note(tmp_path):
    """delta #12: a SMOKE report's summary.md carries the migration note verbatim,
    so the QUICK->SMOKE reclassification is on the artifact record."""
    from goofspiel.training.benchmark import (
        QUICK_TO_SMOKE_MIGRATION_NOTE,
        EvaluationProfile,
        run_unified_benchmark,
        write_benchmark_report,
    )

    report = run_unified_benchmark(EvaluationProfile(name="SMOKE", seeds=[1], num_games=1))
    paths = write_benchmark_report(report, tmp_path / "smoke")
    md = Path(paths["summary_md"]).read_text(encoding="utf-8")
    assert QUICK_TO_SMOKE_MIGRATION_NOTE in md
    assert "NOT_EVALUATED" in md


def test_delta4_evaluate_profile_is_fixed_budget_rejects_sequential_stop():
    """delta #4: the evaluate profile is fixed-budget on every profile. The field
    exists and defaults False, and constructing one with True is rejected — the
    harness has no early-stop path, so the guarantee is enforced, not decorative."""
    import pytest

    from goofspiel.training.benchmark import EvaluationProfile

    for name in ("SMOKE", "QUICK", "FULL"):
        assert EvaluationProfile(name=name).sequential_ci_stop is False
    with pytest.raises(ValueError, match="fixed-budget"):
        EvaluationProfile(name="FULL", sequential_ci_stop=True)
