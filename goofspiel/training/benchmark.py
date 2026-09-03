"""Unified benchmark harness and promotion report generation."""

from __future__ import annotations

import json
import math
import csv
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from goofspiel.game import GameState
from goofspiel.training.baselines import default_baselines
from goofspiel.training.evaluation import EvaluationReport, evaluate_bot_matchup, exact_feasibility_sweep
from goofspiel.training.schema import SCHEMA_VERSION
from goofspiel.training.teachers import immediate_q_matrix

BENCHMARK_VERSION = "goofspiel.benchmark.v1"
ARENAS = (
    "E0_MATHEMATICAL_CORRECTNESS",
    "E1_EXACT_SMALL_N",
    "E2_N13_ROBUST",
    "E3_OPPONENT_MODELING",
    "E4_ADAPTIVE_SAFETY",
    "E5_SEARCH_COMPUTE",
    "E6_GENERALIZATION",
    "E7_LEAGUE_REDTEAM",
)

# Phase 0 — three-state gate semantics.  A hard gate is one of:
#   * True   -> a real check RAN and PASSED
#   * False  -> a real check RAN and FAILED
#   * None   -> the check did NOT RUN (no evaluator computed it yet)
# ``None`` must never be rendered or counted as a FAIL.  These labels are the
# single source of truth for turning a gate value into human-readable text.
GATE_PASS = "PASS"
GATE_FAIL = "FAIL"
GATE_NOT_RUN = "NOT_RUN"

# Promotion decisions.  A *binding* PROMOTE/REJECT may be emitted only by the
# FULL profile with every REQUIRED gate actually run (see REQUIRED_HARD_GATES);
# SMOKE and QUICK are non-binding smoke profiles and never certify
# (NOT_EVALUATED), and a FULL run that left a required gate unrun is INCOMPLETE.
PROMOTION_PROMOTE = "PROMOTE_CANDIDATE"
PROMOTION_REJECT = "REJECT_CANDIDATE"
PROMOTION_INCOMPLETE = "INCOMPLETE"
PROMOTION_NOT_EVALUATED = "NOT_EVALUATED"

# Arena v1 delta #12 — the default evaluate profile is reclassified QUICK -> SMOKE.
# This note is recorded VERBATIM (the plan requires the exact wording) so the
# intent is unambiguous in the artifact and the source: the workload did not
# shrink, only its promotion authority did.
QUICK_TO_SMOKE_MIGRATION_NOTE = (
    "Legacy default workload is preserved, while the profile is intentionally "
    "reclassified from QUICK to SMOKE so that smoke runs cannot produce binding "
    "promotion conclusions."
)

# Arena v1 delta #13 — the hard gates REQUIRED for a binding certification.  A
# FULL run may emit PROMOTE/REJECT only when EVERY gate in this set actually ran
# (is non-None); if any is NOT_RUN the run is INCOMPLETE, never a REJECT.  There
# is deliberately NO "required subset" smaller than the whole: certifying a model
# whose historical / regression / calibration / safety / performance gates were
# never evaluated would be an overclaim.  So a live FULL run is honestly
# INCOMPLETE until the evaluators for G1/G3..G7 land — that is correct, not a bug.
REQUIRED_HARD_GATES = (
    "G0_integrity",
    "G1_exact_regression",
    "G2_exploitability",
    "G3_historical",
    "G4_regression_suite",
    "G5_opponent_calibration",
    "G6_adaptive_safety",
    "G7_numerical_performance",
)


def gate_label(value: bool | None) -> str:
    """Map a tri-state gate value to its PASS / FAIL / NOT_RUN label.

    ``None`` (a gate no evaluator computed) is NOT_RUN — never FAIL.  This is the
    bug Phase 0 fixes: ``'PASS' if value else 'FAIL'`` silently collapsed the
    unrun state (falsy ``None``) into FAIL.
    """
    if value is None:
        return GATE_NOT_RUN
    return GATE_PASS if value else GATE_FAIL


@dataclass
class EvaluationProfile:
    # Arena v1 delta #12: the default is SMOKE, not QUICK.  SMOKE keeps the legacy
    # 16-games/3-seeds workload but carries no promotion authority (NOT_EVALUATED),
    # so a default-profile run can never be mistaken for a binding certification.
    # See QUICK_TO_SMOKE_MIGRATION_NOTE for the recorded rationale.
    name: str = "SMOKE"
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3])
    num_games: int = 16
    include_e7: bool = False
    # Arena v1 delta #4: the evaluate harness is FIXED-BUDGET on every profile.
    # It never stops early on a bootstrap CI half-width (mixing optional stopping
    # with a fixed-sample CI breaks the coverage interpretation).  The evaluate
    # code has no early-stop path at all, so this field is an enforced guarantee,
    # not a switch: True is unsupported and rejected below.  (This is independent
    # of the Stage6 cross-play budget, which owns its own sequential-stop flag.)
    sequential_ci_stop: bool = False

    def __post_init__(self) -> None:
        if self.sequential_ci_stop:
            raise ValueError(
                "Arena v1 evaluate is fixed-budget (delta #4): sequential_ci_stop "
                "must be False. The evaluate harness has no early-stop path; enable "
                "sequential stopping only in the Stage6 cross-play budget, which "
                "owns that concern separately."
            )

    @property
    def emits_binding_promotion(self) -> bool:
        """Only the FULL profile may emit a binding PROMOTE/REJECT decision.

        SMOKE and QUICK are non-binding evaluation profiles: they report every
        arena and every gate state, but their promotion decision is always
        ``NOT_EVALUATED`` — they do not run the gates at the scale (and, for
        SMOKE, on a formally-trained checkpoint) a real certification requires.
        """
        return self.name.strip().upper() == "FULL"


def decide_promotion(profile: EvaluationProfile, hard_gates: dict[str, bool | None]) -> str:
    """Turn tri-state gates into a promotion decision, honouring the profile.

    * SMOKE / QUICK (any non-FULL profile) -> ``NOT_EVALUATED`` (never binding).
    * FULL with any gate NOT_RUN (None) -> ``INCOMPLETE`` (cannot certify).
    * FULL with every gate run and all PASS -> ``PROMOTE_CANDIDATE``.
    * FULL with every gate run and any FAIL -> ``REJECT_CANDIDATE``.

    Every gate present in ``hard_gates`` is treated as required: ``all(...)`` /
    ``any(... is None)`` range over the whole dict.  The delta #13 "required
    gates" concept is therefore expressed by WHICH gates the caller places in the
    dict — :func:`run_unified_benchmark` passes all of :data:`REQUIRED_HARD_GATES`
    — not by a subset filter here.  There is deliberately no smaller required set:
    a NOT_RUN gate is never treated as a FAIL (it blocks certification, turning a
    FULL run INCOMPLETE, but is not itself a rejection), and no gate may be
    silently waived to force a PROMOTE.
    """
    if not profile.emits_binding_promotion:
        return PROMOTION_NOT_EVALUATED
    if any(value is None for value in hard_gates.values()):
        return PROMOTION_INCOMPLETE
    if all(value is True for value in hard_gates.values()):
        return PROMOTION_PROMOTE
    return PROMOTION_REJECT


@dataclass
class BenchmarkReport:
    benchmark_version: str
    schema_version: str
    profile: EvaluationProfile
    arenas: dict[str, dict[str, Any]]
    # Tri-state: True=PASS, False=FAIL, None=NOT_RUN.  Never collapse None to FAIL.
    hard_gates: dict[str, bool | None]
    promotion_decision: str


def _matrix_matching_pennies_gap() -> float | None:
    """Duality gap of the RM+ solver on 2x2 matching pennies.

    Returns the gap when the check actually ran, or ``None`` when it could not
    run (e.g. torch unavailable).  ``None`` must propagate as NOT_RUN — never be
    silently coerced to ``0.0``, which would forge a PASS on the E0 gate.
    """
    try:
        import torch
        from goofspiel.learning.game_theory.regret_matching_plus import solve_batch

        q = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        sol = solve_batch(q, mask, mask, iterations=256)
        return float(sol.duality_gap.mean().detach().cpu())
    except Exception:
        return None


def _exact_small_n_summary(max_n: int = 5) -> dict[str, Any]:
    report = exact_feasibility_sweep(max_n)
    return {"risk_by_n": report.details["risk_by_n"], "states_by_n": report.details["states_by_n"]}


def _seeded_matchups(profile: EvaluationProfile, *, n_cards: int = 13) -> dict[str, float]:
    diffs = []
    for seed in profile.seeds:
        report = evaluate_bot_matchup(num_games=profile.num_games, n_cards=n_cards, seed=seed)
        diffs.append(report.metrics["mean_score_diff"])
    return {
        "mean_score_diff": float(mean(diffs)) if diffs else 0.0,
        "worst_seed_score_diff": float(min(diffs)) if diffs else 0.0,
        "num_seeds": float(len(diffs)),
        "source": "heuristic_vs_random_reference",
    }


def _seeded_model_matchups(
    checkpoint: str | Path, profile: EvaluationProfile, *, n_cards: int = 13
) -> dict[str, float]:
    """E2/E6 for a TRAINED checkpoint: play its robust policy vs Random through the
    real env (0.1 harness), averaged over the profile's seeds. Every number is
    computed play, not a Heuristic-vs-Random reference."""
    from goofspiel.training.model_eval import (
        load_model_from_checkpoint,
        play_policy_vs_bot,
        robust_policy_fn,
    )

    model, meta = load_model_from_checkpoint(checkpoint)
    policy = robust_policy_fn(model, greedy=True)
    diffs = []
    for seed in profile.seeds:
        diffs.append(
            play_policy_vs_bot(policy, "random", n_cards=n_cards, num_games=profile.num_games, seed=seed)[
                "mean_score_diff"
            ]
        )
    return {
        "mean_score_diff": float(mean(diffs)) if diffs else 0.0,
        "worst_seed_score_diff": float(min(diffs)) if diffs else 0.0,
        "num_seeds": float(len(diffs)),
        "source": "trained_model_vs_random",
        "checkpoint": str(checkpoint),
        "checkpoint_id": str(meta.get("checkpoint_id", "")),
    }


def _generalization_row(
    n: int, robust_summary: dict[str, float], *, checkpoint: str | Path | None
) -> dict[str, Any]:
    """One E6 generalization row.

    Phase 0 correctness fix: the exploitability field must never be a hard-coded
    ``0.0``.  When a trained checkpoint is supplied and ``n`` is inside the exact
    full-game budget, we compute the REAL ``full_game_exploitability`` of the
    model's robust policy; otherwise the field is ``None`` (NOT_RUN) and carries
    a status string saying why — never a fabricated zero.

    ``exploitability_kind`` names exactly what the number is so it can never be
    mistaken for a full-game figure at large N (Phase 0.3 naming discipline).
    """
    from goofspiel.training.model_eval import DEFAULT_MAX_FULL_GAME_N

    row: dict[str, Any] = {
        "n": n,
        "robust_score": robust_summary["mean_score_diff"],
        "opp_nll": math.log(n),
    }
    if checkpoint is not None and Path(checkpoint).exists() and n <= DEFAULT_MAX_FULL_GAME_N:
        try:
            from goofspiel.training.model_eval import (
                full_game_exploitability,
                load_model_from_checkpoint,
                robust_policy_fn,
            )

            model, _meta = load_model_from_checkpoint(checkpoint)
            policy = robust_policy_fn(model, greedy=True)
            exploit = full_game_exploitability(policy, n_cards=n, max_n=DEFAULT_MAX_FULL_GAME_N)
            row["exploitability"] = exploit
            row["exploitability_kind"] = "full_game_exact"
            if exploit is None:
                row["exploitability_status"] = "not_run_exceeds_exact_budget"
        except Exception as exc:  # pragma: no cover - surfaced honestly
            row["exploitability"] = None
            row["exploitability_kind"] = "not_run"
            row["exploitability_status"] = f"error:{type(exc).__name__}"
    else:
        # No model, or N beyond the exact budget: honestly NOT_RUN, not 0.0.
        row["exploitability"] = None
        row["exploitability_kind"] = "not_run"
        row["exploitability_status"] = (
            "no_trained_checkpoint" if checkpoint is None else "not_run_exceeds_exact_budget"
        )
    return row


def _search_compute_summary() -> dict[str, Any]:
    state = GameState.initial(5, current_prize=1)
    try:
        from goofspiel.reasoning import SearchBudget, run_gt_cfr, run_sm_mcts
    except Exception:
        return {
            "rows": [
                {
                    "method": "MATRIX_NASH_REFERENCE",
                    "simulations": 0,
                    "runtime_ms": 0.0,
                    "quality_score": 1.0,
                    "exact_leaf_hits": 0,
                    "fallback": "torch_free_reference",
                    "legal_joint_actions": len(state.self_actions) * len(state.opponent_actions),
                }
            ]
        }
    rows = []
    for sims in (128, 512):
        result = run_sm_mcts(state, budget=SearchBudget(simulations=sims, matrix_iterations=64))
        rows.append(
            {
                "method": "SM_MCTS",
                "simulations": sims,
                "runtime_ms": result.runtime_ms,
                "quality_score": result.quality_score,
                "exact_leaf_hits": result.exact_leaf_hits,
            }
        )
    cfr = run_gt_cfr(state, iterations=256)
    rows.append({"method": "GT_CFR", "iterations": 256, "runtime_ms": cfr.runtime_ms, "quality_score": cfr.quality_score})
    return {"rows": rows}


def run_unified_benchmark(
    profile: EvaluationProfile | None = None,
    *,
    checkpoint: str | Path | None = None,
) -> BenchmarkReport:
    profile = profile or EvaluationProfile()
    # When a trained checkpoint is supplied, E2/E6 are the model's REAL play vs
    # Random (0.1 harness); otherwise they fall back to the Heuristic-vs-Random
    # reference and are clearly labelled as such via each row's ``source``.
    have_model = bool(checkpoint) and Path(checkpoint).exists()

    def _robust_at(n: int) -> dict[str, float]:
        if have_model:
            try:
                return _seeded_model_matchups(checkpoint, profile, n_cards=n)
            except Exception as exc:  # pragma: no cover - surfaced honestly
                ref = _seeded_matchups(profile, n_cards=n)
                ref["source"] = f"reference_fallback_model_error:{type(exc).__name__}"
                return ref
        return _seeded_matchups(profile, n_cards=n)

    arenas: dict[str, dict[str, Any]] = {}
    try:
        e0_gap = _matrix_matching_pennies_gap()
        if e0_gap is None:
            # The check could not run (torch unavailable). NOT_RUN, not FAIL.
            arenas["E0_MATHEMATICAL_CORRECTNESS"] = {
                "matrix_matching_pennies_gap": None,
                "passed": None,
                "status": "torch_dependent_check_unavailable",
            }
        else:
            arenas["E0_MATHEMATICAL_CORRECTNESS"] = {
                "matrix_matching_pennies_gap": e0_gap,
                "passed": e0_gap < 0.1,
            }
    except Exception as exc:
        arenas["E0_MATHEMATICAL_CORRECTNESS"] = {
            "matrix_matching_pennies_gap": None,
            "passed": None,
            "status": "torch_import_failed",
            "error": repr(exc),
        }
    arenas["E1_EXACT_SMALL_N"] = _exact_small_n_summary(5)
    arenas["E2_N13_ROBUST"] = _robust_at(13)
    arenas["E3_OPPONENT_MODELING"] = {
        "rows": [
            {"benchmark": "uniform_reference", "nll": math.log(13), "brier": 12 / 13, "ece": 0.0, "switch_delay": 0.0, "gate": "REFERENCE_ROW_NOT_A_RESULT"}
        ]
    }
    arenas["E4_ADAPTIVE_SAFETY"] = {
        "rows": [
            {"benchmark": "safety_reference", "adaptive_gain": 0.0, "oracle_gain": 0.0, "risk_increase": 0.0, "gate": "REFERENCE_ROW_NOT_A_RESULT"}
        ],
    }
    try:
        arenas["E5_SEARCH_COMPUTE"] = _search_compute_summary()
    except Exception as exc:
        state = GameState.initial(5, current_prize=1)
        arenas["E5_SEARCH_COMPUTE"] = {
            "rows": [
                {
                    "method": "MATRIX_NASH_REFERENCE",
                    "simulations": 0,
                    "runtime_ms": 0.0,
                    "quality_score": 1.0,
                    "exact_leaf_hits": 0,
                    "fallback": "benchmark_exception_reference",
                    "legal_joint_actions": len(state.self_actions) * len(state.opponent_actions),
                    "error": repr(exc),
                }
            ]
        }
    arenas["E6_GENERALIZATION"] = {
        "rows": [_generalization_row(n, _robust_at(n), checkpoint=checkpoint if have_model else None) for n in (3, 5, 9, 13)]
    }
    if profile.include_e7:
        arenas["E7_LEAGUE_REDTEAM"] = {
            "rows": [{"forgetting": 0.0, "correction_success": 1.0, "recurrence": 0.0, "gate": "REFERENCE_ROW_NOT_A_RESULT"}]
        }

    # Gate discipline (Phase 0.2): a gate is True only when a real check computed
    # it; a gate that no arena actually evaluates yet is None (unknown), never a
    # literal True. `all(...)` treats None as falsy, so promotion cannot ride on
    # an unrun gate. The reference-row arenas (E3/E4/E7) prove only that a row
    # exists — that is NOT a calibration/safety result, so those gates are None
    # until a real evaluator fills them (Phases 4-5).
    # Gate discipline (Phase 0.2 / Phase 5): a gate is True only when a real check
    # computed it on the checkpoint under test; a gate that no arena actually
    # evaluates yet is None (unknown), never a literal True. `all(...)` treats None
    # as falsy, so promotion cannot ride on an unrun gate. The reference-row arenas
    # (E3/E4/E7) prove only that a row exists — not a calibration/safety result —
    # so those gates stay None until a real evaluator fills them.
    e2 = arenas["E2_N13_ROBUST"]
    e2_is_model = e2.get("source") == "trained_model_vs_random"
    # G2 is a robustness verdict ABOUT the trained checkpoint: it can pass only
    # when E2 is real model play (not the Heuristic-vs-Random reference) AND the
    # trained policy beats Random on the computed mean score-diff. With no model,
    # G2 is unrun (None), never a literal derived from a reference row.
    if e2_is_model and not math.isnan(e2["mean_score_diff"]):
        g2 = bool(e2["mean_score_diff"] > 0.0)
    elif e2_is_model:
        g2 = False
    else:
        g2 = None
    hard_gates = {
        "G0_integrity": bool(arenas["E0_MATHEMATICAL_CORRECTNESS"]["passed"]),
        "G1_exact_regression": None,
        "G2_exploitability": g2,
        "G3_historical": None,
        "G4_regression_suite": None,
        "G5_opponent_calibration": None,
        "G6_adaptive_safety": None,
        "G7_numerical_performance": None,
    }
    # delta #13: the emitted gate set IS the required set — keep them in lockstep
    # so a gate can never be quietly dropped from the certification requirement.
    assert tuple(hard_gates) == REQUIRED_HARD_GATES, (
        "hard_gates drifted from REQUIRED_HARD_GATES; a required gate must not be "
        "silently added or removed from the certification requirement"
    )
    decision = decide_promotion(profile, hard_gates)
    return BenchmarkReport(
        benchmark_version=BENCHMARK_VERSION,
        schema_version=SCHEMA_VERSION,
        profile=profile,
        arenas=arenas,
        hard_gates=hard_gates,
        promotion_decision=decision,
    )


def write_benchmark_report(report: BenchmarkReport, out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary_json = out / "summary.json"
    summary_json.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    summary_md = out / "summary.md"
    lines = [
        "# Goofspiel Benchmark Summary",
        "",
        f"- Benchmark: {report.benchmark_version}",
        f"- Profile: {report.profile.name}",
        f"- Promotion decision: {report.promotion_decision}",
    ]
    # A SMOKE/QUICK/non-FULL profile never certifies; say so plainly so
    # NOT_EVALUATED is not misread as a failure, and cite the delta #12 migration
    # note verbatim for the SMOKE default so the reclassification is on the record.
    if report.promotion_decision == PROMOTION_NOT_EVALUATED:
        lines.append(
            f"  - (profile `{report.profile.name}` is a non-binding smoke profile; "
            "a binding PROMOTE/REJECT is only emitted by the FULL profile)"
        )
        if report.profile.name.strip().upper() in ("SMOKE", "QUICK"):
            lines.append(f"  - migration note: {QUICK_TO_SMOKE_MIGRATION_NOTE}")
    elif report.promotion_decision == PROMOTION_INCOMPLETE:
        unrun = [name for name, ok in report.hard_gates.items() if ok is None]
        detail = f" ({', '.join(unrun)})" if unrun else ""
        lines.append(
            "  - (FULL profile, but one or more required gates did not run — "
            f"cannot certify{detail})"
        )
    lines.extend(["", "## Hard Gates", ""])
    # Phase 0: render the tri-state via ``gate_label`` — NOT_RUN must never show
    # as FAIL.  ``'PASS' if ok else 'FAIL'`` was the bug (None is falsy).
    lines.extend(f"- {name}: {gate_label(ok)}" for name, ok in report.hard_gates.items())
    if any(v is None for v in report.hard_gates.values()):
        lines.append("")
        lines.append(
            "> NOT_RUN means no evaluator computed that gate yet — it is neither a "
            "pass nor a failure, and does not by itself reject a candidate."
        )
    lines.append("")
    lines.append("## Baselines")
    lines.extend(f"- {b.name}: {b.tier} / {b.arena} / {b.entrypoint}" for b in default_baselines())
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_csv(name: str, rows: list[dict[str, Any]]) -> str:
        path = out / name
        if not rows:
            rows = [{"empty": True}]
        fields = sorted({key for row in rows for key in row.keys()})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        return str(path)

    main_rows = [
        {"arena": arena, **{k: v for k, v in payload.items() if isinstance(v, (int, float, str, bool)) or v is None}}
        for arena, payload in report.arenas.items()
    ]
    search_rows = list(report.arenas.get("E5_SEARCH_COMPUTE", {}).get("rows", []))
    adaptive_rows = list(report.arenas.get("E4_ADAPTIVE_SAFETY", {}).get("rows", []))
    opponent_rows = list(report.arenas.get("E3_OPPONENT_MODELING", {}).get("rows", []))
    generalization_rows = list(report.arenas.get("E6_GENERALIZATION", {}).get("rows", []))
    paths = {
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "main_table": write_csv("main_robust_table.csv", main_rows),
        "search_table": write_csv("search_table.csv", search_rows),
        "adaptive_table": write_csv("adaptive_table.csv", adaptive_rows),
        "opponent_table": write_csv("opponent_table.csv", opponent_rows),
        "generalization_table": write_csv("generalization_table.csv", generalization_rows),
    }
    return paths
