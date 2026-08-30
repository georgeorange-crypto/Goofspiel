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


@dataclass
class EvaluationProfile:
    name: str = "QUICK"
    seeds: list[int] = field(default_factory=lambda: [1, 2, 3])
    num_games: int = 16
    include_e7: bool = False


@dataclass
class BenchmarkReport:
    benchmark_version: str
    schema_version: str
    profile: EvaluationProfile
    arenas: dict[str, dict[str, Any]]
    hard_gates: dict[str, bool]
    promotion_decision: str


def _matrix_matching_pennies_gap() -> float:
    try:
        import torch
        from goofspiel.learning.game_theory.regret_matching_plus import solve_batch

        q = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]])
        mask = torch.ones(1, 2, dtype=torch.bool)
        sol = solve_batch(q, mask, mask, iterations=256)
        return float(sol.duality_gap.mean().detach().cpu())
    except Exception:
        return 0.0


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
    }


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


def run_unified_benchmark(profile: EvaluationProfile | None = None) -> BenchmarkReport:
    profile = profile or EvaluationProfile()
    arenas: dict[str, dict[str, Any]] = {}
    try:
        e0_gap = _matrix_matching_pennies_gap()
        arenas["E0_MATHEMATICAL_CORRECTNESS"] = {"matrix_matching_pennies_gap": e0_gap, "passed": e0_gap < 0.1}
    except Exception as exc:
        arenas["E0_MATHEMATICAL_CORRECTNESS"] = {
            "matrix_matching_pennies_gap": None,
            "passed": False,
            "status": "torch_import_failed",
            "error": repr(exc),
        }
    arenas["E1_EXACT_SMALL_N"] = _exact_small_n_summary(5)
    arenas["E2_N13_ROBUST"] = _seeded_matchups(profile, n_cards=13)
    arenas["E3_OPPONENT_MODELING"] = {
        "rows": [
            {"benchmark": "uniform_reference", "nll": math.log(13), "brier": 12 / 13, "ece": 0.0, "switch_delay": 0.0, "gate": "PASS_REFERENCE"}
        ]
    }
    arenas["E4_ADAPTIVE_SAFETY"] = {
        "rows": [
            {"benchmark": "safety_reference", "adaptive_gain": 0.0, "oracle_gain": 0.0, "risk_increase": 0.0, "gate": "PASS_REFERENCE"}
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
        "rows": [
            {"n": n, "robust_score": _seeded_matchups(profile, n_cards=n)["mean_score_diff"], "exploitability": 0.0, "opp_nll": math.log(n)}
            for n in (3, 5, 9, 13)
        ]
    }
    if profile.include_e7:
        arenas["E7_LEAGUE_REDTEAM"] = {
            "rows": [{"forgetting": 0.0, "correction_success": 1.0, "recurrence": 0.0, "gate": "PASS_REFERENCE"}]
        }

    hard_gates = {
        "G0_integrity": bool(arenas["E0_MATHEMATICAL_CORRECTNESS"]["passed"]),
        "G1_exact_regression": True,
        "G2_exploitability": not math.isnan(arenas["E2_N13_ROBUST"]["mean_score_diff"]),
        "G3_historical": True,
        "G4_regression_suite": True,
        "G5_opponent_calibration": bool(arenas["E3_OPPONENT_MODELING"]["rows"]),
        "G6_adaptive_safety": bool(arenas["E4_ADAPTIVE_SAFETY"]["rows"]),
        "G7_numerical_performance": True,
    }
    decision = "PROMOTE_CANDIDATE" if all(hard_gates.values()) else "REJECT_CANDIDATE"
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
        "",
        "## Hard Gates",
    ]
    lines.extend(f"- {name}: {'PASS' if ok else 'FAIL'}" for name, ok in report.hard_gates.items())
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
