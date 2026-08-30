"""Stage 0: environment, solver, schema, and configuration verification."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from goofspiel.env import PLAYER_0, PLAYER_1, GoofspielEnv
from goofspiel.game import GameState, transition
from goofspiel.solver import estimate_complexity, solve_zero_sum_matrix
from goofspiel.training.data import GameCorpusSample, JsonlStore, state_record_from_game_state

Q_PRIORITY = ("EXACT", "CERTIFIED_SEARCH", "NASH_BELLMAN")
POLICY_PRIORITY = ("EXACT", "CERTIFIED_CFR_SEARCH", "REFERENCE_NASH_Q", "TRAINING_NASH_Q")


@dataclass
class VerificationReport:
    ok: bool
    checks: dict[str, bool] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _record(report: VerificationReport, name: str, ok: bool, error: str | None = None) -> None:
    report.checks[name] = bool(ok)
    if not ok and error:
        report.errors.append(f"{name}: {error}")


def run_stage0_verify(*, artifact_dir: str | Path = "artifacts/runs/stage0_verify") -> VerificationReport:
    report = VerificationReport(ok=True)

    try:
        env = GoofspielEnv(num_cards=3, rng=random.Random(20260830))
        obs = env.reset()
        assert obs["round"] == 1
        while not env.done:
            env.step({PLAYER_0: env.legal_actions(PLAYER_0)[0], PLAYER_1: env.legal_actions(PLAYER_1)[0]})
        assert len(env.history) == 3
        _record(report, "environment_contract", True)
    except Exception as exc:
        _record(report, "environment_contract", False, repr(exc))

    try:
        state = GameState.initial(3, current_prize=1)
        out = transition(state, 1, 1, next_prize=2)
        assert out.state.carry_pool == 1
        assert out.state.current_prize == 2
        _record(report, "pure_transition_contract", True)
    except Exception as exc:
        _record(report, "pure_transition_contract", False, repr(exc))

    try:
        rpt = estimate_complexity(5)
        assert rpt.chance_states == 2252
        report.metrics["exact_C5"] = rpt.chance_states
        _record(report, "exact_complexity_estimator", True)
    except Exception as exc:
        _record(report, "exact_complexity_estimator", False, repr(exc))

    try:
        import numpy as np

        q = np.array([[1.0, -1.0], [-1.0, 1.0]])
        value, row, col = solve_zero_sum_matrix(q)
        assert abs(value) < 1e-6
        assert abs(row.sum() - 1.0) < 1e-6
        assert abs(col.sum() - 1.0) < 1e-6
        _record(report, "reference_matrix_solver", True)
    except Exception as exc:
        _record(report, "reference_matrix_solver", False, repr(exc))

    try:
        artifact = Path(artifact_dir)
        store = JsonlStore(artifact / "schema_smoke.jsonl")
        sample = GameCorpusSample(
            sample_id="schema-smoke",
            state=state_record_from_game_state(GameState.initial(3, current_prize=1)),
            round_event=None,
        )
        store.append(sample)
        assert store.count() >= 1
        _record(report, "jsonl_schema_store", True)
    except Exception as exc:
        _record(report, "jsonl_schema_store", False, repr(exc))

    try:
        assert Q_PRIORITY[0] == "EXACT"
        assert POLICY_PRIORITY[0] == "EXACT"
        _record(report, "teacher_priority_contract", True)
    except Exception as exc:
        _record(report, "teacher_priority_contract", False, repr(exc))

    report.ok = all(report.checks.values())
    return report
