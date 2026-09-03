"""Guard tests for the F1–F4 drivers — the §4 preflight, exercised without Arena.

The preflight is the safety gate that keeps F from measuring off an unverified
tree.  These tests build a real temp git repo and inject a fake ``runner_probe``
so the guard's three refusals (SHA mismatch / dirty tree / missing Arena runner)
are re-executed against real git, with no Arena checkout required.

The F2/F3/F4 drivers themselves are exercised with injected ``run_fn`` stubs — we
verify the driver logic (sweep-size guard, record pass-through) without invoking
any Arena runner. None of the real Arena calls run here.
"""

from __future__ import annotations

import subprocess

import pytest

from goofspiel.calibration.drivers import (
    CalibrationPreflightError,
    Preflight,
    preflight,
    run_evaluate_family_timing,
    run_stage6_sweep,
    run_stage7_component_timing,
    stage6_to_points,
    verify_exact_sha,
)
from goofspiel.calibration.provenance import git_commit
from goofspiel.calibration.records import (
    EvaluateBreakdown,
    EvaluateFamilyTiming,
    Stage6Measurement,
    Stage7Breakdown,
    Stage7ComponentTiming,
)


def _run(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


@pytest.fixture
def temp_repo(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "init")
    _run(repo, "config", "user.email", "t@t")
    _run(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("hello", encoding="utf-8")
    _run(repo, "add", "a.txt")
    _run(repo, "commit", "-m", "init")
    return repo, git_commit(repo)


def _ok_probe():
    return None  # Arena importable


def _missing_probe():
    raise ImportError("No module named 'goofspiel.arena'")


def _kwargs():
    return dict(device="cpu", world_size=1, seed=1, n_cards=5)


def test_preflight_passes_clean_matching_sha(temp_repo):
    repo, head = temp_repo
    pf = preflight(repo, head, runner_probe=_ok_probe, **_kwargs())
    assert isinstance(pf, Preflight)
    assert pf.expected_sha == head
    assert pf.provenance.git_commit == head
    assert pf.provenance.dirty is False


def test_preflight_refuses_sha_mismatch(temp_repo):
    repo, _ = temp_repo
    with pytest.raises(CalibrationPreflightError, match="expected Arena integration SHA"):
        preflight(repo, "b" * 40, runner_probe=_ok_probe, **_kwargs())


def test_preflight_refuses_dirty_tree(temp_repo):
    repo, head = temp_repo
    (repo / "a.txt").write_text("modified", encoding="utf-8")  # dirty tracked file
    with pytest.raises(CalibrationPreflightError, match="dirty"):
        preflight(repo, head, runner_probe=_ok_probe, **_kwargs())


def test_preflight_allows_dirty_only_when_explicit(temp_repo):
    repo, head = temp_repo
    (repo / "a.txt").write_text("modified", encoding="utf-8")
    pf = preflight(repo, head, runner_probe=_ok_probe, allow_dirty=True, **_kwargs())
    # Allowed, but the record is tagged DEV / NON-BINDING.
    assert pf.provenance.dirty is True
    assert "NOT FOR PROMOTION" in pf.provenance.disposition()


def test_preflight_refuses_missing_arena_runner(temp_repo):
    repo, head = temp_repo
    with pytest.raises(CalibrationPreflightError, match="not importable"):
        preflight(repo, head, runner_probe=_missing_probe, **_kwargs())


def test_verify_exact_sha_ok_and_fail(temp_repo):
    repo, head = temp_repo
    verify_exact_sha(repo, head)  # no raise
    with pytest.raises(CalibrationPreflightError):
        verify_exact_sha(repo, "c" * 40)


# -- driver logic with injected stubs (no Arena) ----------------------------- #
def _preflight_stub(temp_repo):
    repo, head = temp_repo
    return preflight(repo, head, runner_probe=_ok_probe, **_kwargs())


def test_stage6_sweep_requires_two_points(temp_repo):
    pf = _preflight_stub(temp_repo)

    def run_fn(spec):  # pragma: no cover - should not be called
        raise AssertionError("run_fn must not be called when the sweep guard trips")

    with pytest.raises(CalibrationPreflightError, match="sweep"):
        run_stage6_sweep(pf, [{"games": 100}], run_fn)


def test_stage6_sweep_runs_every_point_no_early_stop(temp_repo):
    pf = _preflight_stub(temp_repo)
    calls = []

    def run_fn(spec):
        calls.append(spec["raw_games"])
        return Stage6Measurement(
            matchup_kind="competitive", seeds=1, prize_sequences=1, games_per_block=10,
            raw_games=spec["raw_games"], bootstrap_blocks=1, paired_blocks=1,
            total_actions=spec["raw_games"] * 5, wall_clock_s=spec["raw_games"] * 0.01,
            play_wall_s=spec["raw_games"] * 0.009, stats_wall_s=spec["raw_games"] * 0.0005,
        )

    specs = [{"raw_games": 1000}, {"raw_games": 2000}, {"raw_games": 3000}]
    out = run_stage6_sweep(pf, specs, run_fn)
    # No sequential stopping: all three executed.
    assert calls == [1000, 2000, 3000]
    assert len(out) == 3
    pts = stage6_to_points(out, n_cards=5)
    assert [p.games for p in pts] == [1000.0, 2000.0, 3000.0]


def test_stage7_and_evaluate_drivers_pass_through(temp_repo):
    pf = _preflight_stub(temp_repo)

    def s7():
        return Stage7Breakdown(components=[
            Stage7ComponentTiming(component="attack_generation", case_count=1, actions=1, wall_s=1.0),
        ])

    def ev():
        return EvaluateBreakdown(families=[
            EvaluateFamilyTiming(family="policy", games=1, decisions=10, wall_s=1.0),
        ])

    assert run_stage7_component_timing(pf, s7).total_wall_s == pytest.approx(1.0)
    assert run_evaluate_family_timing(pf, ev).total_decisions == 10
