"""Re-execution tests for the F5 scaling fit and projection labelling.

The fit is checked against a hand-computed closed-form OLS on an exactly-linear
dataset (slope/intercept recovered exactly), and the labelling invariant is
asserted directly: a fit-derived projection is ALWAYS PROJECTED and can never be
MEASURED.
"""

from __future__ import annotations

import math

import pytest

from goofspiel.calibration.records import WorkloadPoint
from goofspiel.calibration.scaling import (
    LinearFit,
    Projection,
    ProjectionKind,
    ScalingError,
    fit_linear_walltime,
    games_per_sec_stability,
)


def _pts(pairs, *, n_cards=5, measured=True):
    return [
        WorkloadPoint(label=f"p{i}", n_cards=n_cards, games=g, wall_clock_s=w, measured=measured)
        for i, (g, w) in enumerate(pairs)
    ]


def test_fit_recovers_exact_line():
    # wall = 10 + 0.5 * games, exactly.
    pts = _pts([(100, 60.0), (200, 110.0), (300, 160.0), (400, 210.0)])
    fit = fit_linear_walltime(pts)
    assert fit.fixed_overhead_s == pytest.approx(10.0)
    assert fit.per_game_s == pytest.approx(0.5)
    assert fit.r_squared == pytest.approx(1.0)
    assert fit.n_points == 4


def test_fit_predicts_by_recomputed_formula():
    pts = _pts([(100, 60.0), (300, 160.0)])
    fit = fit_linear_walltime(pts)
    # Recompute the prediction independently.
    expected = fit.fixed_overhead_s + fit.per_game_s * 1000
    assert fit.predict_wall_s(1000) == pytest.approx(expected)
    assert fit.predict_wall_s(1000) == pytest.approx(10.0 + 0.5 * 1000)


def test_fit_requires_a_sweep():
    with pytest.raises(ScalingError):
        fit_linear_walltime(_pts([(100, 60.0)]))


def test_fit_rejects_zero_variation_in_games():
    with pytest.raises(ScalingError):
        fit_linear_walltime(_pts([(100, 60.0), (100, 61.0), (100, 59.0)]))


def test_stability_cv_zero_for_constant_throughput():
    # All points at 40 games/sec → CV 0.
    pts = _pts([(400, 10.0), (800, 20.0), (1200, 30.0)])
    cv = games_per_sec_stability(pts)
    assert cv == pytest.approx(0.0, abs=1e-12)


def test_stability_cv_positive_for_varying_throughput():
    pts = _pts([(400, 10.0), (800, 40.0)])  # 40 vs 20 games/sec
    cv = games_per_sec_stability(pts)
    assert cv > 0.0


def test_extrapolation_safe_gate():
    stable = LinearFit(fixed_overhead_s=10, per_game_s=0.5, r_squared=0.99, n_points=4, cv_games_per_sec=0.02)
    assert stable.is_extrapolation_safe() is True
    wobbly = LinearFit(fixed_overhead_s=10, per_game_s=0.5, r_squared=0.99, n_points=4, cv_games_per_sec=0.40)
    assert wobbly.is_extrapolation_safe() is False
    poor_fit = LinearFit(fixed_overhead_s=10, per_game_s=0.5, r_squared=0.5, n_points=4, cv_games_per_sec=0.02)
    assert poor_fit.is_extrapolation_safe() is False


# -- the labelling invariant: THE reason this module exists ------------------ #
def test_measured_projection_from_executed_point():
    p = WorkloadPoint(label="anchor", n_cards=5, games=1000, wall_clock_s=25.0, measured=True)
    proj = Projection.measured(p)
    assert proj.kind is ProjectionKind.MEASURED
    assert proj.is_measured is True
    assert proj.beyond_anchor is False
    assert proj.wall_clock_s == 25.0


def test_measured_refuses_non_executed_point():
    p = WorkloadPoint(label="notrun", n_cards=5, games=1000, wall_clock_s=25.0, measured=False)
    with pytest.raises(ScalingError):
        Projection.measured(p)


def test_from_fit_is_always_projected_never_measured():
    fit = fit_linear_walltime(_pts([(100, 60.0), (300, 160.0)]))
    proj = Projection.from_fit(fit, target_games=5000, n_cards=5)
    assert proj.kind is ProjectionKind.PROJECTED
    assert proj.is_measured is False
    # Value equals the recomputed line.
    assert proj.wall_clock_s == pytest.approx(10.0 + 0.5 * 5000)


def test_from_fit_flags_beyond_anchor_for_n13():
    fit = fit_linear_walltime(_pts([(100, 60.0), (300, 160.0)]))
    proj = Projection.from_fit(fit, target_games=5000, n_cards=13)
    assert proj.kind is ProjectionKind.PROJECTED
    assert proj.beyond_anchor is True
    # Cross-board extrapolation is never marked safe.
    assert proj.extrapolation_safe is False
    assert "BEYOND ANCHOR n=13" in proj.annotated_label()
    assert "PROJECTED" in proj.annotated_label()


def test_from_fit_at_anchor_can_be_safe():
    stable = LinearFit(fixed_overhead_s=10, per_game_s=0.5, r_squared=0.99, n_points=4, cv_games_per_sec=0.02)
    proj = Projection.from_fit(stable, target_games=5000, n_cards=5)
    assert proj.beyond_anchor is False
    assert proj.extrapolation_safe is True


def test_annotated_label_measured_has_no_beyond_tag_at_anchor():
    p = WorkloadPoint(label="anchor", n_cards=5, games=1000, wall_clock_s=25.0, measured=True)
    proj = Projection.measured(p)
    label = proj.annotated_label()
    assert "MEASURED" in label
    assert "BEYOND" not in label
