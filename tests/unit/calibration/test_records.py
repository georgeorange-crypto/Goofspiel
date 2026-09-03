"""Re-execution tests for the F2/F3/F4 accounting records.

Every derived quantity is recomputed here from the raw inputs and compared to the
record's property — the property must never be trusted as a stored value.
"""

from __future__ import annotations

import math

import pytest

from goofspiel.calibration.records import (
    EVAL_FAMILIES,
    STAGE6_MATCHUP_KINDS,
    STAGE7_COMPONENTS,
    EvaluateBreakdown,
    EvaluateFamilyTiming,
    RecordError,
    Stage6Measurement,
    Stage7Breakdown,
    Stage7ComponentTiming,
    WorkloadPoint,
    total_actions,
)


# -- Stage6 ------------------------------------------------------------------ #
def _stage6(**over):
    base = dict(
        matchup_kind="competitive",
        seeds=4,
        prize_sequences=8,
        games_per_block=100,
        raw_games=4000,
        bootstrap_blocks=40,
        paired_blocks=40,
        total_actions=20000,
        wall_clock_s=50.0,
        play_wall_s=47.5,
        stats_wall_s=2.0,
    )
    base.update(over)
    return Stage6Measurement(**base)


def test_stage6_rates_recomputed():
    m = _stage6()
    assert m.games_per_sec == pytest.approx(4000 / 50.0)
    assert m.actions_per_sec == pytest.approx(20000 / 50.0)


def test_stage6_play_fraction_is_about_95_percent():
    m = _stage6()
    # Recompute by hand: 47.5 / 50 = 0.95
    assert m.play_fraction == pytest.approx(47.5 / 50.0)
    assert m.stats_overhead_fraction == pytest.approx(2.0 / 50.0)
    # Remainder is init/report; recompute.
    assert m.other_overhead_fraction == pytest.approx(1.0 - 0.95 - 0.04)


def test_stage6_fractions_sum_to_one():
    m = _stage6()
    total = m.play_fraction + m.stats_overhead_fraction + m.other_overhead_fraction
    assert total == pytest.approx(1.0)


def test_stage6_rejects_component_walls_exceeding_total():
    with pytest.raises(RecordError):
        _stage6(play_wall_s=49.0, stats_wall_s=5.0)  # 54 > 50


def test_stage6_rejects_bad_matchup_kind():
    with pytest.raises(RecordError):
        _stage6(matchup_kind="nonsense")


def test_stage6_matchup_kinds_frozen():
    assert STAGE6_MATCHUP_KINDS == ("ordered", "competitive", "self_play")


def test_stage6_as_workload_point_carries_raw_games():
    m = _stage6()
    wp = m.as_workload_point(n_cards=5)
    assert wp.games == 4000.0
    assert wp.wall_clock_s == 50.0
    assert wp.n_cards == 5
    assert wp.measured is True


def test_total_actions_aggregate():
    ms = [_stage6(total_actions=100), _stage6(total_actions=250)]
    assert total_actions(ms) == 350


# -- Stage7 ------------------------------------------------------------------ #
def test_stage7_component_names_are_the_seven():
    assert STAGE7_COMPONENTS == (
        "attack_generation",
        "canonicalize_dedup_split",
        "before_eval",
        "correction_optimizer",
        "heldout_evaluation",
        "normal_play_regression",
        "artifact_report",
    )
    assert len(STAGE7_COMPONENTS) == 7


def test_stage7_breakdown_totals_and_fractions_recomputed():
    comps = [
        Stage7ComponentTiming(component="attack_generation", case_count=10, actions=100, wall_s=4.0),
        Stage7ComponentTiming(component="correction_optimizer", case_count=10, actions=200, wall_s=16.0),
    ]
    bd = Stage7Breakdown(components=comps)
    assert bd.total_wall_s == pytest.approx(20.0)
    assert bd.total_actions == 300
    fracs = bd.fraction_by_component()
    assert fracs["attack_generation"] == pytest.approx(4.0 / 20.0)
    assert fracs["correction_optimizer"] == pytest.approx(16.0 / 20.0)
    assert sum(fracs.values()) == pytest.approx(1.0)


def test_stage7_missing_components_reported():
    bd = Stage7Breakdown(components=[
        Stage7ComponentTiming(component="attack_generation", case_count=1, actions=1, wall_s=1.0),
    ])
    missing = bd.missing_components()
    assert "correction_optimizer" in missing
    assert "attack_generation" not in missing
    assert len(missing) == 6


def test_stage7_rejects_duplicate_component():
    with pytest.raises(RecordError):
        Stage7Breakdown(components=[
            Stage7ComponentTiming(component="before_eval", case_count=1, actions=1, wall_s=1.0),
            Stage7ComponentTiming(component="before_eval", case_count=1, actions=1, wall_s=1.0),
        ])


def test_stage7_rejects_unknown_component():
    with pytest.raises(RecordError):
        Stage7ComponentTiming(component="made_up", case_count=1, actions=1, wall_s=1.0)


def test_stage7_component_actions_per_sec():
    c = Stage7ComponentTiming(component="heldout_evaluation", case_count=5, actions=500, wall_s=10.0)
    assert c.actions_per_sec == pytest.approx(50.0)


# -- evaluate ---------------------------------------------------------------- #
def test_eval_families_are_the_five():
    assert EVAL_FAMILIES == ("policy", "heuristic", "exact", "search", "fallback")


def test_eval_per_family_unit_cost_recomputed():
    fams = [
        EvaluateFamilyTiming(family="policy", games=100, decisions=1000, wall_s=5.0),
        EvaluateFamilyTiming(family="exact", games=100, decisions=1000, wall_s=50.0),
    ]
    bd = EvaluateBreakdown(families=fams)
    costs = bd.sec_per_decision_by_family()
    # exact is 10x slower per decision — recompute both.
    assert costs["policy"] == pytest.approx(5.0 / 1000)
    assert costs["exact"] == pytest.approx(50.0 / 1000)
    assert bd.total_wall_s == pytest.approx(55.0)
    assert bd.total_decisions == 2000


def test_eval_rejects_duplicate_family():
    with pytest.raises(RecordError):
        EvaluateBreakdown(families=[
            EvaluateFamilyTiming(family="policy", games=1, decisions=1, wall_s=1.0),
            EvaluateFamilyTiming(family="policy", games=1, decisions=1, wall_s=1.0),
        ])


def test_eval_rejects_unknown_family():
    with pytest.raises(RecordError):
        EvaluateFamilyTiming(family="mcts", games=1, decisions=1, wall_s=1.0)


# -- WorkloadPoint ----------------------------------------------------------- #
def test_workload_point_games_per_sec():
    wp = WorkloadPoint(label="x", n_cards=5, games=1000, wall_clock_s=25.0)
    assert wp.games_per_sec == pytest.approx(40.0)


def test_workload_point_zero_wall_is_nan_not_crash():
    wp = WorkloadPoint(label="x", n_cards=5, games=1000, wall_clock_s=0.0)
    assert math.isnan(wp.games_per_sec)


def test_workload_point_rejects_negative():
    with pytest.raises(RecordError):
        WorkloadPoint(label="x", n_cards=5, games=-1, wall_clock_s=1.0)
