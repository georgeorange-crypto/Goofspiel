from __future__ import annotations

from goofspiel.training.distillation import fast_student_plan, strong_student_plan
from goofspiel.training.league import LeagueAgent, LeagueRegistry, ROLE_AGGRESSIVE, ROLE_EXPLOITER, ROLE_ROBUST


def test_league_roles_and_distillation_interfaces(tmp_path):
    registry = LeagueRegistry(tmp_path / "league.json")
    for role in (ROLE_ROBUST, ROLE_AGGRESSIVE, ROLE_EXPLOITER):
        registry.add(LeagueAgent(f"a_{role}", role, None, 1, {"priority": 1.0}))
    assert registry.counts_by_role()[ROLE_ROBUST] == 1
    assert registry.sample() is not None
    assert strong_student_plan("teacher.pt").student_kind == "strong"
    assert fast_student_plan("teacher.pt").student_kind == "fast"
