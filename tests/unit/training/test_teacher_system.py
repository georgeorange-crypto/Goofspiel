from __future__ import annotations

from goofspiel.game import GameState
from goofspiel.training.teacher_system import TeacherEnsemble


def test_teacher_ensemble_filters_and_keeps_exact_anchor():
    sample = TeacherEnsemble().label(GameState.initial(3, current_prize=1))
    assert sample is not None
    assert sample.teacher_source == "EXACT"
    assert sample.teacher_confidence >= 0.75
