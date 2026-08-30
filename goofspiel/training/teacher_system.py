"""Teacher ensemble, filtering, and EMA teacher interfaces."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from goofspiel.game import GameState
from goofspiel.training.data import TeacherSample
from goofspiel.training.teachers import TeacherRouter


@dataclass(frozen=True)
class TeacherFilterConfig:
    min_confidence: float = 0.75
    max_disagreement: float = 0.25


class EMATeacher:
    def __init__(self, model=None, tau: float = 0.005) -> None:
        self.model = copy.deepcopy(model) if model is not None else None
        self.tau = float(tau)

    def update(self, model) -> None:
        if self.model is None:
            self.model = copy.deepcopy(model)
            return
        src = dict(model.named_parameters())
        for name, param in self.model.named_parameters():
            param.data.mul_(1.0 - self.tau).add_(src[name].data, alpha=self.tau)


class TeacherEnsemble:
    def __init__(self, *, router: TeacherRouter | None = None, config: TeacherFilterConfig | None = None) -> None:
        self.router = router or TeacherRouter()
        self.config = config or TeacherFilterConfig()

    def label(self, state: GameState) -> TeacherSample | None:
        sample = self.router.label_state(state)
        if sample.teacher_confidence < self.config.min_confidence:
            return None
        if sample.teacher_disagreement > self.config.max_disagreement:
            return None
        return sample
