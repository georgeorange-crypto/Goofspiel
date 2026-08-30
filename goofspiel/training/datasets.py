"""Separate dataset stores for the frozen training data pools."""

from __future__ import annotations

from pathlib import Path
from typing import Generic, TypeVar

from goofspiel.training.data import (
    AdaptiveTrajectorySample,
    ExactSample,
    GameCorpusSample,
    JsonlStore,
    OpponentSession,
    ReanalysisRecord,
    RobustTrajectorySample,
    TeacherSample,
)

T = TypeVar("T")


class DatasetPool(Generic[T]):
    def __init__(self, path: str | Path) -> None:
        self.store = JsonlStore[T](path)

    @property
    def path(self) -> Path:
        return self.store.path

    def add(self, item: T) -> None:
        self.store.append(item)

    def count(self) -> int:
        return self.store.count()


GameCorpus = DatasetPool[GameCorpusSample]
ExactDataset = DatasetPool[ExactSample]
TeacherDataset = DatasetPool[TeacherSample]
RobustTrajectoryBuffer = DatasetPool[RobustTrajectorySample]
OpponentSessionBuffer = DatasetPool[OpponentSession]
AdaptiveTrajectoryBuffer = DatasetPool[AdaptiveTrajectorySample]
ReanalysisBuffer = DatasetPool[ReanalysisRecord]
