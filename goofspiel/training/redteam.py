"""Red-team failure capture and correction dataset helpers."""

from __future__ import annotations

from pathlib import Path

from goofspiel.training.data import FailureRecord, JsonlStore, ReanalysisRecord


class FailureBuffer:
    def __init__(self, path: str | Path) -> None:
        self.store = JsonlStore(path)

    def add(self, failure: FailureRecord) -> None:
        self.store.append(failure)

    def count(self) -> int:
        return self.store.count()


class CorrectionDataset:
    def __init__(self, path: str | Path) -> None:
        self.store = JsonlStore(path)

    def add_reanalysis(self, record: ReanalysisRecord) -> None:
        self.store.append(record)

    def count(self) -> int:
        return self.store.count()
