"""Replay buffer helpers for self-play trajectories."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from goofspiel.training.data import JsonlStore, RobustTrajectorySample


class TrajectoryReplayBuffer:
    """Small append-only replay buffer backed by JSONL for reproducible smoke runs."""

    def __init__(self, path: str | Path, *, max_size: int = 10_000) -> None:
        self.store = JsonlStore[RobustTrajectorySample](path)
        self.max_size = int(max_size)
        self.items: list[RobustTrajectorySample] = []

    @property
    def path(self) -> Path:
        return self.store.path

    def append_many(self, samples: list[RobustTrajectorySample]) -> None:
        self.items.extend(samples)
        if len(self.items) > self.max_size:
            self.items = self.items[-self.max_size:]
        self.store.extend(samples)

    def extend_in_memory(self, samples: list[RobustTrajectorySample]) -> None:
        """Extend the in-memory buffer without touching disk.

        Used for distributed training where each rank contributes local rollouts
        but only one rank should own the write-on-disk side effect.
        """
        self.items.extend(samples)
        if len(self.items) > self.max_size:
            self.items = self.items[-self.max_size:]

    def sample(self, k: int, rng: random.Random | None = None) -> list[RobustTrajectorySample]:
        if not self.items:
            return []
        rng = rng or random.Random()
        k = min(int(k), len(self.items))
        return rng.sample(self.items, k)

    def count(self) -> int:
        return len(self.items)

    def persisted_count(self) -> int:
        return self.store.count()


def replay_summary(buffer: TrajectoryReplayBuffer) -> dict[str, Any]:
    transitions = sum(len(item.rounds) for item in buffer.items)
    return {
        "path": str(buffer.path),
        "in_memory_samples": buffer.count(),
        "persisted_samples": buffer.persisted_count(),
        "transitions": transitions,
    }
