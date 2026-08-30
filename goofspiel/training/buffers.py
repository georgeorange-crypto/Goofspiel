"""Replay buffers for long Goofspiel training."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from goofspiel.training.data import JsonlStore, RobustTrajectorySample


@dataclass
class ReplayItem:
    item_id: str
    state: dict[str, Any]
    q_target: list[list[float]]
    policy_target: list[float]
    final_score_diff: float
    priority: float = 1.0
    source: str = "selfplay"


class PrioritizedReplayBuffer:
    def __init__(self, path: str | Path, *, capacity: int = 200_000) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.capacity = int(capacity)
        self.items: list[ReplayItem] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.items.append(ReplayItem(**json.loads(line)))
            self.items = self.items[-self.capacity :]

    def add(self, item: ReplayItem) -> None:
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity :]

    def extend(self, items: list[ReplayItem]) -> None:
        for item in items:
            self.add(item)

    def sample(self, batch_size: int, rng: random.Random | None = None) -> list[ReplayItem]:
        if not self.items:
            return []
        rng = rng or random.Random()
        weights = [max(1e-6, item.priority) for item in self.items]
        return rng.choices(self.items, weights=weights, k=int(batch_size))

    def save(self) -> None:
        with self.path.open("w", encoding="utf-8") as handle:
            for item in self.items[-self.capacity :]:
                handle.write(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True))
                handle.write("\n")

    def __len__(self) -> int:
        return len(self.items)


class RobustTrajectoryBufferWriter:
    def __init__(self, path: str | Path) -> None:
        self.store = JsonlStore[RobustTrajectorySample](path)

    def add(self, trajectory: RobustTrajectorySample) -> None:
        self.store.append(trajectory)

    def count(self) -> int:
        return self.store.count()
