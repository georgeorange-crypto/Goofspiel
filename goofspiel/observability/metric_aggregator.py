"""Low-frequency metric aggregation for training/evaluation logs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSummary:
    name: str
    count: int
    mean: float
    last: float


class MetricAggregator:
    def __init__(self) -> None:
        self._values: dict[str, list[float]] = defaultdict(list)

    def add(self, name: str, value: float) -> None:
        self._values[str(name)].append(float(value))

    def summaries(self) -> list[MetricSummary]:
        out = []
        for name, values in sorted(self._values.items()):
            out.append(MetricSummary(name=name, count=len(values), mean=sum(values) / len(values), last=values[-1]))
        return out
