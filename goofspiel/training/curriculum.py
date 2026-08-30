"""Curriculum scheduling for staged Goofspiel training."""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class CurriculumStep:
    step: int
    n_cards: int
    horizon: int
    reason: str


class ProgressiveCurriculum:
    """Conservative N/horizon schedule used by local and server training."""

    def __init__(self, *, target_n: int, warmup_n: int = 3, ramp_every: int = 100) -> None:
        self.target_n = int(target_n)
        self.warmup_n = max(1, min(int(warmup_n), self.target_n))
        self.ramp_every = max(1, int(ramp_every))

    def at(self, step: int) -> CurriculumStep:
        step = int(step)
        n_cards = min(self.target_n, self.warmup_n + step // self.ramp_every)
        return CurriculumStep(
            step=step,
            n_cards=n_cards,
            horizon=n_cards,
            reason="progressive_n_card_ramp" if n_cards < self.target_n else "target_n_reached",
        )

    def manifest(self, *, steps: int) -> dict:
        probes = sorted(set([0, max(0, steps // 2), max(0, steps - 1)]))
        return {
            "target_n": self.target_n,
            "warmup_n": self.warmup_n,
            "ramp_every": self.ramp_every,
            "probes": [asdict(self.at(step)) for step in probes],
        }
