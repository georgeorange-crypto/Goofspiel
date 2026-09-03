"""Workload + timing accounting records for F2/F3/F4 (MEASUREMENT ONLY).

These dataclasses are the calibration schema: the caller fills raw counts and
wall-clock seconds measured while an Arena runner executed; every *derived*
quantity (games/sec, actions/sec, overhead fraction) is recomputed as a property
so a test can re-execute the arithmetic instead of trusting a stored field.

The field sets are dictated by the F directive:

  * Stage6 (league): matchup kind, seeds, prize sequences, games/block, raw
    games, bootstrap/paired blocks, total actions, wall — split into the
    play portion vs. the paired/block-statistics portion so we can show play is
    the ~95% and the statistics machinery is a small overhead.
  * Stage7 (red-team): cost broken into the seven named components
    (attack generation, canonicalize/dedup/split, before-eval, correction
    optimizer, held-out evaluation, normal-play regression, artifact/report),
    each with case count, actions and wall.
  * evaluate/Arena: timing split by *effective agent family*
    (policy/heuristic/exact/search/fallback), not just an average.

Pure stdlib; no torch, no Arena imports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


class RecordError(ValueError):
    """Raised when a measurement record is internally inconsistent."""


# --------------------------------------------------------------------------- #
# F5 fit input
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WorkloadPoint:
    """One (workload, wall-clock) sample for the scaling fit.

    ``games`` is the workload magnitude (raw games for Stage6, cases for Stage7,
    games for evaluate — whatever the fit is over).  ``measured=True`` means this
    point was *directly executed*; the scaling layer refuses to label a
    non-measured point as MEASURED.
    """

    label: str
    n_cards: int
    games: float
    wall_clock_s: float
    measured: bool = True

    def __post_init__(self) -> None:
        if self.games < 0:
            raise RecordError(f"games must be >=0, got {self.games}")
        if self.wall_clock_s < 0:
            raise RecordError(f"wall_clock_s must be >=0, got {self.wall_clock_s}")

    @property
    def games_per_sec(self) -> float:
        return self.games / self.wall_clock_s if self.wall_clock_s > 0 else float("nan")


# --------------------------------------------------------------------------- #
# F2 — Stage6 league
# --------------------------------------------------------------------------- #
STAGE6_MATCHUP_KINDS = ("ordered", "competitive", "self_play")


@dataclass
class Stage6Measurement:
    """Accounting for one Stage6 league measurement at a fixed workload."""

    matchup_kind: str
    seeds: int
    prize_sequences: int
    games_per_block: int
    raw_games: int
    bootstrap_blocks: int
    paired_blocks: int
    total_actions: int
    wall_clock_s: float
    play_wall_s: float
    stats_wall_s: float

    def __post_init__(self) -> None:
        if self.matchup_kind not in STAGE6_MATCHUP_KINDS:
            raise RecordError(
                f"matchup_kind must be one of {STAGE6_MATCHUP_KINDS}, got {self.matchup_kind!r}"
            )
        if self.raw_games <= 0:
            raise RecordError(f"raw_games must be >0, got {self.raw_games}")
        if self.wall_clock_s <= 0:
            raise RecordError(f"wall_clock_s must be >0, got {self.wall_clock_s}")
        # Component walls cannot exceed the whole; they may sum to less (other
        # small overheads like init are the remainder).
        if self.play_wall_s < 0 or self.stats_wall_s < 0:
            raise RecordError("component walls must be non-negative")
        if self.play_wall_s + self.stats_wall_s > self.wall_clock_s + 1e-9:
            raise RecordError(
                f"play_wall_s + stats_wall_s ({self.play_wall_s + self.stats_wall_s}) "
                f"exceeds wall_clock_s ({self.wall_clock_s})"
            )

    @property
    def games_per_sec(self) -> float:
        return self.raw_games / self.wall_clock_s

    @property
    def actions_per_sec(self) -> float:
        return self.total_actions / self.wall_clock_s

    @property
    def play_fraction(self) -> float:
        """Fraction of wall spent actually playing games (target: ~0.95)."""
        return self.play_wall_s / self.wall_clock_s

    @property
    def stats_overhead_fraction(self) -> float:
        """Fraction of wall spent in paired/block bootstrap statistics."""
        return self.stats_wall_s / self.wall_clock_s

    @property
    def other_overhead_fraction(self) -> float:
        """Remainder (init, checkpoint load, report) not in play or stats."""
        return max(0.0, 1.0 - self.play_fraction - self.stats_overhead_fraction)

    def as_workload_point(self, *, n_cards: int, label: str | None = None, measured: bool = True) -> WorkloadPoint:
        return WorkloadPoint(
            label=label or f"stage6:{self.matchup_kind}",
            n_cards=n_cards,
            games=float(self.raw_games),
            wall_clock_s=self.wall_clock_s,
            measured=measured,
        )


# --------------------------------------------------------------------------- #
# F3 — Stage7 red-team component breakdown
# --------------------------------------------------------------------------- #
STAGE7_COMPONENTS = (
    "attack_generation",
    "canonicalize_dedup_split",
    "before_eval",
    "correction_optimizer",
    "heldout_evaluation",
    "normal_play_regression",
    "artifact_report",
)


@dataclass
class Stage7ComponentTiming:
    """Cost of one named Stage7 phase."""

    component: str
    case_count: int
    actions: int
    wall_s: float

    def __post_init__(self) -> None:
        if self.component not in STAGE7_COMPONENTS:
            raise RecordError(
                f"component must be one of {STAGE7_COMPONENTS}, got {self.component!r}"
            )
        if self.wall_s < 0:
            raise RecordError(f"wall_s must be >=0, got {self.wall_s}")

    @property
    def actions_per_sec(self) -> float:
        return self.actions / self.wall_s if self.wall_s > 0 else float("nan")


@dataclass
class Stage7Breakdown:
    """The full Stage7 red-team broken into its component timings."""

    components: list[Stage7ComponentTiming] = field(default_factory=list)

    def __post_init__(self) -> None:
        seen = [c.component for c in self.components]
        dupes = {c for c in seen if seen.count(c) > 1}
        if dupes:
            raise RecordError(f"duplicate Stage7 components: {sorted(dupes)}")

    @property
    def total_wall_s(self) -> float:
        return sum(c.wall_s for c in self.components)

    @property
    def total_actions(self) -> int:
        return sum(c.actions for c in self.components)

    def fraction_by_component(self) -> dict[str, float]:
        total = self.total_wall_s
        if total <= 0:
            return {c.component: float("nan") for c in self.components}
        return {c.component: c.wall_s / total for c in self.components}

    def missing_components(self) -> tuple[str, ...]:
        """Named components the F directive expects but that weren't measured."""
        present = {c.component for c in self.components}
        return tuple(c for c in STAGE7_COMPONENTS if c not in present)


# --------------------------------------------------------------------------- #
# F4 — evaluate/Arena family split
# --------------------------------------------------------------------------- #
EVAL_FAMILIES = ("policy", "heuristic", "exact", "search", "fallback")


@dataclass
class EvaluateFamilyTiming:
    """evaluate timing for one *effective* agent family."""

    family: str
    games: int
    decisions: int
    wall_s: float

    def __post_init__(self) -> None:
        if self.family not in EVAL_FAMILIES:
            raise RecordError(f"family must be one of {EVAL_FAMILIES}, got {self.family!r}")
        if self.wall_s < 0:
            raise RecordError(f"wall_s must be >=0, got {self.wall_s}")

    @property
    def decisions_per_sec(self) -> float:
        return self.decisions / self.wall_s if self.wall_s > 0 else float("nan")

    @property
    def sec_per_decision(self) -> float:
        return self.wall_s / self.decisions if self.decisions > 0 else float("nan")


@dataclass
class EvaluateBreakdown:
    """evaluate cost split by effective family — never a single average."""

    families: list[EvaluateFamilyTiming] = field(default_factory=list)

    def __post_init__(self) -> None:
        seen = [f.family for f in self.families]
        dupes = {f for f in seen if seen.count(f) > 1}
        if dupes:
            raise RecordError(f"duplicate evaluate families: {sorted(dupes)}")

    @property
    def total_wall_s(self) -> float:
        return sum(f.wall_s for f in self.families)

    @property
    def total_decisions(self) -> int:
        return sum(f.decisions for f in self.families)

    def sec_per_decision_by_family(self) -> dict[str, float]:
        """Per-family unit cost — the whole point of the family split."""
        return {f.family: f.sec_per_decision for f in self.families}


def total_actions(records: Iterable[Stage6Measurement]) -> int:
    """Convenience aggregate used by workload-accounting reports."""
    return sum(r.total_actions for r in records)


__all__ = [
    "RecordError",
    "WorkloadPoint",
    "Stage6Measurement",
    "STAGE6_MATCHUP_KINDS",
    "Stage7ComponentTiming",
    "Stage7Breakdown",
    "STAGE7_COMPONENTS",
    "EvaluateFamilyTiming",
    "EvaluateBreakdown",
    "EVAL_FAMILIES",
    "total_actions",
]
