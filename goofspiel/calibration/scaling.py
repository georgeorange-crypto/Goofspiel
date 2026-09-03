"""Scaling fit + projection labelling for F5 (MEASUREMENT ONLY).

F5 turns a workload sweep into a cost model ``walltime ≈ fixed_overhead +
per_game·games`` and lets F6 *project* the cost of a candidate FULL workload.

The one invariant this module exists to enforce: **a projection can never be
labelled MEASURED.**  There are exactly two ways to obtain a :class:`Projection`:

  * :meth:`Projection.measured` — from a :class:`WorkloadPoint` that was
    directly executed (``measured=True``); kind is ``MEASURED``.
  * :meth:`Projection.from_fit` — from a :class:`LinearFit`; kind is *always*
    ``PROJECTED``, and additionally ``beyond_anchor`` when n_cards ≠ 5.

There is no constructor that takes a fit and yields ``MEASURED``, so F6 cannot
present an extrapolated n=13 runtime as if it had been measured.

Closed-form ordinary least squares over stdlib floats — no numpy, so the tests
re-execute the arithmetic by hand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .records import WorkloadPoint
from . import ANCHOR_N_CARDS


class ScalingError(ValueError):
    """Raised when a fit cannot be computed or a projection is misused."""


class ProjectionKind(str, Enum):
    MEASURED = "MEASURED"
    PROJECTED = "PROJECTED"


@dataclass(frozen=True)
class LinearFit:
    """Least-squares fit of wall-clock against workload (games).

    ``wall ≈ fixed_overhead_s + per_game_s · games``.  ``cv_games_per_sec`` is the
    coefficient of variation of the per-point throughput — a cheap stability
    signal: if throughput is roughly constant across the sweep the linear model
    extrapolates safely; if it swings, extrapolation (esp. beyond the anchor) is
    not trustworthy.
    """

    fixed_overhead_s: float
    per_game_s: float
    r_squared: float
    n_points: int
    cv_games_per_sec: float

    def predict_wall_s(self, games: float) -> float:
        return self.fixed_overhead_s + self.per_game_s * games

    def is_extrapolation_safe(self, *, threshold_cv: float = 0.15) -> bool:
        """Heuristic: stable throughput (low CV) and a decent linear fit.

        Deliberately conservative — this only ever *downgrades* confidence; it
        never turns a projection into a measurement.
        """
        return self.cv_games_per_sec <= threshold_cv and self.r_squared >= 0.90


def games_per_sec_stability(points: Sequence[WorkloadPoint]) -> float:
    """Coefficient of variation (std/mean) of per-point games/sec.

    Returns ``nan`` if fewer than 2 usable points or a non-positive mean.
    """
    rates = [p.games_per_sec for p in points if p.wall_clock_s > 0 and not math.isnan(p.games_per_sec)]
    if len(rates) < 2:
        return float("nan")
    mean = sum(rates) / len(rates)
    if mean <= 0:
        return float("nan")
    var = sum((r - mean) ** 2 for r in rates) / len(rates)
    return math.sqrt(var) / mean


def fit_linear_walltime(points: Sequence[WorkloadPoint]) -> LinearFit:
    """Closed-form OLS of wall_clock_s on games across a workload sweep.

    Requires ≥2 points with distinct ``games`` (a sweep, per the F directive:
    the cost model must come from ≥2–3 points, never a single sample).
    """
    pts = list(points)
    if len(pts) < 2:
        raise ScalingError(
            f"linear fit needs >=2 sweep points (F requires a sweep), got {len(pts)}"
        )
    xs = [p.games for p in pts]
    ys = [p.wall_clock_s for p in pts]
    n = len(pts)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        raise ScalingError(
            "workload sweep has no variation in games; cannot fit a slope "
            "(all points at the same workload)"
        )
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = mean_y - slope * mean_x

    # R^2 = 1 - SS_res / SS_tot  (SS_tot==0 → degenerate y; report 0.0)
    ss_tot = sum((y - mean_y) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return LinearFit(
        fixed_overhead_s=intercept,
        per_game_s=slope,
        r_squared=r_squared,
        n_points=n,
        cv_games_per_sec=games_per_sec_stability(pts),
    )


@dataclass(frozen=True)
class Projection:
    """A cost estimate that carries an honest MEASURED/PROJECTED label.

    Construct only via :meth:`measured` or :meth:`from_fit`; the bare constructor
    is not meant to be called directly (and the two factories are the only
    supported entry points).
    """

    kind: ProjectionKind
    label: str
    n_cards: int
    games: float
    wall_clock_s: float
    beyond_anchor: bool
    extrapolation_safe: bool | None = None

    @property
    def is_measured(self) -> bool:
        return self.kind is ProjectionKind.MEASURED

    @classmethod
    def measured(cls, point: WorkloadPoint) -> "Projection":
        """Wrap a *directly executed* point.  Refuses a non-measured point."""
        if not point.measured:
            raise ScalingError(
                f"Projection.measured requires a directly executed point; "
                f"{point.label!r} has measured=False"
            )
        return cls(
            kind=ProjectionKind.MEASURED,
            label=point.label,
            n_cards=point.n_cards,
            games=point.games,
            wall_clock_s=point.wall_clock_s,
            beyond_anchor=(point.n_cards != ANCHOR_N_CARDS),
            extrapolation_safe=None,
        )

    @classmethod
    def from_fit(
        cls,
        fit: LinearFit,
        *,
        target_games: float,
        n_cards: int,
        label: str | None = None,
    ) -> "Projection":
        """Estimate a workload's cost from a fit — ALWAYS labelled PROJECTED.

        When ``n_cards`` differs from the anchor (5), ``beyond_anchor`` is set so
        F6 can flag it (e.g. n=13 must never be shown as a measured FULL time).
        """
        if target_games < 0:
            raise ScalingError(f"target_games must be >=0, got {target_games}")
        beyond = n_cards != ANCHOR_N_CARDS
        return cls(
            kind=ProjectionKind.PROJECTED,
            label=label or f"projected@{int(target_games)}games",
            n_cards=n_cards,
            games=target_games,
            wall_clock_s=fit.predict_wall_s(target_games),
            beyond_anchor=beyond,
            # A cross-board extrapolation is never considered safe: the fit was
            # built at the anchor board and per-game cost changes with n_cards.
            extrapolation_safe=(fit.is_extrapolation_safe() and not beyond),
        )

    def annotated_label(self) -> str:
        """Report string that always exposes the projection status."""
        tag = self.kind.value
        if self.beyond_anchor:
            tag += f" / BEYOND ANCHOR n={self.n_cards}"
        return f"[{tag}] {self.label}"


__all__ = [
    "ScalingError",
    "ProjectionKind",
    "LinearFit",
    "Projection",
    "fit_linear_walltime",
    "games_per_sec_stability",
]
