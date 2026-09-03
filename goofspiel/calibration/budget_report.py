"""Budget report for F6 (MEASUREMENT ONLY — emits a report, never budgets.py).

F6 consumes the per-component unit costs (Stage6 games/sec, Stage7 component
timings, evaluate family costs) and a scaling fit, then proposes three candidate
FULL workloads with their *projected* wall-clock:

  * ``FULL_S``  — dev / smoke-scale sanity (fast, cheap)
  * ``FULL_M``  — nightly-scale
  * ``FULL_L``  — paper-scale

The output is a Markdown + JSON artifact.  This module NEVER writes ``budgets.py``
and never mutates any schema: the candidates are a recommendation for a human to
review, and every non-executed number is carried as a :class:`Projection` so it
prints with a ``PROJECTED`` / ``BEYOND ANCHOR`` tag.  A ``NOT FOR PROMOTION``
banner is emitted unconditionally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .provenance import RunProvenance
from .scaling import LinearFit, Projection, ProjectionKind

# The report filenames are fixed and deliberately NOT "budgets.*": a reviewer
# must copy numbers across by hand, so F can never silently rewrite the schema.
REPORT_BASENAME = "F_calibration_report"
_FORBIDDEN_OUTPUT_NAMES = {"budgets.py", "budgets.json", "budget.py"}

NOT_FOR_PROMOTION_BANNER = (
    "NOT FOR PROMOTION — budget_calibration, non-binding. "
    "Numbers tagged PROJECTED were not directly executed; "
    "n≠5 is BEYOND the measured anchor (formal1k trained at n=5)."
)


@dataclass(frozen=True)
class BudgetLine:
    """One component's measured unit cost, for the report's unit-cost table."""

    component: str
    unit: str  # e.g. "games", "cases", "decisions"
    unit_cost_s: float  # seconds per unit (measured)
    measured: bool = True

    def render_row(self) -> str:
        tag = "measured" if self.measured else "PROJECTED"
        return f"| {self.component} | {self.unit} | {self.unit_cost_s:.6g} s/{self.unit} | {tag} |"


@dataclass(frozen=True)
class FullCandidate:
    """A candidate FULL workload and its projected cost.

    ``tier`` is one of FULL_S/FULL_M/FULL_L.  ``projection`` is the honest label
    carrier — if it says PROJECTED (which cross-board candidates always do), the
    rendered time is shown as an estimate, never a measured runtime.
    """

    tier: str
    purpose: str  # "dev" / "nightly" / "paper"
    workload_games: float
    projection: Projection

    def render_row(self) -> str:
        proj = self.projection
        wall = proj.wall_clock_s
        # Show hours for anything beyond a few minutes; keep raw seconds too.
        human = f"{wall:.1f}s"
        if wall >= 3600:
            human = f"{wall / 3600:.2f}h"
        elif wall >= 60:
            human = f"{wall / 60:.1f}m"
        tag = proj.kind.value
        if proj.beyond_anchor:
            tag += f" / BEYOND ANCHOR n={proj.n_cards}"
        return (
            f"| {self.tier} | {self.purpose} | {int(self.workload_games)} games "
            f"| {human} | {tag} |"
        )


@dataclass
class BudgetReport:
    """The full F6 artifact: unit costs + FULL candidates + provenance."""

    provenance: RunProvenance
    unit_costs: list[BudgetLine] = field(default_factory=list)
    candidates: list[FullCandidate] = field(default_factory=list)
    fit: LinearFit | None = None
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        tiers = [c.tier for c in self.candidates]
        dupes = {t for t in tiers if tiers.count(t) > 1}
        if dupes:
            raise ValueError(f"duplicate FULL candidate tiers: {sorted(dupes)}")

    # -- rendering --------------------------------------------------------- #
    def render_markdown(self) -> str:
        p = self.provenance
        lines: list[str] = []
        lines.append(f"# Workstream F — Arena Budget Calibration Report")
        lines.append("")
        lines.append(f"> **{NOT_FOR_PROMOTION_BANNER}**")
        lines.append("")
        lines.append("## Provenance")
        lines.append("")
        lines.append(f"- commit: `{p.git_commit}`")
        lines.append(f"- branch: `{p.git_branch}`")
        lines.append(f"- dirty: `{p.dirty}` → disposition: **{p.disposition()}**")
        lines.append(f"- device: `{p.device}`  world_size: `{p.world_size}`")
        lines.append(f"- n_cards: `{p.n_cards}` (anchor n=5: {p.is_at_anchor})")
        lines.append(f"- evaluation_purpose: `{p.evaluation_purpose}`  "
                     f"binding_promotion: `{p.binding_promotion}`  "
                     f"profile: `{p.profile_name}`")
        if p.checkpoint_path:
            lines.append(f"- checkpoint: `{p.checkpoint_path}`  sha256: `{p.checkpoint_sha256}`")
        lines.append("")

        lines.append("## Measured unit costs")
        lines.append("")
        lines.append("| Component | Unit | Unit cost | Source |")
        lines.append("| --- | --- | --- | --- |")
        for bl in self.unit_costs:
            lines.append(bl.render_row())
        lines.append("")

        if self.fit is not None:
            f = self.fit
            lines.append("## Scaling fit  (wall ≈ overhead + per_game·games)")
            lines.append("")
            lines.append(f"- fixed_overhead_s: `{f.fixed_overhead_s:.6g}`")
            lines.append(f"- per_game_s: `{f.per_game_s:.6g}`")
            lines.append(f"- R²: `{f.r_squared:.4f}`  points: `{f.n_points}`  "
                         f"CV(games/s): `{f.cv_games_per_sec:.4f}`")
            lines.append(f"- extrapolation_safe(anchor): `{f.is_extrapolation_safe()}`")
            lines.append("")

        lines.append("## Candidate FULL budgets  (projected — for human review)")
        lines.append("")
        lines.append("| Tier | Purpose | Workload | Projected time | Label |")
        lines.append("| --- | --- | --- | --- | --- |")
        for c in self.candidates:
            lines.append(c.render_row())
        lines.append("")

        if self.notes:
            lines.append("## Notes")
            lines.append("")
            for note in self.notes:
                lines.append(f"- {note}")
            lines.append("")

        lines.append("---")
        lines.append("")
        lines.append(
            "_This report proposes budgets for a human to copy into the schema "
            "deliberately. F does not write `budgets.py`, does not early-stop, "
            "and does not promote._"
        )
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        return {
            "not_for_promotion": True,
            "banner": NOT_FOR_PROMOTION_BANNER,
            "provenance": self.provenance.to_dict(),
            "unit_costs": [
                {
                    "component": bl.component,
                    "unit": bl.unit,
                    "unit_cost_s": bl.unit_cost_s,
                    "measured": bl.measured,
                }
                for bl in self.unit_costs
            ],
            "fit": (
                None
                if self.fit is None
                else {
                    "fixed_overhead_s": self.fit.fixed_overhead_s,
                    "per_game_s": self.fit.per_game_s,
                    "r_squared": self.fit.r_squared,
                    "n_points": self.fit.n_points,
                    "cv_games_per_sec": self.fit.cv_games_per_sec,
                }
            ),
            "candidates": [
                {
                    "tier": c.tier,
                    "purpose": c.purpose,
                    "workload_games": c.workload_games,
                    "projected_wall_clock_s": c.projection.wall_clock_s,
                    "kind": c.projection.kind.value,
                    "beyond_anchor": c.projection.beyond_anchor,
                    "n_cards": c.projection.n_cards,
                    "extrapolation_safe": c.projection.extrapolation_safe,
                }
                for c in self.candidates
            ],
            "notes": list(self.notes),
        }

    def save_report(self, out_dir: str | Path) -> dict[str, Path]:
        """Write ``F_calibration_report.md`` and ``.json`` under ``out_dir``.

        Refuses to write anything named like the real budget schema.
        """
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        md_path = out / f"{REPORT_BASENAME}.md"
        json_path = out / f"{REPORT_BASENAME}.json"
        for pth in (md_path, json_path):
            if pth.name in _FORBIDDEN_OUTPUT_NAMES:
                raise ValueError(f"refusing to write forbidden output name {pth.name!r}")
        md_path.write_text(self.render_markdown(), encoding="utf-8")
        json_path.write_text(json.dumps(self.to_json(), indent=2), encoding="utf-8")
        return {"markdown": md_path, "json": json_path}


def make_full_candidates(
    fit: LinearFit,
    *,
    n_cards: int,
    small_games: float,
    medium_games: float,
    large_games: float,
) -> list[FullCandidate]:
    """Build the three FULL_S/M/L candidates from a fit.

    Every candidate's cost is a :meth:`Projection.from_fit`, i.e. PROJECTED —
    there is no path here that yields a MEASURED FULL time.
    """
    spec = (
        ("FULL_S", "dev", small_games),
        ("FULL_M", "nightly", medium_games),
        ("FULL_L", "paper", large_games),
    )
    out: list[FullCandidate] = []
    for tier, purpose, games in spec:
        out.append(
            FullCandidate(
                tier=tier,
                purpose=purpose,
                workload_games=games,
                projection=Projection.from_fit(
                    fit, target_games=games, n_cards=n_cards, label=f"{tier}@{int(games)}"
                ),
            )
        )
    return out


__all__ = [
    "BudgetLine",
    "FullCandidate",
    "BudgetReport",
    "make_full_candidates",
    "REPORT_BASENAME",
    "NOT_FOR_PROMOTION_BANNER",
]
