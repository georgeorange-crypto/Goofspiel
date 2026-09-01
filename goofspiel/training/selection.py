"""Phase 5 — per-axis checkpoint selection built on the 0.1 honest harness.

The registry used to alias ``best_robust`` / ``best_search`` /
``best_generalization`` all onto the *same* P4 file: a per-axis "best" that no
per-axis evaluation ever produced.  This module replaces that fiction with three
**genuinely distinct** evaluations — each computed from the trained policy by
re-playing it, none a literal — so the three aliases can resolve to *different*
checkpoints when the candidates genuinely differ:

``best_robust``          argmax of mean score-diff vs ``Random`` at the primary N
                         (raw robust play strength).
``best_search``          argmin of exact **full-game exploitability** at a small
                         N (the best-response gap a solver/search minimises — the
                         optimality axis, a different objective from win-rate).
``best_generalization``  argmax of the **worst** mean score-diff vs ``Random``
                         across a set of board sizes (does not collapse on an N it
                         was not primarily trained on).

Because the three axes optimise different quantities, a policy that crushes
Random can still be more exploitable than a flatter policy, and the winners split.
Every number below comes from real play through :mod:`goofspiel.training.model_eval`;
nothing is hard-coded.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from goofspiel.training.model_eval import (
    PolicyFn,
    full_game_exploitability,
    load_model_from_checkpoint,
    play_policy_vs_bot,
    robust_policy_fn,
)

# The three axes and whether the winner is the max or the min of the metric.
AXES: dict[str, str] = {
    "best_robust": "max",
    "best_search": "min",
    "best_generalization": "max",
}
_AXIS_METRIC = {
    "best_robust": "robust_score",
    "best_search": "search_exploitability",
    "best_generalization": "generalization_worst",
}


@dataclass
class AxisMetrics:
    """The three computed axis metrics for a single candidate policy."""

    robust_score: float
    generalization_worst: float
    search_exploitability: float | None
    per_n_score: dict[int, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "robust_score": self.robust_score,
            "generalization_worst": self.generalization_worst,
            "search_exploitability": self.search_exploitability,
            "per_n_score": {str(k): v for k, v in self.per_n_score.items()},
        }


@dataclass
class AxisSelection:
    """Per-axis winners + the full per-candidate metric table (all recomputed)."""

    by_alias: dict[str, str]
    table: dict[str, dict[str, Any]]

    def distinct_winner_count(self) -> int:
        return len(set(self.by_alias.values()))


def axis_metrics_for_policy(
    policy: PolicyFn,
    *,
    primary_n: int = 5,
    generalization_ns: Sequence[int] = (3, 5),
    exploit_n: int = 4,
    num_games: int = 24,
    seed: int = 1,
) -> AxisMetrics:
    """Compute the three axis metrics for one policy by REAL play.

    ``robust_score`` and ``generalization_worst`` come from ``play_policy_vs_bot``
    (mean score-diff vs Random); ``search_exploitability`` is the exact full-game
    best-response gap at a small N.  All deterministic given ``seed``.
    """
    per_n: dict[int, float] = {}
    for n in sorted({primary_n, *generalization_ns}):
        per_n[n] = float(
            play_policy_vs_bot(policy, "random", n_cards=n, num_games=num_games, seed=seed)["mean_score_diff"]
        )
    robust_score = per_n.get(primary_n)
    if robust_score is None:  # primary_n outside the union (shouldn't happen) -> compute it
        robust_score = float(
            play_policy_vs_bot(policy, "random", n_cards=primary_n, num_games=num_games, seed=seed)["mean_score_diff"]
        )
        per_n[primary_n] = robust_score
    generalization_worst = min(per_n[n] for n in generalization_ns)
    exploit = full_game_exploitability(policy, n_cards=exploit_n, max_n=exploit_n)
    return AxisMetrics(
        robust_score=float(robust_score),
        generalization_worst=float(generalization_worst),
        search_exploitability=None if exploit is None else float(exploit),
        per_n_score=per_n,
    )


def select_axis_winners(table: Mapping[str, AxisMetrics]) -> dict[str, str]:
    """Given ``candidate_id -> AxisMetrics``, return ``alias -> winning id``.

    Each alias is an independent argmax/argmin over the candidates on its OWN
    metric, so two aliases resolve to the same id only when the same candidate
    happens to win both objectives — never by construction.
    """
    if not table:
        raise ValueError("cannot select over an empty candidate table")
    winners: dict[str, str] = {}
    for alias, mode in AXES.items():
        metric_name = _AXIS_METRIC[alias]
        scored = [
            (cid, getattr(m, metric_name))
            for cid, m in table.items()
            if getattr(m, metric_name) is not None
        ]
        if not scored:
            # No candidate has a computable value for this axis (e.g. exploitability
            # refused for all): fall back to the robust winner so the alias still
            # resolves to a real, evaluated checkpoint rather than nothing.
            scored = [(cid, m.robust_score) for cid, m in table.items()]
            mode = "max"
        # Deterministic winner: best metric, then smallest id on ties. For "max"
        # sort by descending metric; for "min" by ascending; id ascending breaks ties.
        if mode == "max":
            scored.sort(key=lambda pair: (-pair[1], pair[0]))
        else:
            scored.sort(key=lambda pair: (pair[1], pair[0]))
        winners[alias] = scored[0][0]
    return winners


def select_checkpoints_by_axis(
    candidates: Mapping[str, str | Path],
    *,
    device: str = "cpu",
    primary_n: int = 5,
    generalization_ns: Sequence[int] = (3, 5),
    exploit_n: int = 4,
    num_games: int = 24,
    seed: int = 1,
    greedy: bool = True,
    policy_loader: Callable[[str | Path], PolicyFn] | None = None,
) -> AxisSelection:
    """Evaluate every candidate checkpoint on all three axes and pick a winner
    per axis.  Returns the winners AND the full recomputed metric table.

    ``policy_loader`` is injectable for tests (map a checkpoint path to a
    ``policy_fn``); by default it loads the ``GoofspielModel`` robust policy.
    """

    def _default_loader(path: str | Path) -> PolicyFn:
        model, _meta = load_model_from_checkpoint(path, device=device)
        return robust_policy_fn(model, device=device, greedy=greedy)

    loader = policy_loader or _default_loader
    metrics: dict[str, AxisMetrics] = {}
    for cid, path in candidates.items():
        policy = loader(path)
        metrics[cid] = axis_metrics_for_policy(
            policy,
            primary_n=primary_n,
            generalization_ns=generalization_ns,
            exploit_n=exploit_n,
            num_games=num_games,
            seed=seed,
        )
    winners = select_axis_winners(metrics)
    table = {cid: m.as_dict() for cid, m in metrics.items()}
    return AxisSelection(by_alias=winners, table=table)


__all__ = [
    "AXES",
    "AxisMetrics",
    "AxisSelection",
    "axis_metrics_for_policy",
    "select_axis_winners",
    "select_checkpoints_by_axis",
]
