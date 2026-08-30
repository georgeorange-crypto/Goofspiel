"""Strict teacher target priority selectors."""

from __future__ import annotations

from goofspiel.learning.types import PolicyTarget, RobustQTarget

Q_PRIORITY = ("EXACT", "CERTIFIED_SEARCH", "NASH_BELLMAN")
POLICY_PRIORITY = ("EXACT", "CERTIFIED_CFR_SEARCH", "REFERENCE_NASH_Q", "TRAINING_NASH_Q")


def _pick(candidates, priority):
    by_source = {}
    for item in candidates:
        source = item.source[0] if item.source else ""
        by_source[source] = item
    for source in priority:
        if source in by_source:
            return by_source[source]
    raise ValueError("no candidate matches the allowed priority list")


def select_robust_q_target(candidates: list[RobustQTarget]) -> RobustQTarget:
    return _pick(candidates, Q_PRIORITY)


def select_policy_target(candidates: list[PolicyTarget]) -> PolicyTarget:
    return _pick(candidates, POLICY_PRIORITY)
