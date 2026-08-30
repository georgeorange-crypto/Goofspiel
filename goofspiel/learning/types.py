"""Shared dataclasses for learning targets and batches."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class RobustQTarget:
    target_q: Tensor
    valid_mask: Tensor
    source: list[str]
    solver_gap: Tensor | None = None


@dataclass
class PolicyTarget:
    target_policy: Tensor
    source: list[str]
    quality: Tensor


@dataclass
class TrajectoryLearningBatch:
    states: object
    self_actions: Tensor
    opponent_actions: Tensor
    rewards: Tensor
    behavior_prob_self: Tensor
    behavior_prob_opp: Tensor
    final_score_diff: Tensor
    done: Tensor
    policy_version: Tensor


@dataclass
class OpponentLearningBatch:
    public_states: object
    current_game_history: object
    long_term_history: object
    actual_action: Tensor
    opponent_id: Tensor
    strategy_regime_id: Tensor | None = None
    switch_label: Tensor | None = None
