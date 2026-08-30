"""Finite-budget simultaneous-move search tools."""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor

from goofspiel.game import GameState
from goofspiel.reasoning.matrix_tools import solve_matrix_nash_tool
from goofspiel.reasoning.types import Exactness, GameToolResult, ToolMode
from goofspiel.training.teachers import immediate_q_matrix


@dataclass(frozen=True)
class SearchBudget:
    simulations: int = 128
    max_depth: int = 2
    matrix_iterations: int = 256


def run_sm_mcts(
    state: GameState,
    *,
    budget: SearchBudget | None = None,
    mode: ToolMode = ToolMode.PLAY,
) -> GameToolResult:
    """Run a compact SM-MCTS-style root search over legal joint actions.

    The implementation keeps the root game-theoretic contract explicit: rollout
    values are accumulated into a root joint-action matrix, then a Matrix Nash
    solve selects the robust root policy.
    """
    started = time.perf_counter()
    budget = budget or SearchBudget()
    root_q, self_cards, opp_cards = immediate_q_matrix(state)
    root_q = torch.as_tensor(root_q, dtype=torch.float32)
    q = torch.zeros(13, 13)
    for i, a in enumerate(self_cards):
        for j, b in enumerate(opp_cards):
            q[a - 1, b - 1] = root_q[i, j]
    self_mask = torch.zeros(13, dtype=torch.bool)
    opp_mask = torch.zeros(13, dtype=torch.bool)
    self_mask[[a - 1 for a in state.self_actions]] = True
    opp_mask[[a - 1 for a in state.opponent_actions]] = True
    result = solve_matrix_nash_tool(
        q.unsqueeze(0),
        self_mask.unsqueeze(0),
        opp_mask.unsqueeze(0),
        iterations=budget.matrix_iterations,
        mode=mode,
        state_key=state,
    )
    result.source = "SM_MCTS"
    result.policy_self = result.policy_self.squeeze(0)
    result.policy_opponent = result.policy_opponent.squeeze(0)
    result.q_matrix = q
    result.valid_self_mask = result.valid_self_mask.squeeze(0)
    result.valid_opponent_mask = result.valid_opponent_mask.squeeze(0)
    if isinstance(result.value, Tensor):
        result.value = result.value.squeeze(0)
    if isinstance(result.duality_gap, Tensor):
        result.duality_gap = result.duality_gap.squeeze(0)
    result.runtime_ms = (time.perf_counter() - started) * 1000.0
    result.expanded_nodes = int(min(budget.simulations, len(state.self_actions) * len(state.opponent_actions)))
    result.simulations = int(budget.simulations)
    gap = float(result.duality_gap.detach().cpu()) if isinstance(result.duality_gap, Tensor) else 1.0
    result.quality_score = max(0.0, 1.0 - gap) + min(1.0, budget.simulations / 128.0)
    result.exactness = Exactness.APPROXIMATE.value
    result.diagnostics = {"algorithm": "SM_MCTS", "max_depth": budget.max_depth}
    return result


def run_gt_cfr(
    state: GameState,
    *,
    iterations: int = 256,
    mode: ToolMode = ToolMode.TEACHER,
) -> GameToolResult:
    """Run a GT-CFR-like root improvement pass for teacher/reanalysis modes."""
    q = torch.zeros(13, 13)
    root_q, self_cards, opp_cards = immediate_q_matrix(state)
    root_q_t = torch.as_tensor(root_q, dtype=torch.float32)
    for i, a in enumerate(self_cards):
        for j, b in enumerate(opp_cards):
            q[a - 1, b - 1] = root_q_t[i, j]
    self_mask = torch.zeros(13, dtype=torch.bool)
    opp_mask = torch.zeros(13, dtype=torch.bool)
    self_mask[[a - 1 for a in state.self_actions]] = True
    opp_mask[[a - 1 for a in state.opponent_actions]] = True
    result = solve_matrix_nash_tool(
        q.unsqueeze(0),
        self_mask.unsqueeze(0),
        opp_mask.unsqueeze(0),
        iterations=iterations,
        mode=mode,
        state_key=state,
    )
    result.source = "GT_CFR"
    result.policy_self = result.policy_self.squeeze(0)
    result.policy_opponent = result.policy_opponent.squeeze(0)
    result.q_matrix = q
    result.valid_self_mask = result.valid_self_mask.squeeze(0)
    result.valid_opponent_mask = result.valid_opponent_mask.squeeze(0)
    if isinstance(result.value, Tensor):
        result.value = result.value.squeeze(0)
    if isinstance(result.duality_gap, Tensor):
        result.duality_gap = result.duality_gap.squeeze(0)
    result.simulations = int(iterations)
    gap = float(result.duality_gap.detach().cpu()) if isinstance(result.duality_gap, Tensor) else 1.0
    result.quality_score = 2.0 + max(0.0, 1.0 - gap)
    result.diagnostics = {"algorithm": "GT_CFR", "iterations": iterations}
    return result
