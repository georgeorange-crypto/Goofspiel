"""Hard-rule tool router for Goofspiel decisions."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from goofspiel.game import GameState
from goofspiel.reasoning.decision import FinalDecision, final_decision
from goofspiel.reasoning.exact_tool import solve_exact_tool
from goofspiel.reasoning.matrix_tools import solve_matrix_nash_tool
from goofspiel.reasoning.search import SearchBudget, run_gt_cfr, run_sm_mcts
from goofspiel.reasoning.state import ReasoningState
from goofspiel.reasoning.types import GameToolResult, ToolMode


@dataclass(frozen=True)
class DecisionBudget:
    max_wall_ms: int = 200
    exact_max_remaining: int = 4
    sm_mcts_low: int = 128
    sm_mcts_mid: int = 512
    sm_mcts_high: int = 2048
    gt_cfr_iterations: int = 256


@dataclass
class AgentReasoningResult:
    robust_result: GameToolResult
    tool_results: list[GameToolResult]
    final: FinalDecision
    traces: list[dict[str, object]] = field(default_factory=list)


def _immediate_q13(state: GameState) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from goofspiel.training.teachers import immediate_q_matrix

    q = torch.zeros(1, 13, 13)
    q_small, self_cards, opp_cards = immediate_q_matrix(state)
    q_small_t = torch.as_tensor(q_small, dtype=torch.float32)
    for i, a in enumerate(self_cards):
        for j, b in enumerate(opp_cards):
            q[0, a - 1, b - 1] = q_small_t[i, j]
    self_mask = torch.zeros(1, 13, dtype=torch.bool)
    opp_mask = torch.zeros(1, 13, dtype=torch.bool)
    self_mask[0, [a - 1 for a in state.self_actions]] = True
    opp_mask[0, [a - 1 for a in state.opponent_actions]] = True
    return q, self_mask, opp_mask


def strategic_importance(state: GameState) -> float:
    future_mass = sum(state.opponent_actions) / max(1, state.total_prize_mass)
    current = state.stake / max(1, state.total_prize_mass)
    early = 1.0 - (state.round_index - 1) / max(1, state.n)
    return float(min(1.0, 0.5 * current + 0.25 * future_mass + 0.25 * early))


class ToolRouter:
    """Implements the fixed router order from the design documents."""

    def __init__(self, budget: DecisionBudget | None = None) -> None:
        self.budget = budget or DecisionBudget()

    def think(
        self,
        reasoning_state: ReasoningState,
        *,
        mode: ToolMode = ToolMode.PLAY,
        generator: torch.Generator | None = None,
    ) -> AgentReasoningResult:
        robust_state = reasoning_state.robust_view().public_state
        q, self_mask, opp_mask = _immediate_q13(robust_state)
        matrix = solve_matrix_nash_tool(
            q,
            self_mask,
            opp_mask,
            iterations=128,
            mode=mode,
            state_key=reasoning_state.canonical_key,
            model_version=reasoning_state.model_version,
        )
        matrix.policy_self = matrix.policy_self.squeeze(0)
        matrix.policy_opponent = matrix.policy_opponent.squeeze(0)
        matrix.q_matrix = matrix.q_matrix.squeeze(0)
        matrix.valid_self_mask = matrix.valid_self_mask.squeeze(0)
        matrix.valid_opponent_mask = matrix.valid_opponent_mask.squeeze(0)
        tools = [matrix]
        traces = [{"step": "matrix_nash", "valid": matrix.valid}]

        exact = solve_exact_tool(robust_state, max_remaining=self.budget.exact_max_remaining, mode=mode)
        exact.state_key = reasoning_state.canonical_key
        tools.append(exact)
        traces.append({"step": "exact_preflight_and_solve", "valid": exact.valid, "exactness": exact.exactness})
        if not exact.valid and mode != ToolMode.PLAY:
            gt = run_gt_cfr(robust_state, iterations=self.budget.gt_cfr_iterations, mode=mode)
            gt.state_key = reasoning_state.canonical_key
            tools.append(gt)
            traces.append({"step": "gt_cfr", "valid": gt.valid, "iterations": gt.simulations})
        if not exact.valid and strategic_importance(robust_state) >= 0.35:
            sims = self.budget.sm_mcts_mid if mode == ToolMode.PLAY else self.budget.sm_mcts_high
            sm = run_sm_mcts(robust_state, budget=SearchBudget(simulations=sims), mode=mode)
            sm.state_key = reasoning_state.canonical_key
            tools.append(sm)
            traces.append({"step": "sm_mcts", "valid": sm.valid, "simulations": sm.simulations})

        final = final_decision(tools, generator=generator, state_key=reasoning_state.canonical_key)
        robust = next(tool for tool in tools if tool.source == final.robust_source and tool.valid)
        return AgentReasoningResult(robust_result=robust, tool_results=tools, final=final, traces=traces)
