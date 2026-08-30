"""Shared leaf evaluator for search tools."""

from __future__ import annotations

from dataclasses import dataclass

from goofspiel.game import GameState
from goofspiel.reasoning.exact_br import solve_exact_best_response
from goofspiel.reasoning.exact_tool import solve_exact_tool
from goofspiel.reasoning.search import run_sm_mcts
from goofspiel.reasoning.types import GameToolResult, ToolMode


@dataclass
class LeafEvaluation:
    result: GameToolResult
    provenance: str


class LeafEvaluator:
    def __init__(self, *, exact_max_remaining: int = 4) -> None:
        self.exact_max_remaining = exact_max_remaining

    def evaluate_robust(self, state: GameState, *, mode: ToolMode = ToolMode.PLAY) -> LeafEvaluation:
        exact = solve_exact_tool(state, max_remaining=self.exact_max_remaining, mode=mode)
        if exact.valid:
            return LeafEvaluation(exact, "EXACT_LEAF_OVERRIDE")
        return LeafEvaluation(run_sm_mcts(state, mode=mode), "SM_MCTS_LEAF_FALLBACK")

    def evaluate_adaptive(
        self,
        state: GameState,
        opponent_policy: list[float],
        *,
        mode: ToolMode = ToolMode.TEACHER,
    ) -> LeafEvaluation:
        return LeafEvaluation(solve_exact_best_response(state, opponent_policy, mode=mode), "EXACT_BR_LEAF")
