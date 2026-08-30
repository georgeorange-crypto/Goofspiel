"""Top-level GameAgent API."""

from __future__ import annotations

import torch

from goofspiel.game import GameState
from goofspiel.reasoning.router import AgentReasoningResult, DecisionBudget, ToolRouter
from goofspiel.reasoning.state import ReasoningState
from goofspiel.reasoning.types import ToolMode


class GameAgent:
    def __init__(self, *, model_version: str = "unversioned", budget: DecisionBudget | None = None, seed: int = 0) -> None:
        self.model_version = model_version
        self.router = ToolRouter(budget)
        self.generator = torch.Generator().manual_seed(seed)

    def think(
        self,
        state: GameState,
        *,
        mode: ToolMode = ToolMode.PLAY,
        opponent_history: tuple[object, ...] = (),
    ) -> AgentReasoningResult:
        reasoning_state = ReasoningState(
            public_state=state,
            model_version=self.model_version,
            opponent_history=opponent_history,
        )
        return self.router.think(reasoning_state, mode=mode, generator=self.generator)

    def act(self, state: GameState, *, opponent_history: tuple[object, ...] = ()) -> int:
        return self.think(state, opponent_history=opponent_history).final.action_rank
