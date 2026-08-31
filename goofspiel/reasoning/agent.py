"""Top-level GameAgent API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from goofspiel.game import GameState
from goofspiel.reasoning.router import AgentReasoningResult, DecisionBudget, ToolRouter
from goofspiel.reasoning.state import ReasoningState
from goofspiel.reasoning.types import ToolMode


class GameAgent:
    def __init__(
        self,
        *,
        model_version: str = "unversioned",
        budget: DecisionBudget | None = None,
        seed: int = 0,
        model_provider: Any | None = None,
        checkpoint: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_version = model_version
        # Phase 4.1: optionally back the router with a trained checkpoint.  A
        # checkpoint path is loaded into a TrainedModelProvider; an explicit
        # provider is used as-is.  Absent both, the router uses the handcrafted
        # immediate matrix exactly as before.
        if model_provider is None and checkpoint is not None:
            from goofspiel.reasoning.model_provider import TrainedModelProvider

            model_provider = TrainedModelProvider.from_checkpoint(checkpoint, device=device)
        self.model_provider = model_provider
        self.router = ToolRouter(budget, model_provider=model_provider)
        self.generator = torch.Generator().manual_seed(seed)

    def think(
        self,
        state: GameState,
        *,
        mode: ToolMode = ToolMode.PLAY,
        opponent_history: tuple[object, ...] = (),
        opponent_belief: tuple[float, ...] | None = None,
        opponent_memory: Any = None,
        current_game_history: Any = None,
    ) -> AgentReasoningResult:
        reasoning_state = ReasoningState(
            public_state=state,
            model_version=self.model_version,
            opponent_history=opponent_history,
            opponent_belief=opponent_belief,
            opponent_memory=opponent_memory,
            current_game_history=current_game_history,
        )
        return self.router.think(reasoning_state, mode=mode, generator=self.generator)

    def act(
        self,
        state: GameState,
        *,
        opponent_history: tuple[object, ...] = (),
        opponent_belief: tuple[float, ...] | None = None,
        opponent_memory: Any = None,
        current_game_history: Any = None,
    ) -> int:
        return self.think(
            state,
            opponent_history=opponent_history,
            opponent_belief=opponent_belief,
            opponent_memory=opponent_memory,
            current_game_history=current_game_history,
        ).final.action_rank
