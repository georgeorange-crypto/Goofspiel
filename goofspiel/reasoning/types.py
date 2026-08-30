"""Shared tool-result schemas for agent reasoning.

This module deliberately avoids importing torch at module import time.  The
data contract is used by routers, logs, and dry-run checks that must remain
inspectable before the GPU training environment is ready.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolMode(str, Enum):
    PLAY = "play"
    TEACHER = "teacher"
    REANALYSE = "reanalyse"
    REDTEAM_CORRECTION = "redteam_correction"
    EVALUATION = "evaluation"


class Exactness(str, Enum):
    NONE = "NONE"
    APPROXIMATE = "APPROXIMATE"
    NUMERICAL_EXACT = "NUMERICAL_EXACT"
    EXACT_WRT_OPPONENT_MODEL = "EXACT_WRT_OPPONENT_MODEL"
    RATIONAL_EXACT = "RATIONAL_EXACT"


@dataclass
class GameToolResult:
    source: str
    mode: str
    policy_self: Any
    valid_self_mask: Any
    valid_opponent_mask: Any
    policy_opponent: Any | None = None
    q_matrix: Any | None = None
    value: float | Any | None = None
    quality_score: float = 0.0
    duality_gap: float | Any | None = None
    exactness: str = Exactness.NONE.value
    runtime_ms: float = 0.0
    expanded_nodes: int = 0
    simulations: int = 0
    exact_leaf_hits: int = 0
    neural_leaf_hits: int = 0
    state_key: Any = None
    model_version: str | None = None
    opponent_model_version: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    valid: bool = True

    @classmethod
    def matrix_nash(
        cls,
        policy_self: Any,
        policy_opponent: Any,
        q_matrix: Any,
        value: Any,
        masks: tuple[Any, Any],
        *,
        duality_gap: Any | None = None,
        mode: ToolMode = ToolMode.PLAY,
    ) -> "GameToolResult":
        import torch

        return cls(
            source="MODEL_MATRIX_NASH",
            mode=mode.value,
            policy_self=policy_self,
            policy_opponent=policy_opponent,
            q_matrix=q_matrix,
            value=value,
            valid_self_mask=masks[0],
            valid_opponent_mask=masks[1],
            quality_score=1.0,
            duality_gap=duality_gap,
            exactness=Exactness.APPROXIMATE.value,
            valid=bool(torch.isfinite(policy_self).all().item()),
        )
