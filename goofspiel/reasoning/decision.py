"""Final Decision Protocol implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch
from torch import Tensor

from goofspiel.reasoning.safe_mixture import safe_exploit_mixture
from goofspiel.reasoning.types import Exactness, GameToolResult

TOOL_PRIORITY = {
    "EXACT_NASH": 50,
    "GT_CFR": 40,
    "SM_MCTS": 30,
    "MODEL_MATRIX_NASH": 20,
    "RAW_ACTOR": 10,
}


@dataclass
class FinalDecision:
    policy: Tensor
    action_rank: int
    robust_source: str
    adaptive_used: bool
    safe_alpha: float = 0.0
    provenance: dict[str, object] = field(default_factory=dict)


def validate_tool_result(result: GameToolResult, *, state_key: object | None = None) -> bool:
    p = result.policy_self.float()
    mask = result.valid_self_mask.bool()
    if state_key is not None and result.state_key not in (None, state_key):
        return False
    if not result.valid or not torch.isfinite(p).all():
        return False
    if (p[mask] < -1e-7).any() or (p[~mask].abs() > 1e-6).any():
        return False
    return bool(torch.isclose(p[mask].sum(), torch.tensor(1.0, device=p.device), atol=1e-4))


def select_robust_result(results: Iterable[GameToolResult], *, state_key: object | None = None) -> GameToolResult:
    valid = [r for r in results if validate_tool_result(r, state_key=state_key)]
    exact = [r for r in valid if r.source == "EXACT_NASH" and r.exactness in {Exactness.NUMERICAL_EXACT.value, Exactness.RATIONAL_EXACT.value}]
    if exact:
        return exact[0]
    if not valid:
        raise ValueError("no valid robust tool result")
    return max(valid, key=lambda r: (TOOL_PRIORITY.get(r.source, 0), float(r.quality_score), int(r.simulations)))


def categorical_sample(policy: Tensor, *, generator: torch.Generator | None = None) -> int:
    action_index = int(torch.multinomial(policy.float(), 1, generator=generator).item())
    return action_index + 1


def final_decision(
    robust_results: Iterable[GameToolResult],
    *,
    adaptive_result: GameToolResult | None = None,
    opponent_belief: Tensor | None = None,
    epsilon: float = 0.0,
    generator: torch.Generator | None = None,
    state_key: object | None = None,
) -> FinalDecision:
    robust = select_robust_result(robust_results, state_key=state_key)
    policy = robust.policy_self.float()
    alpha = torch.tensor(0.0, device=policy.device)
    adaptive_used = False
    if adaptive_result is not None and opponent_belief is not None and validate_tool_result(adaptive_result, state_key=state_key):
        q = robust.q_matrix.float()
        if q.ndim == 2:
            q = q.unsqueeze(0)
        mixed, alpha_b = safe_exploit_mixture(
            policy.unsqueeze(0),
            adaptive_result.policy_self.float().unsqueeze(0),
            q,
            opponent_belief.float().unsqueeze(0) if opponent_belief.ndim == 1 else opponent_belief.float(),
            robust.valid_self_mask.bool().unsqueeze(0),
            epsilon=epsilon,
        )
        policy = mixed.squeeze(0)
        alpha = alpha_b.squeeze(0)
        adaptive_used = bool(float(alpha.detach().cpu()) > 0.0)
    action = categorical_sample(policy, generator=generator)
    return FinalDecision(
        policy=policy,
        action_rank=action,
        robust_source=robust.source,
        adaptive_used=adaptive_used,
        safe_alpha=float(alpha.detach().cpu()),
        provenance={"robust_exactness": robust.exactness, "robust_quality": robust.quality_score},
    )
