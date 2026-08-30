"""Exact best-response interface for opponent-adaptive evaluation."""

from __future__ import annotations

from typing import Sequence

import torch

from goofspiel.game import GameState
from goofspiel.reasoning.types import Exactness, GameToolResult, ToolMode
from goofspiel.training.teachers import immediate_q_matrix


def solve_exact_best_response(
    state: GameState,
    opponent_policy: Sequence[float],
    *,
    mode: ToolMode = ToolMode.TEACHER,
) -> GameToolResult:
    """Compute a one-step exact best response to a supplied opponent policy.

    This is the adaptive mathematical tool boundary.  It does not feed
    opponent history into robust search; callers pass an explicit belief.
    """
    q_np, self_cards, opp_cards = immediate_q_matrix(state)
    q_small = torch.as_tensor(q_np, dtype=torch.float32)
    belief_full = torch.zeros(13, dtype=torch.float32)
    for idx, prob in enumerate(opponent_policy[:13]):
        belief_full[idx] = float(prob)
    belief = torch.tensor([belief_full[b - 1] for b in opp_cards], dtype=torch.float32)
    belief = belief / belief.sum().clamp_min(1e-12)
    action_values = q_small @ belief
    best = int(action_values.argmax().item())
    policy = torch.zeros(13)
    policy[self_cards[best] - 1] = 1.0
    opp = torch.zeros(13)
    for idx, b in enumerate(opp_cards):
        opp[b - 1] = belief[idx]
    q = torch.zeros(13, 13)
    for i, a in enumerate(self_cards):
        for j, b in enumerate(opp_cards):
            q[a - 1, b - 1] = q_small[i, j]
    self_mask = torch.zeros(13, dtype=torch.bool)
    opp_mask = torch.zeros(13, dtype=torch.bool)
    self_mask[[a - 1 for a in state.self_actions]] = True
    opp_mask[[a - 1 for a in state.opponent_actions]] = True
    return GameToolResult(
        source="EXACT_BEST_RESPONSE",
        mode=mode.value,
        policy_self=policy,
        policy_opponent=opp,
        q_matrix=q,
        value=float(action_values[best].item()),
        valid_self_mask=self_mask,
        valid_opponent_mask=opp_mask,
        exactness=Exactness.EXACT_WRT_OPPONENT_MODEL.value,
        quality_score=5.0,
        diagnostics={"opponent_policy_source": "explicit_belief"},
    )
