from __future__ import annotations

import pytest

from goofspiel.game import GameState

try:
    import torch
except OSError as exc:  # pragma: no cover
    pytestmark = pytest.mark.skip(reason=f"torch cannot be imported in this environment: {exc}")


def test_game_agent_exact_priority_endgame():
    from goofspiel.reasoning import DecisionBudget, GameAgent

    agent = GameAgent(budget=DecisionBudget(exact_max_remaining=3), seed=0)
    result = agent.think(GameState.initial(3, current_prize=1))
    assert result.robust_result.source == "EXACT_NASH"
    assert result.final.action_rank in {1, 2, 3}
    assert any(trace["step"] == "exact_preflight_and_solve" for trace in result.traces)


def test_router_uses_robust_view_without_opponent_history():
    from goofspiel.reasoning import DecisionBudget, ReasoningState, ToolRouter

    state = GameState.initial(5, current_prize=1)
    router = ToolRouter(DecisionBudget(exact_max_remaining=2, sm_mcts_mid=16))
    a = router.think(ReasoningState(state, opponent_history=("x",)), generator=torch.Generator().manual_seed(3))
    b = router.think(ReasoningState(state, opponent_history=("y",)), generator=torch.Generator().manual_seed(3))
    assert torch.allclose(a.robust_result.policy_self, b.robust_result.policy_self)


def test_exact_best_response_returns_pure_adaptive_candidate():
    from goofspiel.reasoning import solve_exact_best_response

    result = solve_exact_best_response(GameState.initial(3, current_prize=1), [1.0, 0.0, 0.0])
    assert result.source == "EXACT_BEST_RESPONSE"
    assert result.policy_self.sum().item() == pytest.approx(1.0)
