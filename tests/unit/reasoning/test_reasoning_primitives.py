from __future__ import annotations

from goofspiel.game import GameState, transition


def test_pure_transition_carry_and_next_prize_are_deterministic():
    s = GameState.initial(3, current_prize=1)
    out = transition(s, 1, 1, next_prize=2)
    assert out.reward_self == 0
    assert out.state.carry_pool == 1
    assert out.state.current_prize == 2
    out2 = transition(out.state, 3, 2, next_prize=3)
    assert out2.reward_self == 3
    assert out2.normalized_reward == 3 / 6
    assert out2.state.carry_pool == 0


def test_safe_mixture_respects_robust_floor():
    import pytest
    try:
        import torch
    except OSError as exc:  # pragma: no cover - machine environment guard
        pytest.skip(f"torch cannot be imported in this environment: {exc}")
    from goofspiel.reasoning import safe_exploit_mixture

    robust = torch.tensor([[0.5, 0.5]])
    adaptive = torch.tensor([[1.0, 0.0]])
    q = torch.tensor([[[0.0, 0.0], [1.0, -1.0]]])
    belief = torch.tensor([[1.0, 0.0]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    mixed, alpha = safe_exploit_mixture(robust, adaptive, q, belief, mask, robust_value=torch.tensor([0.0]))
    pure_floor = torch.einsum("bi,bij->bj", mixed, q).min(dim=-1).values
    assert pure_floor.item() >= -1e-6
    assert 0.0 <= alpha.item() <= 1.0


def test_final_decision_prefers_exact_over_search():
    import pytest
    try:
        import torch
    except OSError as exc:  # pragma: no cover - machine environment guard
        pytest.skip(f"torch cannot be imported in this environment: {exc}")
    from goofspiel.reasoning import Exactness, GameToolResult, final_decision

    mask = torch.tensor([True, True, False])
    exact = GameToolResult(
        source="EXACT_NASH",
        mode="play",
        policy_self=torch.tensor([0.0, 1.0, 0.0]),
        policy_opponent=torch.tensor([0.5, 0.5, 0.0]),
        q_matrix=torch.zeros(3, 3),
        value=0.0,
        valid_self_mask=mask,
        valid_opponent_mask=mask,
        exactness=Exactness.NUMERICAL_EXACT.value,
    )
    search = GameToolResult(
        source="GT_CFR",
        mode="play",
        policy_self=torch.tensor([1.0, 0.0, 0.0]),
        q_matrix=torch.zeros(3, 3),
        valid_self_mask=mask,
        valid_opponent_mask=mask,
        quality_score=999.0,
    )
    generator = torch.Generator().manual_seed(0)
    decision = final_decision([search, exact], generator=generator)
    assert decision.robust_source == "EXACT_NASH"
    assert decision.action_rank == 2


def test_sm_mcts_and_gt_cfr_return_valid_root_policies():
    import pytest
    try:
        import torch  # noqa: F401
    except OSError as exc:  # pragma: no cover - machine environment guard
        pytest.skip(f"torch cannot be imported in this environment: {exc}")
    from goofspiel.reasoning import SearchBudget, run_gt_cfr, run_sm_mcts

    state = GameState.initial(3, current_prize=1)
    sm = run_sm_mcts(state, budget=SearchBudget(simulations=16, matrix_iterations=32))
    cfr = run_gt_cfr(state, iterations=32)
    assert sm.source == "SM_MCTS"
    assert cfr.source == "GT_CFR"
    assert abs(float(sm.policy_self.sum()) - 1.0) < 1e-4
    assert abs(float(cfr.policy_self.sum()) - 1.0) < 1e-4
