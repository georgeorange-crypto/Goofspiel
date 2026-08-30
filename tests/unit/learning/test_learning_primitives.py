from __future__ import annotations

import pytest

try:
    import torch
except OSError as exc:  # pragma: no cover - machine environment guard
    pytest.skip(f"torch cannot be imported in this environment: {exc}", allow_module_level=True)

from goofspiel.learning.game_theory import nash_anchor_kl, neurd_loss, solve_batch
from goofspiel.learning.losses import opponent_style_infonce
from goofspiel.learning.targets import (
    joint_vtrace_targets,
    lambda_returns,
    project_score_difference_two_hot,
    select_policy_target,
)
from goofspiel.learning.types import PolicyTarget


def test_rm_plus_matching_pennies_close_to_uniform():
    q = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    sol = solve_batch(q, mask, mask, iterations=256)
    assert torch.allclose(sol.row_policy, torch.tensor([[0.5, 0.5]]), atol=0.05)
    assert torch.allclose(sol.column_policy, torch.tensor([[0.5, 0.5]]), atol=0.05)
    assert abs(sol.value.item()) < 0.05


def test_neurd_gradient_is_raw_logit_direction():
    logits_self = torch.tensor([[0.0, 0.0]], requires_grad=True)
    logits_opp = torch.tensor([[0.0, 0.0]])
    q = torch.tensor([[[1.0, -1.0], [0.0, 0.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    loss = neurd_loss(logits_self, logits_opp, q, mask, mask)
    loss.backward()
    assert logits_self.grad is not None
    assert logits_self.grad[0, 0] < 0.0
    assert logits_self.grad[0, 1] > 0.0


def test_lambda_return_endpoints():
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    values = torch.tensor([[10.0], [20.0], [30.0], [0.0]])
    done = torch.tensor([[False], [False], [True]])
    td0 = lambda_returns(rewards, values, done, lambda_=0.0, gamma=1.0)
    mc = lambda_returns(rewards, values, done, lambda_=1.0, gamma=1.0)
    assert torch.allclose(td0, torch.tensor([[21.0], [32.0], [3.0]]))
    assert torch.allclose(mc, torch.tensor([[6.0], [5.0], [3.0]]))


def test_vtrace_on_policy_matches_lambda_return():
    rewards = torch.tensor([[1.0], [2.0]])
    values = torch.tensor([[0.5], [0.25], [0.0]])
    done = torch.tensor([[False], [True]])
    probs = torch.ones_like(rewards)
    vt = joint_vtrace_targets(rewards, values, probs, probs, probs, probs, done, lambda_=0.7)
    lr = lambda_returns(rewards, values, done, lambda_=0.7)
    assert torch.allclose(vt, lr)


def test_two_hot_projection_sums_to_one():
    target = project_score_difference_two_hot(torch.tensor([-1.0, 0.0, 0.125, 1.0]), num_bins=201)
    assert target.shape == (4, 201)
    assert torch.allclose(target.sum(dim=-1), torch.ones(4))


def test_teacher_priority_chooses_exact_policy():
    exact = PolicyTarget(torch.tensor([[1.0, 0.0]]), ["EXACT"], torch.ones(1))
    nash = PolicyTarget(torch.tensor([[0.5, 0.5]]), ["REFERENCE_NASH_Q"], torch.ones(1))
    assert select_policy_target([nash, exact]).source == ["EXACT"]


def test_style_infonce_skips_no_positive_pairs():
    emb = torch.randn(3, 8, requires_grad=True)
    loss = opponent_style_infonce(emb, torch.tensor([1, 2, 3]), torch.tensor([1, 1, 1]))
    assert torch.isfinite(loss)
    assert loss.item() == 0.0


def test_nash_anchor_kl_ignores_illegal_actions():
    logits = torch.tensor([[0.0, 100.0, 0.0]])
    target = torch.tensor([[0.5, 0.0, 0.5]])
    mask = torch.tensor([[True, False, True]])
    loss = nash_anchor_kl(logits, target, mask)
    assert torch.isfinite(loss)
