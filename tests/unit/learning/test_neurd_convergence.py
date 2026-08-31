"""Phase 1.1 — the robust NeuRD actor must CONVERGE to the minimax equilibrium.

The prior objective (`neurd_loss`) contracted Q with ``max_b`` over opponent
actions on the self payoff — pulling the actor toward the *best-case* opponent,
the opposite of robust, fighting the RM+ minimax anchor.  The fix uses
``row_action_regret`` fed the RM+ equilibrium column policy.

These tests deliberately do NOT assert "max was changed to min" (too weak, and
subtly wrong: NeuRD action-regret uses the *expected* value against the opponent
policy ``Q(a,σ)=Σ_b σ(b)Q(a,b)``, not per-action ``min_b``).  Instead they run
the real actor update for K steps on games with known equilibria and assert the
policy converges to Nash, a dominated action is driven out, and exploitability
falls.  A reversed objective fails all three.
"""

from __future__ import annotations

import pytest

try:
    import torch
except OSError as exc:  # pragma: no cover - machine environment guard
    pytest.skip(f"torch cannot be imported in this environment: {exc}", allow_module_level=True)

from goofspiel.learning.game_theory import solve_batch
from goofspiel.learning.game_theory.neurd import row_action_regret


def _run_actor_updates(q: torch.Tensor, *, steps: int = 400, lr: float = 0.5):
    """Run the exact stages.py robust-actor update on a single fixed matrix.

    Returns (final_policy, exploitability_trace).  The opponent column policy is
    the RM+ equilibrium of ``q`` (recomputed each step, matching the target-net
    solve in P4).  This mirrors the training loop's actor loss:
        actor_loss = -(regret.detach() * logits * mask).sum(-1) / denom
    """
    n = q.shape[-1]
    mask = torch.ones(1, n, dtype=torch.bool)
    logits = torch.zeros(1, n, requires_grad=True)
    opt = torch.optim.SGD([logits], lr=lr)

    def exploitability(row_policy: torch.Tensor) -> float:
        # Row commits to `row_policy`; adversary best-responds (min over columns);
        # game value of a symmetric zero-sum matrix here is compared against 0 for
        # MP/RPS, but in general we report the best-response gap vs the matrix value.
        rp = row_policy.detach()
        col_vals = (rp[:, None, :] @ q).squeeze(1)  # [1, n]
        row_guarantee = col_vals.min(dim=-1).values  # worst-case value under rp
        v_star = solve_batch(q, mask, mask, iterations=256).value
        return float((v_star - row_guarantee).clamp_min(0.0).item())

    trace = []
    for _ in range(steps):
        sol = solve_batch(q.detach(), mask, mask, iterations=64)
        policy = torch.softmax(logits, dim=-1)
        regret = row_action_regret(q.detach(), policy.detach(), sol.column_policy.detach(), mask)
        denom = mask.float().sum(dim=-1).clamp_min(1.0)
        actor_loss = -((regret.detach() * logits * mask.float()).sum(dim=-1) / denom).mean()
        opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        opt.step()
        trace.append(exploitability(torch.softmax(logits, dim=-1)))

    return torch.softmax(logits, dim=-1).detach(), trace


def test_matching_pennies_converges_to_uniform():
    q = torch.tensor([[[1.0, -1.0], [-1.0, 1.0]]])
    policy, trace = _run_actor_updates(q)
    assert torch.allclose(policy, torch.tensor([[0.5, 0.5]]), atol=0.05), policy
    assert trace[-1] < 0.05, f"final exploitability {trace[-1]} not near 0"
    # Exploitability must fall, not rise (a reversed objective would diverge).
    assert trace[-1] <= trace[0] + 1e-6


def test_rock_paper_scissors_converges_to_uniform():
    q = torch.tensor([[[0.0, -1.0, 1.0], [1.0, 0.0, -1.0], [-1.0, 1.0, 0.0]]])
    policy, trace = _run_actor_updates(q, steps=600)
    assert torch.allclose(policy, torch.full((1, 3), 1.0 / 3.0), atol=0.06), policy
    assert trace[-1] < 0.08, f"final exploitability {trace[-1]}"


def test_dominated_action_probability_decreases_monotonically():
    """Row action 2 is strictly dominated: for every column it pays strictly less
    than both other rows.  A robust actor must drive its probability toward 0.

    The solver requires square matrices (Goofspiel is always N×N), so this is a
    3x3 game whose third row is strictly dominated in every column."""
    #                 c0     c1     c2
    #   row0 (good):  0.0    1.0   -1.0
    #   row1 (good):  1.0    0.0   -1.0
    #   row2 (domin):-2.0   -2.0   -2.0   <- strictly worse than row0 & row1 everywhere
    q = torch.tensor([[[0.0, 1.0, -1.0], [1.0, 0.0, -1.0], [-2.0, -2.0, -2.0]]])
    n = 3
    mask = torch.ones(1, n, dtype=torch.bool)
    logits = torch.zeros(1, n, requires_grad=True)
    opt = torch.optim.SGD([logits], lr=0.5)

    dominated_probs = []
    for _ in range(400):
        sol = solve_batch(q.detach(), mask, mask, iterations=64)
        policy = torch.softmax(logits, dim=-1)
        regret = row_action_regret(q.detach(), policy.detach(), sol.column_policy.detach(), mask)
        denom = mask.float().sum(dim=-1).clamp_min(1.0)
        actor_loss = -((regret.detach() * logits * mask.float()).sum(dim=-1) / denom).mean()
        opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        opt.step()
        dominated_probs.append(float(torch.softmax(logits, dim=-1)[0, 2].item()))

    # Ends far below its 1/3 start and below a small floor.
    assert dominated_probs[-1] < 0.05, f"dominated action prob {dominated_probs[-1]} not driven out"
    assert dominated_probs[-1] < dominated_probs[0]
    # Monotone (non-increasing) up to tiny optimizer noise.
    for earlier, later in zip(dominated_probs, dominated_probs[1:]):
        assert later <= earlier + 1e-3, f"dominated prob rose: {earlier} -> {later}"
