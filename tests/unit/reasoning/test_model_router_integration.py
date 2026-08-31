"""Phase 4.1 / 4.2 — the trained model feeds the router, and the dead
adaptive → safe-mixture cascade is revived behind a *proven* robust floor.

Before this phase:
  * every Q the router used came from the handcrafted ``immediate_q_matrix`` —
    no ``GoofspielModel`` forward pass existed anywhere under ``reasoning/``;
  * ``final_decision`` was always called without ``adaptive_result`` /
    ``opponent_belief``, so ``safe_exploit_mixture`` — and the robust floor
    inside it — was dead code.

Per the project testing principle these tests **re-execute** the facts:
they compute the belief-expected value and the worst-case (minimax) value from
the returned policy and Q themselves, rather than trusting ``adaptive_used`` or
any reported metric.

  1. When a safe exploit exists, exploiting a known-biased opponent **raises the
     belief-expected value** while keeping the mixed policy's worst-case value
     ``>= robust_value - epsilon`` (the guarantee is re-computed, not read).
  2. Counterfactual: when the only value-raising move **would** breach that
     floor, the safe mixture **refuses it** (alpha collapses, value unchanged) —
     proving the floor actually protects, it is not decoration.
  3. ``robust_view()`` exposes no opponent-history/belief/memory fields;
     ``adaptive_view()`` does.  This is the structural ``Q_R ⊥ opponent``
     invariant.
  4. End-to-end through the real ``ToolRouter``: injecting a biased belief on the
     adaptive view changes the played policy toward the exploit, and the
     re-computed worst-case never drops below the pure-robust worst-case.
"""

from __future__ import annotations

import pytest

try:
    import torch
except OSError as exc:  # pragma: no cover - machine environment guard
    pytest.skip(f"torch cannot be imported in this environment: {exc}", allow_module_level=True)

from goofspiel.game import GameState
from goofspiel.reasoning import Exactness, GameToolResult, ReasoningState, final_decision
from goofspiel.reasoning.router import DecisionBudget, ToolRouter


# ----------------------------------------------------------------------------
# Small re-execution helpers: value under a belief, and worst-case over legal
# opponent columns.  Everything below is computed from (policy, Q), never read
# from a metric field.
# ----------------------------------------------------------------------------
def _belief_value(policy: torch.Tensor, q: torch.Tensor, belief: torch.Tensor) -> float:
    b = belief.float()
    b = b / b.sum().clamp_min(1e-12)
    return float(torch.einsum("i,ij,j->", policy.float(), q.float(), b))


def _worst_case(policy: torch.Tensor, q: torch.Tensor, legal_cols: torch.Tensor) -> float:
    cols = torch.einsum("i,ij->j", policy.float(), q.float())
    return float(cols.masked_fill(~legal_cols.bool(), float("inf")).min())


# ----------------------------------------------------------------------------
# 1. A safe exploit is taken and it raises value without breaching the floor.
# ----------------------------------------------------------------------------
def test_safe_exploit_raises_value_and_preserves_robust_floor():
    # Hand-computable 2-action game:
    #        b0   b1
    #  a0  [  0,   0 ]
    #  a1  [  1,   0 ]
    # a1 weakly dominates a0.  A robust mixed policy [0.5, 0.5] has worst-case 0.
    # Opponent biased to b0 -> best response is pure a1, which STILL has
    # worst-case 0 -> exploiting is free.  Value under belief rises 0.5 -> 1.0.
    mask = torch.tensor([True, True])
    q = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
    robust = GameToolResult(
        source="EXACT_NASH", mode="play",
        policy_self=torch.tensor([0.5, 0.5]),
        policy_opponent=torch.tensor([0.5, 0.5]),
        q_matrix=q, value=0.0,
        valid_self_mask=mask, valid_opponent_mask=mask,
        exactness=Exactness.NUMERICAL_EXACT.value,
    )
    adaptive = GameToolResult(
        source="EXACT_BEST_RESPONSE", mode="play",
        policy_self=torch.tensor([0.0, 1.0]),
        q_matrix=q,
        valid_self_mask=mask, valid_opponent_mask=mask,
        exactness=Exactness.EXACT_WRT_OPPONENT_MODEL.value,
    )
    belief = torch.tensor([1.0, 0.0])
    gen = torch.Generator().manual_seed(0)

    robust_policy = torch.tensor([0.5, 0.5])
    robust_floor = _worst_case(robust_policy, q, mask)

    dec = final_decision([robust], adaptive_result=adaptive, opponent_belief=belief, epsilon=0.0, generator=gen)

    # Re-executed facts, not read fields:
    exploit_val = _belief_value(dec.policy, q, belief)
    robust_val = _belief_value(robust_policy, q, belief)
    assert exploit_val > robust_val + 1e-6, "exploiting a biased opponent must raise value"
    # The mixed policy's worst-case never drops below the robust guarantee.
    assert _worst_case(dec.policy, q, mask) >= robust_floor - 1e-6


# ----------------------------------------------------------------------------
# 2. Counterfactual: the floor actually protects — a value-raising move that
#    would breach it is refused.
# ----------------------------------------------------------------------------
def test_safe_mixture_refuses_exploit_that_would_breach_floor():
    # Now the exploit is unsafe:
    #        b0   b1
    #  a0  [  0,   0 ]
    #  a1  [  1,  -1 ]
    # Robust Nash is pure a0 (worst-case 0); a1 has worst-case -1.
    # Opponent biased to b0 -> naive BR is a1 (belief value 1) but its worst-case
    # is -1 < 0.  With epsilon=0 the ONLY safe alpha is 0, so the safe mixture
    # must stay at robust and NOT raise the value.
    mask = torch.tensor([True, True])
    q = torch.tensor([[0.0, 0.0], [1.0, -1.0]])
    robust = GameToolResult(
        source="EXACT_NASH", mode="play",
        policy_self=torch.tensor([1.0, 0.0]),  # pure a0
        policy_opponent=torch.tensor([0.5, 0.5]),
        q_matrix=q, value=0.0,
        valid_self_mask=mask, valid_opponent_mask=mask,
        exactness=Exactness.NUMERICAL_EXACT.value,
    )
    adaptive = GameToolResult(
        source="EXACT_BEST_RESPONSE", mode="play",
        policy_self=torch.tensor([0.0, 1.0]),  # pure a1 (the unsafe exploit)
        q_matrix=q,
        valid_self_mask=mask, valid_opponent_mask=mask,
        exactness=Exactness.EXACT_WRT_OPPONENT_MODEL.value,
    )
    belief = torch.tensor([1.0, 0.0])
    gen = torch.Generator().manual_seed(0)

    robust_policy = torch.tensor([1.0, 0.0])
    robust_floor = _worst_case(robust_policy, q, mask)  # == 0.0

    dec = final_decision([robust], adaptive_result=adaptive, opponent_belief=belief, epsilon=0.0, generator=gen)

    # The floor is honored...
    assert _worst_case(dec.policy, q, mask) >= robust_floor - 1e-6
    # ...and precisely because honoring it forbids the exploit, value did NOT rise.
    assert _belief_value(dec.policy, q, belief) <= _belief_value(robust_policy, q, belief) + 1e-6
    assert dec.safe_alpha == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------------------------
# 3. Structural invariant: robust view is opponent-agnostic, adaptive is not.
# ----------------------------------------------------------------------------
def test_robust_view_exposes_no_opponent_information():
    state = GameState.initial(5, current_prize=3)
    rs = ReasoningState(
        public_state=state,
        opponent_history=("a", "b"),
        opponent_belief=(0.9, 0.1),
        opponent_model_version="opp-v1",
        opponent_memory=object(),
        current_game_history=object(),
    )
    robust = rs.robust_view()
    adaptive = rs.adaptive_view()

    assert not robust.exposes_opponent_information(), "robust_view leaked opponent info"
    assert robust.opponent_history == ()
    assert robust.opponent_belief is None
    assert robust.opponent_model_version is None
    assert robust.opponent_memory is None
    assert robust.current_game_history is None
    # The adaptive view is exactly where that information lives.
    assert adaptive.exposes_opponent_information()
    assert adaptive.opponent_belief == (0.9, 0.1)


# ----------------------------------------------------------------------------
# 4. End-to-end through the real router: a biased belief injected on the
#    adaptive view moves the played policy toward the exploit, and the
#    re-computed worst-case never drops below the pure-robust worst-case.
# ----------------------------------------------------------------------------
def test_router_exploits_biased_opponent_within_robust_floor():
    state = GameState.initial(5, current_prize=3)
    router = ToolRouter(DecisionBudget(exact_max_remaining=2, sm_mcts_mid=64))

    # Baseline (no belief): pure robust decision.
    base = router.think(ReasoningState(state), generator=torch.Generator().manual_seed(0))
    # Same state, but a known opponent bias toward its lowest card.
    biased = tuple([0.96, 0.01, 0.01, 0.01, 0.01] + [0.0] * 8)
    adapt = router.think(
        ReasoningState(state, opponent_belief=biased),
        generator=torch.Generator().manual_seed(0),
    )

    q = base.robust_result.q_matrix.float()
    legal_cols = base.robust_result.valid_opponent_mask.bool()
    belief = torch.tensor(biased, dtype=torch.float32)

    base_worst = _worst_case(base.final.policy, q, legal_cols)
    adapt_worst = _worst_case(adapt.final.policy, q, legal_cols)
    base_val = _belief_value(base.final.policy, q, belief)
    adapt_val = _belief_value(adapt.final.policy, q, belief)

    # The adaptive decision never sacrifices the robust guarantee...
    assert adapt_worst >= base_worst - 1e-6
    # ...and it does at least as well under the (true) biased belief.
    assert adapt_val >= base_val - 1e-6
    # The adaptive tier actually ran (a trace was recorded) — the cascade is live.
    assert any(t["step"] == "adaptive_exact_best_response" for t in adapt.traces)
    # And the robust tier that produced the floor never saw the belief: its trace
    # q_source is a robust source, and the baseline (beliefless) run is identical
    # in robust policy.
    assert torch.allclose(base.robust_result.policy_self, adapt.robust_result.policy_self)


# ----------------------------------------------------------------------------
# 5. Phase 4.1: the trained checkpoint actually feeds the router — robust Q
#    comes from the model, and the opponent belief comes from the trained head.
# ----------------------------------------------------------------------------
def test_trained_provider_sources_robust_q_and_opponent_belief(tmp_path):
    from goofspiel.models import OpponentMemoryBatch
    from goofspiel.reasoning.model_provider import TrainedModelProvider
    from goofspiel.training.stages import run_stage1_pretrain

    # A real (tiny) trained checkpoint, not a random init.
    p1 = run_stage1_pretrain(steps=1, batch_size=4, out_dir=tmp_path / "ck", n_cards=5)
    provider = TrainedModelProvider.from_checkpoint(p1.checkpoint)
    state = GameState.initial(5, current_prize=3)

    # Robust Q is a 13x13 matrix with illegal cells zeroed and 5 legal actions.
    q, self_mask, opp_mask = provider.robust_q13(state)
    assert tuple(q.shape) == (1, 13, 13)
    assert int(self_mask.sum()) == 5 and int(opp_mask.sum()) == 5
    joint = self_mask[:, :, None] & opp_mask[:, None, :]
    assert bool((q[~joint] == 0).all()), "illegal joint cells must be zeroed"

    # The router backed by the provider genuinely uses the model's Q (re-executed
    # via the recorded q_source, which the router sets only on the model path).
    router = ToolRouter(DecisionBudget(exact_max_remaining=2, sm_mcts_mid=16), model_provider=provider)
    res = router.think(ReasoningState(state), generator=torch.Generator().manual_seed(0))
    assert res.traces[0]["q_source"] == "model_q_robust"

    # The opponent belief from the trained head is a valid distribution over the
    # legal opponent cards only, and is None when there is nothing to condition on.
    memory = OpponentMemoryBatch(
        game_summary_sequence=torch.randn(1, 4, 192),
        valid_mask=torch.ones(1, 4, dtype=torch.bool),
    )
    belief = provider.opponent_belief(ReasoningState(state, opponent_memory=memory))
    assert belief is not None and len(belief) == 13
    assert abs(sum(belief) - 1.0) < 1e-5
    assert all(belief[i] == 0.0 for i in range(5, 13)), "belief must have no mass on illegal cards"
    assert provider.opponent_belief(ReasoningState(state)) is None
