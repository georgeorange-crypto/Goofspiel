"""Phase 3.2 / 3.2b — P5 trains the opponent/adaptive branch behind a hard firewall.

Before this phase P5 trained *nothing*: it emitted a constant uniform-NLL
diagnostic, reported `opponent_model_usable=0.0`, and saved no checkpoint.  The
opponent-memory (LSTM/Mamba), the `opp_short/long/fused` heads, and the whole
adaptive branch were still at initialization in the final artifact.

Phase 3.2 makes P5 actually train those parameters on scripted-regime sessions;
Phase 3.2b makes the `Q_R ⊥ Q_A` separation a *hard, explicit* firewall rather
than something that merely happens to hold because of a `.detach()` call.

Per the project testing principle these tests RE-EXECUTE the facts, they do not
read a JSON field:

  1. The robust/adaptive partition is an exact partition of the model's params
     (disjoint AND exhaustive) — so no module can silently escape the firewall.
  2. After real P5 steps the robust parameters are **byte-for-byte unchanged**
     (`‖Δθ_R‖ == 0`) while the adaptive parameters **receive gradient**
     (`‖∇θ_A‖ > 0`) and **actually move**.  This is the firewall, re-run.
  3. `set_robust_requires_grad(False)` genuinely stops robust gradient: after a
     backward through the opponent loss, every robust param has `grad is None`.
  4. P5 saves a non-None checkpoint whose robust weights byte-equal the inherited
     P4 weights (the frozen backbone was carried through untouched), and whose
     trained opponent model beats the uniform reference NLL.
"""

from __future__ import annotations

import pytest

try:
    import torch
except OSError as exc:  # pragma: no cover - machine environment guard
    pytest.skip(f"torch cannot be imported in this environment: {exc}", allow_module_level=True)

import torch.nn.functional as F

from goofspiel.models import GoofspielModel, HistoryBatch, OpponentMemoryBatch, public_state_from_game
from goofspiel.game import GameState
from goofspiel.training.checkpoint import load_checkpoint


# ----------------------------------------------------------------------------
# 1. The partition is exact (disjoint + exhaustive).
# ----------------------------------------------------------------------------
def test_robust_adaptive_partition_is_exact():
    m = GoofspielModel(max_cards=13)
    m.assert_partition_is_complete()  # raises if overlap or missing

    robust = {id(p) for p in m.robust_parameters()}
    adaptive = {id(p) for p in m.adaptive_parameters()}
    allp = {id(p) for p in m.parameters()}
    assert robust.isdisjoint(adaptive), "robust and adaptive param sets overlap"
    assert robust | adaptive == allp, "partition does not cover every parameter"
    assert robust and adaptive, "a partition half is empty"


# ----------------------------------------------------------------------------
# 2/3. Freezing robust genuinely stops its gradient; adaptive still flows.
# ----------------------------------------------------------------------------
def _tiny_batch(device="cpu"):
    states = [GameState.initial(5, current_prize=2), GameState.initial(5, current_prize=4)]
    batch = public_state_from_game(states, max_cards=13, device=device)
    steps = 3
    history = HistoryBatch(
        prize=torch.randint(1, 6, (2, steps)),
        self_action=torch.randint(1, 6, (2, steps)),
        opponent_action=torch.randint(1, 6, (2, steps)),
        score_diff=torch.zeros(2, steps),
        outcome=torch.zeros(2, steps),
        round_idx=torch.arange(steps).float().repeat(2, 1),
        valid_mask=torch.ones(2, steps, dtype=torch.bool),
    )
    memory = OpponentMemoryBatch(
        game_summary_sequence=torch.randn(2, 4, 192),
        valid_mask=torch.ones(2, 4, dtype=torch.bool),
    )
    return batch, history, memory


def test_freezing_robust_blocks_its_gradient_but_not_adaptive():
    torch.manual_seed(0)
    m = GoofspielModel(max_cards=13)
    m.set_robust_requires_grad(False)
    batch, history, memory = _tiny_batch()
    out = m(batch, current_game_history=history, long_term_memory=memory)
    target = torch.tensor([0, 4])
    loss = F.cross_entropy(out.opponent_fused_logits, target)
    loss.backward()

    # Every robust param is frozen -> grad is None. Every adaptive param that
    # participates in this loss carries a real gradient.
    for p in m.robust_parameters():
        assert p.grad is None, "a frozen robust param received gradient"
    adaptive_grad = sum(
        float(p.grad.abs().sum()) for p in m.adaptive_parameters() if p.grad is not None
    )
    assert adaptive_grad > 0.0, "adaptive params received no gradient"


# ----------------------------------------------------------------------------
# 4. The full P5 stage: robust unchanged, adaptive moves, checkpoint saved,
#    opponent NLL beats uniform, and robust weights byte-equal inherited P4.
# ----------------------------------------------------------------------------
def test_p5_trains_adaptive_and_freezes_robust_end_to_end(tmp_path):
    from goofspiel.training.stages import run_stage1_pretrain, run_stage5_adaptive

    # Chain from a real P1 so the frozen backbone is a *trained* one.
    p1 = run_stage1_pretrain(steps=1, batch_size=4, out_dir=tmp_path / "ck", n_cards=5)
    p1_state = load_checkpoint(p1.checkpoint)["model_state"]

    p5 = run_stage5_adaptive(
        steps=8, out_dir=tmp_path / "p5", n_cards=5,
        init_from_checkpoint=p1.checkpoint,
    )
    # A real checkpoint now exists (was None before Phase 3.2).
    assert p5.checkpoint is not None
    # The firewall held during training (recorded from the run itself).
    assert p5.metrics["robust_param_delta_l1"] == 0.0
    assert p5.metrics["adaptive_grad_norm_last"] > 0.0
    # The trained opponent model beats the honest uniform reference.
    assert p5.metrics["opponent_nll"] < p5.metrics["uniform_reference_nll"]
    assert p5.metrics["opponent_model_usable"] == 1.0

    p5_state = load_checkpoint(p5.checkpoint)["model_state"]

    # Robust weights in the SAVED checkpoint byte-equal the inherited P1 weights:
    # P5 carried the frozen backbone through untouched.
    m = GoofspielModel(max_cards=13)
    robust_names = set(m._robust_modules().keys())
    adaptive_names = set(m._adaptive_modules().keys())

    def owner(param_name: str) -> str:
        top = param_name.split(".")[0]
        return top

    robust_checked = adaptive_moved = 0
    for name, tensor in p5_state.items():
        top = owner(name)
        if top in robust_names:
            assert torch.equal(tensor, p1_state[name]), f"robust param {name} changed in P5"
            robust_checked += 1
        elif top in adaptive_names:
            # Not every adaptive tensor is guaranteed to move (e.g. unused biases),
            # but the group as a whole must.
            if not torch.equal(tensor, p1_state[name]):
                adaptive_moved += 1
    assert robust_checked > 0, "no robust params were checked — partition/owner mismatch"
    assert adaptive_moved > 0, "no adaptive params moved — P5 did not actually train them"
