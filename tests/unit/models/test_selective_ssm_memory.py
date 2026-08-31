"""Phase 2.3 — `inter_game_mamba` must be a genuine selective state-space model.

Before this phase the inter-game memory was a depthwise conv + `nn.GRU` gate
named ``PlaceholderSequenceMemory`` that deliberately did *not* claim to be a
Mamba.  Phase 2.3 replaces it with a real selective SSM (Mamba/S6): a scan-based
linear recurrence over the games (time) dimension with **input-dependent**
(selective) Δ, B, C discretization.

Per the project testing principle these tests RE-EXECUTE the defining fact rather
than read a name:

  1. There is no GRU/LSTM/RNN anywhere in the module (the placeholder's gate is
     gone), but the SSM parameters (diagonal state matrix A, skip D, selective Δ
     and B/C projections) are present.
  2. The forward pass IS the selective-SSM scan — an independent re-implementation
     of the scan from the module's own weights reproduces `forward()` exactly.  A
     GRU could not match a hand-rolled SSM recurrence.
  3. Selectivity: the discretized state transition Δ⊙A genuinely depends on the
     input (S6, not a linear-time-invariant S4).
  4. State propagates across MORE than the conv width, so it is the scan (not the
     local depthwise conv) carrying game-0 information to the final step.
"""

from __future__ import annotations

import pytest

try:
    import torch
except OSError as exc:  # pragma: no cover - machine environment guard
    pytest.skip(f"torch cannot be imported in this environment: {exc}", allow_module_level=True)

import torch.nn as nn
import torch.nn.functional as F

from goofspiel.models import GoofspielModel, OpponentMemoryBatch
from goofspiel.models.goofspiel_model import SelectiveStateSpaceMemory


def _memory(batch: int, games: int, seed: int, dim: int = 192) -> OpponentMemoryBatch:
    g = torch.Generator().manual_seed(seed)
    return OpponentMemoryBatch(
        game_summary_sequence=torch.randn(batch, games, dim, generator=g),
        valid_mask=torch.ones(batch, games, dtype=torch.bool),
    )


def test_module_is_ssm_not_gru():
    """No recurrent-gate module survives, and the SSM parameters are present."""
    mem = SelectiveStateSpaceMemory(192, 192)
    # The placeholder's gate is gone: no GRU/LSTM/RNN anywhere.
    for sub in mem.modules():
        assert not isinstance(sub, (nn.GRU, nn.LSTM, nn.RNN)), type(sub)
    # The defining SSM parameters exist.
    assert hasattr(mem, "A_log") and mem.A_log.shape == (mem.d_inner, mem.d_state)
    assert hasattr(mem, "D") and mem.D.shape == (mem.d_inner,)
    assert isinstance(mem.dt_proj, nn.Linear)          # selective Δ
    assert isinstance(mem.x_proj, nn.Linear)           # selective B, C
    # A must be a negative-real (stable) diagonal state matrix: A = -exp(A_log).
    A = -torch.exp(mem.A_log)
    assert (A < 0).all()

    # The model's parameter report names it truthfully as mamba_memory and it is
    # exactly this module.
    model = GoofspielModel(max_cards=13)
    assert isinstance(model.inter_game_mamba, SelectiveStateSpaceMemory)
    assert model.parameter_count_by_module()["mamba_memory"] > 0


def test_forward_equals_independent_selective_scan():
    """Re-implement the selective-SSM scan from the module's own weights and show
    it reproduces `forward()` — proving the forward pass is that scan, not a GRU."""
    torch.manual_seed(0)
    mem = SelectiveStateSpaceMemory(192, 192).eval()
    memory = _memory(batch=2, games=6, seed=3)

    with torch.no_grad():
        got = mem(memory, batch=2, device=torch.device("cpu"))

        # --- Independent reference implementation of the same recurrence. ---
        xz = mem.in_proj(memory.game_summary_sequence.float())
        x, z = xz.chunk(2, dim=-1)
        length = x.shape[1]
        x = mem.conv1d(x.transpose(1, 2))[..., :length].transpose(1, 2)
        x = F.silu(x)

        A = -torch.exp(mem.A_log)  # (d_inner, d_state)
        proj = mem.x_proj(x)
        dt, B_mat, C_mat = torch.split(proj, [mem.dt_rank, mem.d_state, mem.d_state], dim=-1)
        delta = F.softplus(mem.dt_proj(dt))
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB_x = delta.unsqueeze(-1) * B_mat.unsqueeze(2) * x.unsqueeze(-1)

        b = x.shape[0]
        h = torch.zeros(b, mem.d_inner, mem.d_state)
        ys = []
        for t in range(length):
            h = deltaA[:, t] * h + deltaB_x[:, t]
            ys.append((h * C_mat[:, t].unsqueeze(1)).sum(dim=-1))
        y = torch.stack(ys, dim=1) + x * mem.D
        y = y * F.silu(z)
        out = mem.out_proj(y)
        idx = memory.valid_mask.float().sum(dim=1).long().clamp_min(1) - 1
        ref = mem.norm(out[torch.arange(b), idx])

    assert torch.allclose(got, ref, atol=1e-5, rtol=1e-4), (got - ref).abs().max()


def test_state_transition_is_input_selective():
    """Δ⊙A (the discretized state transition) must depend on the input — the S6
    property that separates a *selective* SSM from a linear-time-invariant one."""
    torch.manual_seed(1)
    mem = SelectiveStateSpaceMemory(192, 192).eval()

    def delta_of(seed: int) -> torch.Tensor:
        memory = _memory(batch=1, games=5, seed=seed)
        with torch.no_grad():
            xz = mem.in_proj(memory.game_summary_sequence.float())
            x, _ = xz.chunk(2, dim=-1)
            length = x.shape[1]
            x = F.silu(mem.conv1d(x.transpose(1, 2))[..., :length].transpose(1, 2))
            dt = mem.x_proj(x)[..., : mem.dt_rank]
            return F.softplus(mem.dt_proj(dt))

    d0, d1 = delta_of(10), delta_of(20)
    # Different inputs => different Δ (hence different state transition). An LTI
    # SSM would produce identical Δ regardless of input.
    assert not torch.allclose(d0, d1)


def test_scan_propagates_beyond_conv_width():
    """Changing only the FIRST game changes the last-step output across a sequence
    longer than the conv kernel — so the SSM scan (not the local conv) carries the
    long-range dependency."""
    torch.manual_seed(2)
    mem = SelectiveStateSpaceMemory(192, 192).eval()
    games = mem.d_conv + 5  # strictly longer than the conv receptive field
    base = _memory(batch=1, games=games, seed=7)

    perturbed = OpponentMemoryBatch(
        game_summary_sequence=base.game_summary_sequence.clone(),
        valid_mask=base.valid_mask.clone(),
    )
    perturbed.game_summary_sequence[:, 0] += 5.0  # perturb ONLY game 0

    with torch.no_grad():
        y_base = mem(base, batch=1, device=torch.device("cpu"))
        y_pert = mem(perturbed, batch=1, device=torch.device("cpu"))
    # The final-step output at index games-1 (> d_conv-1 steps away from game 0)
    # reacts — impossible for a pure width-d_conv conv, only the recurrent state
    # can carry it.
    assert not torch.allclose(y_base, y_pert, atol=1e-6)
