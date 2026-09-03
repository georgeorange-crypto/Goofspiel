"""Phase 2 — Stage5 GPU-ification: device threading, RE-EXECUTED, firewall intact.

Phase 2 moves the *rank0-only* Stage5 adaptive branch off hardcoded CPU onto the
device the coordinator resolves (default ``"cpu"``), WITHOUT DDP-wrapping Stage5
and WITHOUT touching the gradient firewall, control-plane, loss, architecture, or
session generation.  The single device-sensitive hazard is the nested
public-state batch: nested tensors do NOT move recursively under ``.to(cuda)``,
so Stage5 must CONSTRUCT every tensor natively on the target device.

Per the project testing principle these prove the device facts by RE-EXECUTION —
they run the real tensor builder / a real Stage5 training step / the real
``run_stage5_adaptive`` dispatch and inspect the actual ``.device`` of the
resulting tensors and model params.  They never read a stored field, and they
never mock the path under test.

Every device-parametrised test runs on CPU everywhere and ALSO on CUDA wherever a
GPU is present.  So the exact assertions that pass locally on cpu become the
GPU-correctness proof on the H100 gate — no GPU-only path goes unexercised.
"""

from __future__ import annotations

import dataclasses

import pytest

try:
    import torch
except OSError as exc:  # pragma: no cover - machine environment guard
    pytest.skip(f"torch cannot be imported in this environment: {exc}", allow_module_level=True)

import torch.nn.functional as F

from goofspiel.models import GoofspielModel
from goofspiel.training.data import OpponentSession, RoundRecord
from goofspiel.training.stages import _build_adaptive_training_tensors


# Parametrise every device fact over cpu (always) + cuda (only where present),
# so the SAME assertions become the H100 GPU-correctness proof on a GPU host.
def _devices() -> list[str]:
    devs = ["cpu"]
    if torch.cuda.is_available():  # pragma: no cover - only on GPU hosts (H100 gate)
        devs.append("cuda")
    return devs


def _named_tensors(obj):
    """Yield ``(field_name, tensor)`` for every *declared* tensor field of a batch
    dataclass.  Walking the declared fields (not a hand-picked few) is what makes
    a single tensor left on the wrong device impossible to slip past the check."""
    for f in dataclasses.fields(obj):
        val = getattr(obj, f.name)
        if isinstance(val, torch.Tensor):
            yield f.name, val


def _make_sessions() -> list[OpponentSession]:
    """Two real opponent sessions of a couple of multi-round games each — the same
    shape Stage5's own generator hands the builder.  Two games per session means
    the inter-game memory sequence is non-empty, so the Mamba path is exercised
    too (a decision point in game 2 sees game 1 as prior context)."""

    def game(rows):
        return [
            RoundRecord(
                round_index=i,
                prize=p,
                self_action=sa,
                opponent_action=oa,
                reward_self=rs,
                reward_opponent=ro,
                carry_in=0,
                carry_out=0,
                done=(i == len(rows) - 1),
            )
            # (prize, self_action, opponent_action, reward_self, reward_opponent)
            for i, (p, sa, oa, rs, ro) in enumerate(rows)
        ]

    g1 = game([(2, 1, 3, 0, 2), (3, 2, 1, 3, 0), (1, 3, 2, 0, 0)])
    g2 = game([(1, 2, 1, 1, 0), (3, 3, 2, 3, 0), (2, 1, 3, 0, 2)])
    return [
        OpponentSession("s0", "opp_a", "regime_hi", [g1, g2]),
        OpponentSession("s1", "opp_b", "regime_lo", [g2, g1]),
    ]


# ---------------------------------------------------------------------------
# A. The adaptive training tensors are BUILT NATIVELY on the target device.
#    This is the nested-batch device trap: a shallow ``.to(cuda)`` would leave
#    the nested public-state / history / memory tensors on cpu.  We assert every
#    declared field of every batch, plus the target, lands on the device.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", _devices())
def test_adaptive_tensors_built_natively_on_device(device):
    built = _build_adaptive_training_tensors(_make_sessions(), max_cards=13, device=device)
    assert built is not None, "builder returned None — no decision rows generated"
    batch, history, memory, target_t, _n_cards_row = built
    want = torch.device(device).type

    for name, t in _named_tensors(batch):
        assert t.device.type == want, f"public-state.{name} on {t.device}, want {device}"
    for name, t in _named_tensors(history):
        assert t.device.type == want, f"history.{name} on {t.device}, want {device}"
    for name, t in _named_tensors(memory):
        assert t.device.type == want, f"memory.{name} on {t.device}, want {device}"
    assert target_t.device.type == want, f"target on {target_t.device}, want {device}"
    # The batch's own device view (self_cards.device) agrees with the request.
    assert batch.device.type == want


# ---------------------------------------------------------------------------
# B. A real Stage5 training micro-step runs entirely on the device AND the
#    firewall still holds there.  On CUDA the forward/backward across the whole
#    nested batch would raise a device-mismatch if anything were left on cpu, so
#    this doubles as the local half of the H100 numerical-correctness gate.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("device", _devices())
def test_stage5_step_runs_on_device_and_keeps_firewall(device):
    torch.manual_seed(0)
    built = _build_adaptive_training_tensors(_make_sessions(), max_cards=13, device=device)
    assert built is not None
    batch, history, memory, target_t, _ = built
    want = torch.device(device).type

    model = GoofspielModel(max_cards=13).to(device)
    model.set_robust_requires_grad(False)

    # The model actually resides on the device (every param and buffer).
    for p in model.parameters():
        assert p.device.type == want, f"a model param on {p.device}, want {device}"
    for b in model.buffers():
        assert b.device.type == want, f"a model buffer on {b.device}, want {device}"

    robust_before = [p.detach().clone() for p in model.robust_parameters()]

    opt = torch.optim.AdamW(model.adaptive_parameters(), lr=1e-3)
    out = model(batch, current_game_history=history, long_term_memory=memory)
    # Forward crossed the whole nested batch with no device mismatch and the
    # logits live on the device — the "no shallow .to(cuda) crutch" proof.
    assert out.opponent_fused_logits.device.type == want
    loss = F.cross_entropy(out.opponent_fused_logits, target_t)
    loss.backward()

    # Firewall, re-run on-device: frozen robust gets no gradient; adaptive does.
    for p in model.robust_parameters():
        assert p.grad is None, "a frozen robust param received gradient on device"
    adaptive_grad = sum(
        float(p.grad.abs().sum()) for p in model.adaptive_parameters() if p.grad is not None
    )
    assert adaptive_grad > 0.0, "adaptive params received no gradient on device"

    opt.step()
    # After a real optimizer step the robust params are byte-unchanged (‖Δθ_R‖==0).
    delta = sum(
        float((a - b).abs().sum()) for a, b in zip(model.robust_parameters(), robust_before)
    )
    assert delta == 0.0, f"robust params moved on device: ‖Δθ_R‖={delta}"


# ---------------------------------------------------------------------------
# C. The real ``run_stage5_adaptive`` dispatch threads the RESOLVED device to the
#    rank0 helper.  The original bug discarded it (``runtime, _ = setup_torch_
#    distributed("auto")``); Phase 2 keeps the resolved value and passes it on.
#    We wrap the real helper to observe the value at its boundary, then delegate
#    to it — the fact under test is the wiring, and the helper (covered by A/B)
#    still runs for real end-to-end.
# ---------------------------------------------------------------------------
def test_run_stage5_adaptive_threads_resolved_device_to_rank0_helper(tmp_path, monkeypatch):
    import goofspiel.training.stages as stages_mod

    captured: dict[str, object] = {}
    real_helper = stages_mod._run_stage5_adaptive_rank0

    def capturing_helper(**kwargs):
        captured["device"] = kwargs["device"]
        return real_helper(**kwargs)

    monkeypatch.setattr(stages_mod, "_run_stage5_adaptive_rank0", capturing_helper)

    # setup_torch_distributed runs for REAL (not mocked); single-process resolves
    # "cpu" -> "cpu"; the identical path resolves "cuda:0" on the H100.
    result = stages_mod.run_stage5_adaptive(steps=1, out_dir=tmp_path, n_cards=3, device="cpu")

    assert captured.get("device") == "cpu", "resolved device was not threaded to the rank0 helper"
    assert result.checkpoint is not None, "real rank0 helper did not complete"
