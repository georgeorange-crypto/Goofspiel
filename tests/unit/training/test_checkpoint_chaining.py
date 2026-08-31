"""Phase 3.1 — P1 → P3 → P4 checkpoint chaining via two DISTINCT interfaces.

Before this phase every stage random-initialised and the orchestration layer had
no seam to feed weights forward: `stage1.checkpoint` / `stage3.checkpoint` were
captured only to write into the summary JSON, never loaded.  The final
`stage4_robust_rl.pt` therefore had no causal link to P1/P3 at all.

These tests assert the fix at the level of *bytes and behaviour*, per the project
testing principle (re-execute the fact, don't read a field):

  1. The two interfaces are genuinely different: `init_from_checkpoint` copies θ
     ONLY (fresh optimizer, step 0); `resume_checkpoint` restores the full
     training state (optimizer + target network + global_step).  Conflating them
     is the exact accident the split prevents.
  2. **θ_{P3,t=0} == θ_{P1,final}** and **θ_{P4,t=0} == θ_{P3,final}**, byte-equal
     on the shared encoder — the single assertion that would have caught the
     original disconnect immediately.  Re-run with steps>0 and θ must then move,
     proving the stage trains *from* the inherited weights rather than ignoring
     them.
  3. Lineage (3.1b) is recorded: a chained P3 checkpoint names P1 as its
     `init_checkpoint_id` / `parent_checkpoint_id`, with `optimizer_reset=True`.
"""

from __future__ import annotations

import pytest

try:
    import torch
except OSError as exc:  # pragma: no cover - machine environment guard
    pytest.skip(f"torch cannot be imported in this environment: {exc}", allow_module_level=True)

from goofspiel.models import GoofspielModel
from goofspiel.training.checkpoint import (
    CheckpointMetadata,
    init_from_checkpoint,
    load_checkpoint,
    model_config_hash,
    resume_checkpoint,
    save_checkpoint,
)

# The shared robust encoder/backbone whose weights must survive a stage
# transition (the adaptive branch and heads may diverge, but this must chain).
SHARED_ENCODER_PREFIXES = (
    "rank_encoder.",
    "global_encoder.",
    "card_projector.",
    "card_transformer.",
    "role_embed.",
    "node_projector.",
    "gnn_layers.",
)


def _encoder_state(state: dict) -> dict:
    return {
        k: v for k, v in state.items() if any(k.startswith(p) for p in SHARED_ENCODER_PREFIXES)
    }


def _assert_state_equal(a: dict, b: dict, keys: dict) -> None:
    assert set(a) >= set(keys) and set(b) >= set(keys)
    for k in keys:
        assert torch.equal(a[k], b[k]), f"tensor {k} differs across the stage boundary"


# ----------------------------------------------------------------------------
# 1. The two interfaces are different operations.
# ----------------------------------------------------------------------------
def test_init_copies_theta_only_but_resume_restores_optimizer(tmp_path):
    src = GoofspielModel(max_cards=13)
    opt = torch.optim.AdamW(src.parameters(), lr=1e-3)
    # Take one real optimizer step so the optimizer carries non-trivial state.
    loss = sum(p.float().pow(2).sum() for p in src.parameters())
    loss.backward()
    opt.step()
    path = tmp_path / "src.pt"
    save_checkpoint(
        path,
        model=src,
        optimizers={"strategic_sft": opt},
        metadata=CheckpointMetadata(
            "stage1_pretrain", "P1_PRETRAIN", 7, 0, {},
            model_config_hash=model_config_hash(src),
        ),
    )

    # init_from_checkpoint: θ copied, optimizer NOT touched (stays empty/step-0).
    m_init = GoofspielModel(max_cards=13)
    opt_init = torch.optim.AdamW(m_init.parameters(), lr=1e-3)
    prov = init_from_checkpoint(m_init, path)
    _assert_state_equal(m_init.state_dict(), src.state_dict(), src.state_dict())
    assert opt_init.state_dict()["state"] == {}, "init must NOT restore optimizer state"
    assert prov["init_checkpoint_id"] == "stage1_pretrain"

    # resume_checkpoint: θ AND optimizer restored (state non-empty, step preserved).
    m_res = GoofspielModel(max_cards=13)
    opt_res = torch.optim.AdamW(m_res.parameters(), lr=1e-3)
    info = resume_checkpoint(m_res, path, optimizers={"strategic_sft": opt_res})
    _assert_state_equal(m_res.state_dict(), src.state_dict(), src.state_dict())
    assert opt_res.state_dict()["state"] != {}, "resume must restore optimizer state"
    assert info["global_step"] == 7
    assert "strategic_sft" in info["restored_optimizers"]


def test_init_and_resume_are_mutually_exclusive():
    from goofspiel.training.stages import _apply_init_or_resume

    m = GoofspielModel(max_cards=13)
    with pytest.raises(ValueError, match="mutually exclusive"):
        _apply_init_or_resume(
            m, init_from_checkpoint_path="a.pt", resume_checkpoint_path="b.pt"
        )


def test_config_hash_guards_architecture_mismatch(tmp_path):
    src = GoofspielModel(max_cards=13)
    path = tmp_path / "src.pt"
    save_checkpoint(
        path, model=src, optimizers={},
        metadata=CheckpointMetadata(
            "s", "P1_PRETRAIN", 1, 0, {}, model_config_hash="deadbeef-not-a-match"
        ),
    )
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        init_from_checkpoint(GoofspielModel(max_cards=13), path)


# ----------------------------------------------------------------------------
# 2. θ_{P3,t=0} == θ_{P1,final} and θ_{P4,t=0} == θ_{P3,final} (byte-equal).
# ----------------------------------------------------------------------------
def test_p3_inherits_p1_weights_then_trains_from_them(tmp_path):
    from goofspiel.training.stages import run_stage1_pretrain, run_stage3_sft

    p1 = run_stage1_pretrain(steps=1, batch_size=4, out_dir=tmp_path / "ck", n_cards=3)
    # steps=0 P3: no update, so the saved θ is exactly θ_{P3,t=0} = the inherited θ.
    p3_t0 = run_stage3_sft(
        steps=0, batch_size=4, out_dir=tmp_path / "ck_t0", n_cards=3,
        init_from_checkpoint=p1.checkpoint,
    )
    p1_state = load_checkpoint(p1.checkpoint)["model_state"]
    p3_t0_state = load_checkpoint(p3_t0.checkpoint)["model_state"]
    # θ_{P3,t=0} == θ_{P1,final} on the shared encoder, byte-equal.
    _assert_state_equal(p3_t0_state, p1_state, _encoder_state(p1_state))

    # And with real steps the encoder MOVES — P3 trains from the inherited init,
    # it does not merely copy and freeze.
    p3_trained = run_stage3_sft(
        steps=2, batch_size=8, out_dir=tmp_path / "ck_tr", n_cards=5,
        init_from_checkpoint=p1.checkpoint,
    )
    trained_state = load_checkpoint(p3_trained.checkpoint)["model_state"]
    enc = _encoder_state(p1_state)
    assert any(not torch.equal(trained_state[k], p1_state[k]) for k in enc), (
        "P3 encoder did not move — it is not actually training from the inherited θ"
    )


def test_p4_inherits_p3_weights(tmp_path):
    from goofspiel.training.stages import run_stage1_pretrain, run_stage3_sft, run_stage4_robust_rl

    p1 = run_stage1_pretrain(steps=1, batch_size=4, out_dir=tmp_path / "ck", n_cards=3)
    p3 = run_stage3_sft(
        steps=1, batch_size=8, out_dir=tmp_path / "ck", n_cards=5,
        init_from_checkpoint=p1.checkpoint,
    )
    # steps=0 P4: θ_{P4,t=0} == θ_{P3,final}.
    p4_t0 = run_stage4_robust_rl(
        steps=0, batch_size=4, out_dir=tmp_path / "ck_p4", n_cards=3,
        init_from_checkpoint=p3.checkpoint,
    )
    p3_state = load_checkpoint(p3.checkpoint)["model_state"]
    p4_state = load_checkpoint(p4_t0.checkpoint)["model_state"]
    _assert_state_equal(p4_state, p3_state, _encoder_state(p3_state))
    # The P4 target network is seeded from the inherited (P3) weights too.
    target_state = load_checkpoint(p4_t0.checkpoint)["extra"]["target_model_state"]
    _assert_state_equal(target_state, p3_state, _encoder_state(p3_state))


# ----------------------------------------------------------------------------
# 3. Lineage metadata (3.1b).
# ----------------------------------------------------------------------------
def test_chained_checkpoint_records_lineage(tmp_path):
    from goofspiel.training.resume import validate_checkpoint_resume
    from goofspiel.training.stages import run_stage1_pretrain, run_stage3_sft

    p1 = run_stage1_pretrain(steps=1, batch_size=4, out_dir=tmp_path / "ck", n_cards=3)
    p3 = run_stage3_sft(
        steps=1, batch_size=8, out_dir=tmp_path / "ck", n_cards=5,
        init_from_checkpoint=p1.checkpoint,
    )
    report = validate_checkpoint_resume(p3.checkpoint)
    assert report["init_checkpoint_id"] == "stage1_pretrain"
    assert report["parent_checkpoint_id"] == "stage1_pretrain"
    assert report["optimizer_reset"] is True  # a stage boundary resets the optimizer
    # Same architecture => same config hash across the boundary.
    p1_hash = load_checkpoint(p1.checkpoint)["metadata"]["model_config_hash"]
    assert report["model_config_hash"] == p1_hash
