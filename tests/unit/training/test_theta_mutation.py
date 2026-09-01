"""Priority ② — θ MUTATION is substantive, not cosmetic.

`test_checkpoint_chaining.py` proves two boundary facts: θ_{child,t=0} is
byte-equal to θ_{parent,final} (inheritance), and after real steps *some*
encoder tensor is no longer byte-equal (`any(not torch.equal)`).  That second
assertion is necessary but weak: a single flipped ULP in one tensor passes it.
It cannot tell "the stage genuinely trained from the inherited init" from "the
stage nudged one weight by 1e-9 and is otherwise the frozen parent".

These tests close that gap by QUANTIFYING the mutation and asserting it clears a
substantive floor — recomputing every metric from the loaded bytes of the two
checkpoints (never reading a stored 'delta' field):

  * parameter_l2_delta  — ‖θ_child − θ_parent‖₂ over shared tensors, > ε.
  * changed_parameter_ratio — fraction of individual scalar params that moved by
    more than a tiny tol; must exceed a floor (a real optimizer step touches a
    large fraction of a dense net, not a handful of weights).
  * cosine_similarity   — direction of θ_child vs θ_parent stays high (< a small
    angle): the child is a TRAINED DESCENDANT of the parent, not a random
    reinit that happens to differ.  This is the pair with l2_delta: together
    they say "moved meaningfully, but still recognisably the parent's child".

The contrast anchor is a fresh random model: its l2_delta from the parent is
huge and its cosine is ~0.  The trained child must sit firmly between "frozen"
and "random" — that band is what "trained from the inherited weights" means.
"""

from __future__ import annotations

import pytest

try:
    import torch
except OSError as exc:  # pragma: no cover - machine environment guard
    pytest.skip(f"torch cannot be imported in this environment: {exc}", allow_module_level=True)

from goofspiel.models import GoofspielModel
from goofspiel.training.checkpoint import load_checkpoint
from tests.unit.training.test_checkpoint_chaining import _encoder_state


def _shared_float_keys(a: dict, b: dict) -> list[str]:
    return [
        k
        for k in a
        if k in b
        and torch.is_floating_point(a[k])
        and a[k].shape == b[k].shape
    ]


def _flat(state: dict, keys: list[str]) -> "torch.Tensor":
    return torch.cat([state[k].float().reshape(-1) for k in keys])


def _mutation_metrics(parent: dict, child: dict, keys: list[str]) -> dict:
    """Recompute mutation stats from raw θ — the fact, not a stored field."""
    p = _flat(parent, keys)
    c = _flat(child, keys)
    l2 = (c - p).norm().item()
    # per-scalar change ratio at a tolerance well above fp jitter
    changed = (c - p).abs() > 1e-6
    changed_ratio = changed.float().mean().item()
    cos = torch.nn.functional.cosine_similarity(c, p, dim=0).item()
    return {"l2_delta": l2, "changed_ratio": changed_ratio, "cosine": cos}


def test_p4_is_a_trained_descendant_of_p3_not_a_frozen_copy(tmp_path):
    """P4 must move substantively off the inherited P3 θ — and stay its child.

    Train P4 for several real steps from the P3 init, then measure the mutation
    on the shared encoder and contrast it with a fresh random model.  The child
    has to clear a real floor (it learned) while remaining directionally close to
    the parent (it inherited), whereas the random model is far and uncorrelated.
    """
    from goofspiel.training.stages import (
        run_stage1_pretrain,
        run_stage3_sft,
        run_stage4_robust_rl,
    )

    p1 = run_stage1_pretrain(steps=1, batch_size=4, out_dir=tmp_path / "ck", n_cards=3)
    p3 = run_stage3_sft(
        steps=1, batch_size=8, out_dir=tmp_path / "ck", n_cards=5,
        init_from_checkpoint=p1.checkpoint,
    )
    # Enough steps that a working optimizer visibly moves the dense net.
    p4 = run_stage4_robust_rl(
        steps=6, batch_size=8, out_dir=tmp_path / "ck_p4", n_cards=5,
        init_from_checkpoint=p3.checkpoint,
    )

    p3_state = load_checkpoint(p3.checkpoint)["model_state"]
    p4_state = load_checkpoint(p4.checkpoint)["model_state"]
    enc_keys = list(_encoder_state(p3_state))
    keys = _shared_float_keys(
        {k: p3_state[k] for k in enc_keys}, p4_state
    )
    assert keys, "no shared float encoder tensors to compare"

    m = _mutation_metrics(p3_state, p4_state, keys)

    # (1) Substantive movement: not a single-ULP nudge.
    assert m["l2_delta"] > 1e-4, f"P4 barely moved off P3 (l2={m['l2_delta']:.2e}) — not really trained"
    # (2) A real step touches a broad swath of a dense net, not a few weights.
    assert m["changed_ratio"] > 0.10, (
        f"only {m['changed_ratio']*100:.1f}% of encoder params moved — looks frozen, not trained"
    )
    # (3) Still recognisably P3's child: same direction, small angle.
    assert m["cosine"] > 0.90, f"P4 direction diverged from P3 (cos={m['cosine']:.3f}) — not an inheritance"

    # Contrast anchor: a fresh random model is far and uncorrelated. This is what
    # makes the bounds above meaningful rather than arbitrary.
    fresh = GoofspielModel(max_cards=13).state_dict()
    fresh_keys = _shared_float_keys({k: p3_state[k] for k in enc_keys}, fresh)
    fm = _mutation_metrics(p3_state, fresh, fresh_keys)
    assert fm["l2_delta"] > m["l2_delta"] * 5, (
        f"contrast failed: random l2={fm['l2_delta']:.2e} not >> trained {m['l2_delta']:.2e}"
    )
    assert fm["cosine"] < m["cosine"], (
        f"contrast failed: random cos={fm['cosine']:.3f} not < trained {m['cosine']:.3f}"
    )


def test_more_steps_move_theta_farther(tmp_path):
    """Monotonicity: training longer moves θ farther from the same init.

    A stage that ignored its inherited weights (e.g. re-inited each call) would
    show no such relationship — the distance would be dominated by the reinit,
    not the step count.  Two P4 runs from the SAME P3 init, one short and one
    long, must satisfy l2(long) > l2(short): the optimizer is doing cumulative
    work on top of the inherited θ.
    """
    from goofspiel.training.stages import (
        run_stage1_pretrain,
        run_stage3_sft,
        run_stage4_robust_rl,
    )

    p1 = run_stage1_pretrain(steps=1, batch_size=4, out_dir=tmp_path / "ck", n_cards=3)
    p3 = run_stage3_sft(
        steps=1, batch_size=8, out_dir=tmp_path / "ck", n_cards=5,
        init_from_checkpoint=p1.checkpoint,
    )
    p3_state = load_checkpoint(p3.checkpoint)["model_state"]
    enc_keys = list(_encoder_state(p3_state))

    short = run_stage4_robust_rl(
        steps=2, batch_size=8, out_dir=tmp_path / "short", n_cards=5,
        init_from_checkpoint=p3.checkpoint,
    )
    long = run_stage4_robust_rl(
        steps=10, batch_size=8, out_dir=tmp_path / "long", n_cards=5,
        init_from_checkpoint=p3.checkpoint,
    )
    short_state = load_checkpoint(short.checkpoint)["model_state"]
    long_state = load_checkpoint(long.checkpoint)["model_state"]
    keys = _shared_float_keys({k: p3_state[k] for k in enc_keys}, short_state)

    d_short = _mutation_metrics(p3_state, short_state, keys)["l2_delta"]
    d_long = _mutation_metrics(p3_state, long_state, keys)["l2_delta"]
    assert d_long > d_short, (
        f"longer training did not move θ farther (long={d_long:.2e} <= short={d_short:.2e}) — "
        f"stage may not be accumulating updates on the inherited init"
    )
