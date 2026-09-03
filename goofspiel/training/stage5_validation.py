"""Held-out evaluation of the Stage5 opponent/adaptive branch.

Phase 3B needs a validation signal that is genuinely *held out*: computed on data
the optimizer never touched, in eval mode, under ``no_grad``, with no parameter or
optimizer-state mutation and -- critically -- with **no observable effect on the
training run** whether or not it is called.  :func:`evaluate_opponent_model`
snapshots and restores the torch RNG around the forward pass so interleaving
validation between optimizer updates cannot perturb the training RNG stream, which
keeps a legacy (version-1) run byte-identical and a version-2 run reproducible.

This module deliberately owns no data generation and no training; it consumes the
sessions produced by :mod:`goofspiel.training.stage5_data` and reuses the exact
featurisation the trainer uses (imported lazily to avoid an import cycle with
``stages``).
"""

from __future__ import annotations

import hashlib
from typing import Sequence

from goofspiel.training.data import OpponentSession


def evaluate_opponent_model(
    model,
    sessions: Sequence[OpponentSession],
    *,
    max_cards: int,
    device: str,
) -> dict[str, float]:
    """Evaluate the opponent head on held-out ``sessions`` without side effects.

    Returns NLL, a uniform-random reference NLL and the gain over it, accuracy,
    expected calibration error and Brier score, plus the evaluated row count.
    The model is put in eval mode for the forward pass and restored to its prior
    train/eval mode afterwards; no gradient is taken, no optimizer step happens,
    and the global torch RNG state is saved and restored so calling this function
    is invisible to a concurrently-progressing training loop.
    """
    # Lazy import breaks the stages <-> stage5_validation cycle: by the time this
    # runs, ``stages`` is fully imported.  Reusing the trainer's own featuriser
    # and ECE guarantees train/val are measured on identical machinery.
    from goofspiel.training.stages import (
        _build_adaptive_training_tensors,
        _expected_calibration_error,
        _torch_import,
    )

    torch, F = _torch_import()

    tensors = _build_adaptive_training_tensors(list(sessions), max_cards=max_cards, device=device)
    if tensors is None:
        raise RuntimeError("Stage5 validation produced no rows from the held-out sessions")
    batch, history, memory, target_t, _n_cards_row = tensors

    legal_counts = batch.opponent_action_mask.sum(dim=-1).clamp_min(1).float()
    uniform_reference_nll = float(torch.log(legal_counts).mean())

    was_training = model.training
    rng_state = torch.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        model.eval()
        with torch.no_grad():
            out_model = model(batch, current_game_history=history, long_term_memory=memory)
            logits = out_model.opponent_fused_logits
            nll = float(F.cross_entropy(logits, target_t))
            probs = F.softmax(logits.masked_fill(~batch.opponent_action_mask, -1e9), dim=-1)
            acc = float((probs.argmax(dim=-1) == target_t).float().mean())
            ece = float(_expected_calibration_error(probs, target_t))
            onehot = F.one_hot(target_t, num_classes=probs.shape[-1]).to(probs.dtype)
            brier = float(((probs - onehot) ** 2).sum(dim=-1).mean())
    finally:
        # Restore train/eval mode and the RNG streams exactly as they were, so a
        # mid-training validation call leaves the optimizer's RNG untouched.
        if was_training:
            model.train()
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)

    return {
        "val_nll": nll,
        "val_uniform_reference_nll": uniform_reference_nll,
        "val_nll_gain": uniform_reference_nll - nll,
        "val_accuracy": acc,
        "val_ece": ece,
        "val_brier": brier,
        "val_rows": float(int(target_t.shape[0])),
    }


def model_param_signature(model) -> str:
    """A content hash over the model's full ``state_dict`` (params + buffers).

    Used by the tests to prove that a validation pass mutates nothing: the
    signature is identical before and after :func:`evaluate_opponent_model`.
    """
    from goofspiel.training.stages import _torch_import

    torch, _ = _torch_import()
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        arr = tensor.detach().cpu()
        if arr.dtype in (torch.bfloat16, torch.float16):
            arr = arr.float()
        h.update(arr.contiguous().numpy().tobytes())
    return h.hexdigest()
