from __future__ import annotations

from pathlib import Path

import pytest


def test_resume_validator_checks_required_checkpoint_fields(tmp_path):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"torch cannot be imported in this environment: {exc}")
    from goofspiel.models import GoofspielModel
    from goofspiel.training.checkpoint import CheckpointMetadata, save_checkpoint
    from goofspiel.training.resume import validate_checkpoint_resume

    model = GoofspielModel()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(
        path,
        model=model,
        optimizers={"adamw": opt},
        metadata=CheckpointMetadata("test", "P4_ROBUST_RL", 100, 1, {}),
        extra={"target_model_state": model.state_dict()},
    )
    report = validate_checkpoint_resume(path)
    assert report["has_optimizer_state"]
    assert report["has_rng_state"]
    assert report["has_target_model_state"]


def test_checkpoint_registry_tracks_required_kinds(tmp_path):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"torch cannot be imported in this environment: {exc}")
    from goofspiel.models import GoofspielModel
    from goofspiel.training.checkpoint import CheckpointMetadata, save_checkpoint
    from goofspiel.training.checkpoint_registry import CHECKPOINT_KINDS, CheckpointRegistry

    model = GoofspielModel()
    path = tmp_path / "source.pt"
    save_checkpoint(
        path,
        model=model,
        optimizers={},
        metadata=CheckpointMetadata("source", "P4_ROBUST_RL", 1, 1, {}),
    )
    registry = CheckpointRegistry(tmp_path / "registry")
    for kind in CHECKPOINT_KINDS:
        entry = registry.register(kind, path, global_step=1, metrics={"score": 0.0})
        assert Path(entry.registry_path).exists()
    assert registry.missing_kinds() == []
