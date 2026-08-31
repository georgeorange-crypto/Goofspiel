"""Resume validation helpers for checkpoint integrity tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from goofspiel.training.checkpoint import load_checkpoint, sha256_file


def validate_checkpoint_resume(path: str | Path) -> dict[str, Any]:
    payload = load_checkpoint(path, verify_checksum=True)
    metadata = payload.get("metadata", {})
    optimizers = payload.get("optimizer_states", {})
    rng_state = payload.get("rng_state", {})
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "training_stage": metadata.get("training_stage"),
        "global_step": metadata.get("global_step"),
        "has_optimizer_state": bool(optimizers),
        "has_rng_state": bool(rng_state),
        "has_model_state": bool(payload.get("model_state")),
        "has_target_model_state": "target_model_state" in payload.get("extra", {}),
        # 3.1b lineage — who this checkpoint descends from.
        "parent_checkpoint_id": metadata.get("parent_checkpoint_id"),
        "init_checkpoint_id": metadata.get("init_checkpoint_id"),
        "model_config_hash": metadata.get("model_config_hash"),
        "optimizer_reset": metadata.get("optimizer_reset"),
    }
