"""Checkpoint save/load helpers with metadata and checksums."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CheckpointMetadata:
    checkpoint_id: str
    training_stage: str
    global_step: int
    policy_version: int
    config: dict[str, Any]
    metrics: dict[str, float] = field(default_factory=dict)
    git_commit: str | None = None
    python_version: str = platform.python_version()
    created_at: float = field(default_factory=time.time)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_rng_state() -> dict[str, Any]:
    state = {
        "python_random": repr(random.getstate()),
        "numpy_random": repr(np.random.get_state()),
    }
    try:
        import torch

        state["torch_cpu_rng"] = torch.get_rng_state().cpu().numpy().tolist()
        if torch.cuda.is_available():
            state["torch_cuda_rng"] = [s.cpu().numpy().tolist() for s in torch.cuda.get_rng_state_all()]
    except Exception as exc:  # pragma: no cover - environment dependent
        state["torch_rng_unavailable"] = repr(exc)
    return state


def save_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizers: dict[str, Any] | None,
    metadata: CheckpointMetadata,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": asdict(metadata),
        "model_state": model.state_dict(),
        "optimizer_states": {
            name: opt.state_dict() for name, opt in (optimizers or {}).items()
        },
        "rng_state": collect_rng_state(),
        "extra": extra or {},
    }
    torch.save(payload, path)
    checksum = sha256_file(path)
    manifest = {"path": str(path), "sha256": checksum, "metadata": asdict(metadata)}
    manifest_path = path.with_suffix(path.suffix + ".sha256.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def load_checkpoint(path: str | Path, *, verify_checksum: bool = True) -> dict[str, Any]:
    import torch

    path = Path(path)
    manifest_path = path.with_suffix(path.suffix + ".sha256.json")
    if verify_checksum and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual = sha256_file(path)
        if manifest.get("sha256") != actual:
            raise RuntimeError(f"checkpoint checksum mismatch for {path}: {actual}")
    return torch.load(path, map_location="cpu")
