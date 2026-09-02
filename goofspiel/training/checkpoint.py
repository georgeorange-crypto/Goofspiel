"""Checkpoint save/load helpers with metadata and checksums."""

from __future__ import annotations

import hashlib
import json
import os
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
    # --- 3.1b lineage contract -------------------------------------------------
    # These make a chained run auditable: which checkpoint's θ this stage was
    # initialised from, which run it resumed, the architecture signature the
    # weights were trained against, and whether the optimizer was reset (a stage
    # transition) or restored (a crash resume).  Lineage should read like
    # ``stage4_robust ⟵ stage3_sft ⟵ stage1_pretrain``, not merely "the file exists".
    parent_checkpoint_id: str | None = None      # the checkpoint this run continued from
    init_checkpoint_id: str | None = None        # θ-only source (init_from_checkpoint)
    parent_checkpoint_sha256: str | None = None  # sha256 of the parent FILE at inherit-time
    model_config_hash: str | None = None         # architecture (named-param shape) signature
    dataset_manifest_ids: list[str] = field(default_factory=list)
    teacher_dataset_ids: list[str] = field(default_factory=list)
    # Structured content provenance for every dataset this stage trained on:
    # each entry is {"path", "sha256", "num_samples", "role"}.  The sha256 is of
    # the dataset FILE's bytes, so a silently-changed dataset is detectable —
    # ``teacher_dataset_ids`` above records only the path, which cannot.
    datasets: list[dict[str, Any]] = field(default_factory=list)
    optimizer_reset: bool = False                # True = fresh optimizer (stage boundary)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_provenance(
    path: str | Path, *, num_samples: int, role: str
) -> dict[str, Any]:
    """Content-addressed provenance for a training dataset file.

    Records the file's byte-level ``sha256`` (not just its path), the sample
    count the stage actually consumed, and the ``role`` the dataset played.  A
    dataset whose contents change between runs produces a different sha256, so a
    later lineage audit can catch "same path, different data" — which a path
    string alone silently hides.
    """
    p = Path(path)
    return {
        "path": str(p),
        "sha256": sha256_file(p) if p.exists() else None,
        "num_samples": int(num_samples),
        "role": role,
    }


def serialize_python_random_state(state: object) -> dict[str, Any]:
    """Serialize ``random.getstate()`` / ``Random.getstate()`` losslessly.

    The native state is a tuple containing another tuple.  Storing it as a
    structured list keeps the checkpoint schema explicit and lets us restore a
    per-rank ``random.Random`` stream without relying on ``repr`` parsing.
    """
    version, internal_state, gauss_next = state  # type: ignore[misc]
    return {
        "version": int(version),
        "internal_state": [int(x) for x in internal_state],
        "gauss_next": gauss_next,
    }


def deserialize_python_random_state(payload: dict[str, Any]) -> object:
    return (
        int(payload["version"]),
        tuple(int(x) for x in payload["internal_state"]),
        payload.get("gauss_next"),
    )


def restore_python_random_state(payload: dict[str, Any], rng: random.Random | None = None) -> None:
    state = deserialize_python_random_state(payload)
    if rng is None:
        random.setstate(state)
    else:
        rng.setstate(state)


def serialize_numpy_random_state(state: object) -> dict[str, Any]:
    name, keys, pos, has_gauss, cached_gaussian = state  # type: ignore[misc]
    return {
        "bit_generator": str(name),
        "keys": [int(x) for x in keys.tolist()],
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def restore_numpy_random_state(payload: dict[str, Any]) -> None:
    np.random.set_state(
        (
            str(payload["bit_generator"]),
            np.asarray(payload["keys"], dtype=np.uint32),
            int(payload["pos"]),
            int(payload["has_gauss"]),
            float(payload["cached_gaussian"]),
        )
    )


def _torch_rng_tensor(values: list[int]) -> Any:
    import torch

    return torch.tensor([int(x) for x in values], dtype=torch.uint8)


def _cuda_device_index(torch_module: Any, torch_device: str | int | None) -> int | None:
    if not torch_module.cuda.is_available():
        return None
    if torch_device is None:
        return int(torch_module.cuda.current_device())
    if isinstance(torch_device, int):
        return int(torch_device)
    if str(torch_device).startswith("cuda"):
        device = torch_module.device(str(torch_device))
        return int(device.index if device.index is not None else torch_module.cuda.current_device())
    return None


def collect_rng_state(*, torch_device: str | int | None = None) -> dict[str, Any]:
    state = {
        "python_random": serialize_python_random_state(random.getstate()),
        "numpy_random": serialize_numpy_random_state(np.random.get_state()),
    }
    try:
        import torch

        state["torch_cpu_rng"] = torch.get_rng_state().cpu().numpy().tolist()
        cuda_device = _cuda_device_index(torch, torch_device)
        if cuda_device is not None:
            state["torch_cuda_device"] = int(cuda_device)
            state["torch_cuda_current_device_rng"] = torch.cuda.get_rng_state(cuda_device).cpu().numpy().tolist()
    except Exception as exc:  # pragma: no cover - environment dependent
        state["torch_rng_unavailable"] = repr(exc)
    return state


def restore_rng_state(state: dict[str, Any], *, torch_device: str | int | None = None) -> None:
    """Restore the process-global Python, NumPy, and Torch RNG streams."""
    if "python_random" in state:
        restore_python_random_state(state["python_random"])
    if "numpy_random" in state:
        restore_numpy_random_state(state["numpy_random"])
    try:
        import torch

        if "torch_cpu_rng" in state:
            torch.set_rng_state(_torch_rng_tensor(state["torch_cpu_rng"]))
        cuda_payload = state.get("torch_cuda_current_device_rng")
        if cuda_payload is not None and torch.cuda.is_available():
            cuda_device = _cuda_device_index(torch, torch_device)
            if cuda_device is None:
                cuda_device = int(state.get("torch_cuda_device", torch.cuda.current_device()))
            torch.cuda.set_rng_state(_torch_rng_tensor(cuda_payload), device=cuda_device)
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"failed to restore torch RNG state: {exc}") from exc


def state_dict_sha256(state_dict: dict[str, Any]) -> str:
    """Content hash for a torch ``state_dict`` independent of object identity."""
    h = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        h.update(str(name).encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("utf-8"))
        h.update(str(tensor.dtype).encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def save_checkpoint(
    path: str | Path,
    *,
    model: Any,
    optimizers: dict[str, Any] | None,
    metadata: CheckpointMetadata,
    extra: dict[str, Any] | None = None,
    atomic: bool = False,
    rng_state: dict[str, Any] | None = None,
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
        "rng_state": rng_state if rng_state is not None else collect_rng_state(),
        "extra": extra or {},
    }
    if atomic:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            with tmp.open("wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
    else:
        torch.save(payload, path)
    checksum = sha256_file(path)
    manifest = {"path": str(path), "sha256": checksum, "metadata": asdict(metadata)}
    manifest_path = path.with_suffix(path.suffix + ".sha256.json")
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False)
    if atomic:
        _write_text_atomic(manifest_path, manifest_text)
    else:
        manifest_path.write_text(manifest_text, encoding="utf-8")
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


def model_config_hash(model: Any) -> str:
    """A stable signature of the architecture (named-param shapes and dtypes).

    Two models with the same signature can exchange ``state_dict``s; a mismatch
    means an ``init_from_checkpoint`` would silently reshape or partially load.
    Used to stamp lineage and to guard weight-inheritance across a stage boundary.
    """
    parts = [f"{name}:{tuple(p.shape)}:{p.dtype}" for name, p in model.state_dict().items()]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def init_from_checkpoint(
    model: Any,
    path: str | Path,
    *,
    strict: bool = True,
    verify_config: bool = True,
) -> dict[str, Any]:
    """**Stage transition** load: copy θ (model weights) ONLY, nothing else.

    This is the P1→P3→P4 seam.  It restores the *parameters* so the next stage
    starts from the previous stage's learned representation, but deliberately
    leaves the optimizer, scheduler, RNG, replay, and global-step at their fresh
    values — the new stage owns a new optimization problem.  It is intentionally
    NOT :func:`resume_checkpoint`: conflating "inherit weights" with "resume a
    crashed run" is the exact research accident this split prevents.

    Returns a small provenance dict (``init_checkpoint_id``, source ``sha256``,
    ``model_config_hash``) to stamp into the new checkpoint's lineage metadata.
    """
    payload = load_checkpoint(path)
    if verify_config:
        want = payload.get("metadata", {}).get("model_config_hash")
        have = model_config_hash(model)
        if want is not None and want != have:
            raise RuntimeError(
                f"init_from_checkpoint architecture mismatch for {path}: "
                f"source model_config_hash={want} != target {have}"
            )
    model.load_state_dict(payload["model_state"], strict=strict)
    meta = payload.get("metadata", {})
    return {
        "init_checkpoint_id": meta.get("checkpoint_id"),
        "init_checkpoint_path": str(path),
        "init_checkpoint_sha256": sha256_file(path),
        "init_global_step": int(meta.get("global_step", 0)),
        "model_config_hash": model_config_hash(model),
    }


def resume_checkpoint(
    model: Any,
    path: str | Path,
    *,
    optimizers: dict[str, Any] | None = None,
    target_model: Any | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """**Crash recovery** load: restore the FULL training state, not just θ.

    Restores model weights, every named optimizer, the target network (if the
    checkpoint carries ``extra['target_model_state']``), and returns the stored
    ``global_step`` / ``rng_state`` / ``extra`` so the caller can continue the
    *same* run where it stopped.  This is the counterpart to
    :func:`init_from_checkpoint`; the two are never conflated.
    """
    payload = load_checkpoint(path)
    model.load_state_dict(payload["model_state"], strict=strict)
    restored_opt = []
    if optimizers:
        stored = payload.get("optimizer_states", {})
        for name, opt in optimizers.items():
            if name in stored:
                opt.load_state_dict(stored[name])
                restored_opt.append(name)
    extra = payload.get("extra", {})
    if target_model is not None and "target_model_state" in extra:
        target_model.load_state_dict(extra["target_model_state"], strict=strict)
    meta = payload.get("metadata", {})
    return {
        "parent_checkpoint_id": meta.get("checkpoint_id"),
        "resume_checkpoint_path": str(path),
        "resume_sha256": sha256_file(path),
        "global_step": int(meta.get("global_step", 0)),
        "restored_optimizers": restored_opt,
        "restored_target_model": target_model is not None and "target_model_state" in extra,
        "rng_state": payload.get("rng_state", {}),
        "extra": extra,
    }
