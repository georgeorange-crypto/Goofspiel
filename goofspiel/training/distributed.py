"""Distributed training launch configuration for production servers."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

STAGE_SEQUENCE = [
    "stage0_verify",
    "build_corpus",
    "stage1_pretrain",
    "stage2_semi_supervised",
    "stage3_sft",
    "stage4_robust_rl",
    "stage5_adaptive",
    "stage6_league",
    "stage7_redteam",
    "evaluate",
    "smoke_pipeline",
]


@dataclass(frozen=True)
class WorkerRole:
    name: str
    gpu_ids: tuple[int, ...]
    purpose: str


@dataclass
class DistributedTrainingConfig:
    num_nodes: int = 1
    gpus_per_node: int = 8
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    artifact_dir: str = "artifacts/runs/h200_full"
    roles: list[WorkerRole] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.roles:
            self.roles = default_h200_roles(self.gpus_per_node)

    @property
    def world_size(self) -> int:
        return int(self.num_nodes) * int(self.gpus_per_node)


def default_h200_roles(gpus_per_node: int = 8) -> list[WorkerRole]:
    ids = tuple(range(gpus_per_node))
    if gpus_per_node >= 8:
        return [
            WorkerRole("learner", (0, 1, 2, 3), "main neural learner and optimizer"),
            WorkerRole("actors", (4, 5), "self-play actors and trajectory generation"),
            WorkerRole("search", (6,), "batched matrix/search inference worker"),
            WorkerRole("evaluator", (7,), "evaluation, league manager, red-team relabel"),
        ]
    return [WorkerRole("single_node_all_roles", ids, "compact development or smaller server run")]


def validate_stage_sequence(stages: list[str]) -> None:
    allowed = set(STAGE_SEQUENCE)
    unknown = [stage for stage in stages if stage not in allowed]
    if unknown:
        raise ValueError(f"unknown stages: {unknown}")
    order = {stage: i for i, stage in enumerate(STAGE_SEQUENCE)}
    if [order[s] for s in stages] != sorted(order[s] for s in stages):
        raise ValueError(f"stages must preserve frozen order: {STAGE_SEQUENCE}")


def torchrun_command(
    config: DistributedTrainingConfig,
    *,
    stage: str,
    steps: int,
    batch_size: int,
    n_cards: int = 13,
) -> list[str]:
    validate_stage_sequence([stage])
    return [
        "torchrun",
        "--nnodes",
        str(config.num_nodes),
        "--nproc_per_node",
        str(config.gpus_per_node),
        "--master_addr",
        config.master_addr,
        "--master_port",
        str(config.master_port),
        str(Path("scripts") / "train_goofspiel_full.py"),
        "--artifact-dir",
        config.artifact_dir,
        "--stage",
        stage,
        "--steps",
        str(steps),
        "--batch-size",
        str(batch_size),
        "--n-cards",
        str(n_cards),
        "--device",
        "cuda",
    ]


def distributed_manifest(config: DistributedTrainingConfig) -> dict[str, Any]:
    return {
        "num_nodes": config.num_nodes,
        "gpus_per_node": config.gpus_per_node,
        "world_size": config.world_size,
        "master_addr": config.master_addr,
        "master_port": config.master_port,
        "artifact_dir": config.artifact_dir,
        "roles": [{"name": r.name, "gpu_ids": list(r.gpu_ids), "purpose": r.purpose} for r in config.roles],
        "stage_sequence": STAGE_SEQUENCE,
    }


@dataclass(frozen=True)
class DistributedRuntime:
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


def current_runtime() -> DistributedRuntime:
    return DistributedRuntime(
        rank=int(os.environ.get("RANK", "0")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
    )


def setup_torch_distributed(device: str = "cpu") -> tuple[DistributedRuntime, str]:
    """Initialize torch.distributed when launched under torchrun."""
    runtime = current_runtime()
    resolved_device = device
    if runtime.is_distributed:
        import torch
        import torch.distributed as dist

        if device.startswith("cuda"):
            torch.cuda.set_device(runtime.local_rank)
            resolved_device = f"cuda:{runtime.local_rank}"
        if not dist.is_initialized():
            backend = "nccl" if resolved_device.startswith("cuda") else "gloo"
            dist.init_process_group(backend=backend)
    return runtime, resolved_device


def barrier_if_distributed() -> None:
    runtime = current_runtime()
    if runtime.is_distributed:
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()
