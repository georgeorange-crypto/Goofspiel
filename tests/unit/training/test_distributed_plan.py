from __future__ import annotations

import pytest

from goofspiel.training.distributed import (
    DistributedTrainingConfig,
    STAGE_SEQUENCE,
    default_h200_roles,
    distributed_manifest,
    torchrun_command,
    validate_stage_sequence,
)


def test_h200_role_plan_uses_all_eight_gpus():
    roles = default_h200_roles(8)
    used = sorted(gpu for role in roles for gpu in role.gpu_ids)
    assert used == list(range(8))


def test_stage_sequence_rejects_reordered_pipeline():
    with pytest.raises(ValueError):
        validate_stage_sequence(["stage4_robust_rl", "stage1_pretrain"])
    validate_stage_sequence(STAGE_SEQUENCE)


def test_torchrun_command_is_generated_not_executed():
    config = DistributedTrainingConfig()
    cmd = torchrun_command(config, stage="stage4_robust_rl", steps=10, batch_size=4)
    assert cmd[0] == "torchrun"
    assert "--device" in cmd
    assert "cuda" in cmd
    assert distributed_manifest(config)["world_size"] == 8
