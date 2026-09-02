from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

try:
    import torch
except OSError as exc:  # pragma: no cover - environment guard
    pytest.skip(f"torch cannot be imported in this environment: {exc}", allow_module_level=True)

from goofspiel.training.checkpoint import (
    deserialize_python_random_state,
    load_checkpoint,
    serialize_python_random_state,
    state_dict_sha256,
)
from goofspiel.models import GoofspielModel
from goofspiel.training.replay import TrajectoryReplayBuffer
from goofspiel.training.stages import _restore_stage4_resume_state, run_stage4_robust_rl


def _replay_ids(path: Path) -> list[str]:
    return [
        json.loads(line)["sample_id"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_stage4_resume_continues_from_next_step_and_restores_dynamic_state(tmp_path: Path):
    first = run_stage4_robust_rl(
        steps=2,
        batch_size=2,
        out_dir=tmp_path / "run",
        n_cards=2,
        device="cpu",
        seed=17,
        resume_checkpoint_interval=1,
    )
    resume_path = tmp_path / "run" / "stage4_resume_latest.pt"
    assert resume_path.exists()

    resume_payload = load_checkpoint(resume_path)
    extra = resume_payload["extra"]
    assert extra["checkpoint_kind"] == "stage4_periodic_resume"
    assert extra["stage_step_completed"] == 1
    assert extra["next_stage_step"] == 2
    assert extra["total_steps"] == 2
    assert extra["target_model_sha256"] == state_dict_sha256(extra["target_model_state"])
    assert "rng_state_by_rank" in extra
    assert "0" in extra["rng_state_by_rank"]

    replay_snapshot = extra["replay_snapshot"]
    buffer = TrajectoryReplayBuffer(tmp_path / "mirror.jsonl")
    buffer.restore_snapshot(replay_snapshot)
    assert buffer.count() == 4
    assert not buffer.path.exists()
    buffer.rewrite_store_from_memory()
    assert _replay_ids(buffer.path) == _replay_ids(tmp_path / "run" / "replay" / "selfplay_robust.jsonl")

    restored_model = GoofspielModel(max_cards=13)
    restored_opt = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_target = GoofspielModel(max_cards=13)
    rollout_rng = random.Random()
    replay_rng = random.Random()

    class FakeRuntime:
        rank = 0
        local_rank = 0
        world_size = 1

        @property
        def is_rank0(self) -> bool:
            return True

    lineage = _restore_stage4_resume_state(
        checkpoint_path=resume_path,
        model=restored_model,
        optimizer=restored_opt,
        target_model=restored_target,
        replay_buffer=TrajectoryReplayBuffer(tmp_path / "fresh_mirror.jsonl"),
        rollout_rng=rollout_rng,
        replay_rng=replay_rng,
        runtime=FakeRuntime(),
        device="cpu",
        steps=2,
        batch_size=2,
        n_cards=2,
        lr=1e-4,
        seed=17,
    )
    assert lineage["resumed_stage_step_completed"] == 1
    assert lineage["next_stage_step"] == 2
    assert restored_opt.state_dict()["state"] != {}
    assert state_dict_sha256(restored_target.state_dict()) == extra["target_model_sha256"]

    payload_rng = extra["rng_state_by_rank"]["0"]["replay_rng"]
    ref_rng = random.Random()
    ref_rng.setstate(deserialize_python_random_state(payload_rng))
    assert replay_rng.random() == ref_rng.random()
    payload_global = extra["rng_state_by_rank"]["0"]["python_random"]
    ref_global = random.Random()
    ref_global.setstate(deserialize_python_random_state(payload_global))
    assert random.random() == ref_global.random()

    ref_buffer = TrajectoryReplayBuffer(tmp_path / "ref.jsonl")
    ref_buffer.restore_snapshot(replay_snapshot)
    sample_rng = random.Random()
    sample_rng.setstate(deserialize_python_random_state(payload_rng))
    ref_sample_rng = random.Random()
    ref_sample_rng.setstate(deserialize_python_random_state(payload_rng))
    assert [s.sample_id for s in buffer.sample(1, sample_rng)] == [s.sample_id for s in ref_buffer.sample(1, ref_sample_rng)]

    resumed = run_stage4_robust_rl(
        steps=4,
        batch_size=2,
        out_dir=tmp_path / "resume",
        n_cards=2,
        device="cpu",
        seed=17,
        resume_checkpoint=resume_path,
        resume_checkpoint_interval=1,
    )
    assert resumed.metrics["start_stage_step"] == 2.0
    assert resumed.metrics["stage_step_completed"] == 3.0
    assert resumed.metrics["next_stage_step"] == 4.0
    assert resumed.metrics["resumed_stage_step_completed"] == 1.0
    assert resumed.metrics["same_world_size_resume_required"] == 1.0
    assert resumed.checkpoint == str(tmp_path / "resume" / "stage4_robust_rl.pt")


def test_stage4_resume_restore_rejects_world_size_mismatch(tmp_path: Path, monkeypatch):
    first = run_stage4_robust_rl(
        steps=1,
        batch_size=2,
        out_dir=tmp_path / "run",
        n_cards=2,
        device="cpu",
        seed=29,
        resume_checkpoint_interval=1,
    )
    resume_path = tmp_path / "run" / "stage4_resume_latest.pt"
    assert first.checkpoint

    class FakeRuntime:
        rank = 0
        local_rank = 0
        world_size = 2

        @property
        def is_rank0(self) -> bool:
            return True

        @property
        def is_distributed(self) -> bool:
            return False

    from goofspiel.training import stages as stages_mod

    monkeypatch.setattr(stages_mod, "setup_torch_distributed", lambda device="cpu": (FakeRuntime(), device))

    with pytest.raises(RuntimeError, match="same world_size"):
        run_stage4_robust_rl(
            steps=2,
            batch_size=2,
            out_dir=tmp_path / "mismatch",
            n_cards=2,
            device="cpu",
            seed=29,
            resume_checkpoint=resume_path,
            resume_checkpoint_interval=1,
        )


def test_rng_state_helpers_roundtrip_python_and_torch():
    from goofspiel.training.checkpoint import collect_rng_state, restore_rng_state

    random_state = serialize_python_random_state(__import__("random").getstate())
    payload = collect_rng_state(torch_device="cpu")
    payload["python_random"] = random_state
    payload["rollout_rng"] = random_state
    payload["replay_rng"] = random_state
    before = __import__("random").random()
    restore_rng_state(payload, torch_device="cpu")
    after = __import__("random").random()
    restore_rng_state(payload, torch_device="cpu")
    again = __import__("random").random()
    assert after == again
    assert before != again or before == after
