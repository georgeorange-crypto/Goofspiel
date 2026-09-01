from __future__ import annotations

from pathlib import Path

import json

import pytest

from goofspiel.game import GameState
from goofspiel.training.data import RobustTrajectorySample, RoundRecord, state_record_from_game_state
from goofspiel.training.distributed import derive_rank_seed
from goofspiel.training.stages import (
    _rank_shard_range,
    _rollout_selfplay_game,
    _stable_trajectory_prefix,
    _trajectory_hash,
    run_stage4_robust_rl,
    run_stage5_adaptive,
)

try:
    import torch
except OSError as exc:  # pragma: no cover - machine environment guard
    pytest.skip(f"torch cannot be imported in this environment: {exc}", allow_module_level=True)

from goofspiel.models import GoofspielModel


def _sample(sample_id: str, *, score: int = 0, n: int = 2) -> RobustTrajectorySample:
    state = GameState.initial(n, current_prize=1)
    return RobustTrajectorySample(
        sample_id=sample_id,
        states=[state_record_from_game_state(state)],
        rounds=[
            RoundRecord(
                round_index=state.round_index,
                prize=state.current_prize,
                self_action=1,
                opponent_action=2 if n >= 2 else 1,
                reward_self=score,
                reward_opponent=-score,
                done=True,
            )
        ],
        behavior_policy_self=[[1.0, 0.0] + [0.0] * 11],
        behavior_policy_opponent=[[0.0, 1.0] + [0.0] * 11],
        action_prob_self=[1.0],
        action_prob_opponent=[1.0],
        final_score_diff=score,
        model_version="test",
        opponent_version="test",
        n=n,
    )


def test_stage4_replay_fallback_global_batch_is_world_size_invariant():
    assert sum(_rank_shard_range(5, rank, 1)[1] for rank in range(1)) == 5
    assert sum(_rank_shard_range(5, rank, 2)[1] for rank in range(2)) == 5
    assert [_rank_shard_range(5, rank, 2)[1] for rank in range(2)] == [3, 2]
    rank0 = [_sample("a", score=1), _sample("b", score=2)]
    rank1 = [_sample("c", score=3), _sample("d", score=4)]
    one_rank = _stable_trajectory_prefix(rank0, 2)
    two_rank = _stable_trajectory_prefix(rank0 + rank1, 2)
    assert len(one_rank) == 2
    assert len(two_rank) == 2
    assert {t.sample_id for t in one_rank} <= {"a", "b"}
    assert {t.sample_id for t in two_rank} <= {"a", "b", "c", "d"}
    assert len({_trajectory_hash(t) for t in two_rank}) == 2


def test_trajectory_hash_ignores_sample_id_for_rng_deduplication():
    left = _sample("rank0:local-id", score=1)
    right = _sample("rank1:different-id", score=1)
    assert _trajectory_hash(left) == _trajectory_hash(right)


def test_stage4_source_does_not_reintroduce_chosen_q_pg_baseline():
    source = Path("goofspiel/training/stages.py").read_text(encoding="utf-8")
    stage4_source = source.split("def run_stage4_robust_rl(", 1)[1].split(
        "def _opponent_regime_distribution(",
        1,
    )[0]
    assert "baseline = chosen_q.detach()" not in stage4_source
    assert "pg_loss" not in stage4_source
    assert "logp * (returns_t - baseline)" not in stage4_source


def test_stage4_rank_rng_decorrelates_trajectory_hashes():
    torch.manual_seed(0)
    model = GoofspielModel(max_cards=13)
    rank0_rng = __import__("random").Random(derive_rank_seed(91, 0))
    rank1_rng = __import__("random").Random(derive_rank_seed(91, 1))
    t0 = _rollout_selfplay_game(
        model,
        n_cards=3,
        rng=rank0_rng,
        device="cpu",
        model_version="r0",
        game_index=0,
        sample_id="rank0",
    )
    t1 = _rollout_selfplay_game(
        model,
        n_cards=3,
        rng=rank1_rng,
        device="cpu",
        model_version="r1",
        game_index=0,
        sample_id="rank1",
    )
    assert _trajectory_hash(t0) != _trajectory_hash(t1)
    rank0_rng_replay = __import__("random").Random(derive_rank_seed(91, 0))
    t0_replay = _rollout_selfplay_game(
        model,
        n_cards=3,
        rng=rank0_rng_replay,
        device="cpu",
        model_version="r0",
        game_index=0,
        sample_id="rank0",
    )
    assert _trajectory_hash(t0) == _trajectory_hash(t0_replay)


def test_stage4_reports_neurd_stability_metrics(tmp_path: Path):
    result = run_stage4_robust_rl(
        steps=1,
        batch_size=2,
        out_dir=tmp_path / "stage4",
        n_cards=2,
        device="cpu",
        seed=11,
    )
    metrics = result.metrics
    for key in [
        "q_loss_last",
        "actor_loss_last",
        "entropy_last",
        "min_logit_last",
        "max_logit_last",
        "logit_gap_last",
        "regret_scale_last",
        "grad_norm_last",
    ]:
        assert key in metrics
        assert metrics[key] == pytest.approx(float(metrics[key]))
    assert metrics["policy_gradient_removed"] == 1.0
    assert metrics["neurd_logit_threshold"] == 2.0
    assert metrics["global_training_batch_trajectories"] == 2.0
    assert metrics["unique_training_trajectory_hashes"] == 2.0


def test_stage4_same_seed_replays_same_trajectory_hashes(tmp_path: Path):
    first = run_stage4_robust_rl(
        steps=2,
        batch_size=2,
        out_dir=tmp_path / "a",
        n_cards=2,
        device="cpu",
        seed=19,
    )
    second = run_stage4_robust_rl(
        steps=2,
        batch_size=2,
        out_dir=tmp_path / "b",
        n_cards=2,
        device="cpu",
        seed=19,
    )
    assert first.metrics["replay_samples"] == second.metrics["replay_samples"] == 4.0

    def replay_ids(root: Path) -> list[str]:
        rows = []
        for line in (root / "replay" / "selfplay_robust.jsonl").read_text(encoding="utf-8").splitlines():
            rows.append(json.loads(line)["sample_id"])
        return rows

    assert replay_ids(tmp_path / "a") == replay_ids(tmp_path / "b")


def test_stage5_nonzero_rank_does_not_train_or_write(monkeypatch, tmp_path: Path):
    from goofspiel.training import stages as stages_mod

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")

    def fake_setup(device="cpu"):
        runtime = stages_mod.current_runtime()
        return runtime, device

    def fake_broadcast(obj, *, src=0):
        assert obj is None
        return {
            "checkpoint": str(tmp_path / "stage5_adaptive.pt"),
            "metrics": {"stage5_rank_owner": 0.0, "stage5_write_once": 1.0},
        }

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("non-rank0 must not enter P5 training/write path")

    monkeypatch.setattr(stages_mod, "setup_torch_distributed", fake_setup)
    monkeypatch.setattr(stages_mod, "broadcast_object", fake_broadcast)
    monkeypatch.setattr(stages_mod, "barrier_if_distributed", lambda: None)
    monkeypatch.setattr(stages_mod, "JsonlStore", fail_if_called)

    result = run_stage5_adaptive(steps=2, out_dir=tmp_path, n_cards=3)
    assert result.checkpoint == str(tmp_path / "stage5_adaptive.pt")
    assert result.metrics["stage5_write_once"] == 1.0
    assert not (tmp_path / "adaptive" / "opponent_sessions.jsonl").exists()
