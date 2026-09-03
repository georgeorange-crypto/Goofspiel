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
    from goofspiel.training.stage_control import (
        STATE_SUCCESS,
        Stage5Status,
        control_dir_for,
        current_invocation_id,
        write_status,
    )

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")
    # Both "ranks" of a launch share this (as torchrun sets it); the pre-seed
    # below stamps the SAME invocation id the non-rank0 branch will expect, so
    # the record is accepted as belonging to THIS invocation (not a stale one).
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "unit-stage5-nonzero")

    def fake_setup(device="cpu"):
        # Real rank1 runtime, but skip live process-group init — a genuine
        # 2-process PG is exercised by tests/unit/training/test_stage5_control_plane.py,
        # not this single-process unit test.
        runtime = stages_mod.current_runtime()
        return runtime, device

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("non-rank0 must not enter P5 training/write path")

    # Pre-seed the control plane as if rank0 had already finished.  The non-rank0
    # branch must read this through the REAL wait_for_rank0 / read_status control
    # plane (NOT a mock) and return the carried result without training/writing.
    control_dir = control_dir_for(tmp_path)
    control_dir.mkdir(parents=True, exist_ok=True)
    write_status(
        control_dir,
        Stage5Status(
            state=STATE_SUCCESS,
            stage_invocation_id=current_invocation_id(),
            checkpoint=str(tmp_path / "stage5_adaptive.pt"),
            metrics={"stage5_rank_owner": 0.0, "stage5_write_once": 1.0},
            updated_at=1.0,
        ),
    )

    # Only the process-group primitives are stubbed (no live PG in a unit test);
    # the control-plane read path itself runs for real.
    monkeypatch.setattr(stages_mod, "setup_torch_distributed", fake_setup)
    monkeypatch.setattr(stages_mod, "barrier_if_distributed", lambda: None)
    monkeypatch.setattr(stages_mod, "JsonlStore", fail_if_called)

    result = run_stage5_adaptive(steps=2, out_dir=tmp_path, n_cards=3)
    assert result.checkpoint == str(tmp_path / "stage5_adaptive.pt")
    assert result.metrics["stage5_write_once"] == 1.0
    assert not (tmp_path / "adaptive" / "opponent_sessions.jsonl").exists()


def test_stage5_fresh_run_rebuilds_opponent_sessions_jsonl(tmp_path: Path):
    from goofspiel.training.stages import run_stage5_adaptive

    run_stage5_adaptive(steps=2, out_dir=tmp_path, n_cards=3, seed=19)
    run_stage5_adaptive(steps=1, out_dir=tmp_path, n_cards=3, seed=23)

    rows = [
        json.loads(line)
        for line in (tmp_path / "adaptive" / "opponent_sessions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["session_id"] == "adaptive_session_0:seed530:n3"


def test_stage2_rebuilds_dataset_and_reports_nonzero_sources(tmp_path: Path):
    from goofspiel.training.stages import run_stage2_semi_supervised

    out = tmp_path / "data"
    stale = out / "teacher_dataset.jsonl"
    out.mkdir(parents=True)
    stale.write_text('{"schema_version":"goofspiel.training.v1","stale":true}\n', encoding="utf-8")

    result = run_stage2_semi_supervised(steps=1, out_dir=out, n_cards=3)
    rows = [json.loads(line) for line in stale.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    assert all("teacher_source" in row for row in rows)
    nonzero = sum(
        1
        for key, value in result.metrics.items()
        if key.startswith("teacher_source_") and key.endswith("_samples") and value > 0
    )
    assert result.metrics["distinct_teacher_sources"] == float(nonzero)
    assert result.metrics["teacher_dataset_rows"] == float(len(rows))
    assert result.metrics["seed"] == 1.0


def test_stage2_nonzero_rank_does_not_write_dataset(monkeypatch, tmp_path: Path):
    from goofspiel.training import stages as stages_mod
    from goofspiel.training.stages import run_stage2_semi_supervised

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")

    def fake_setup(device="auto"):
        runtime = stages_mod.current_runtime()
        return runtime, device

    def fake_broadcast(obj, *, src=0):
        assert obj is None
        return {"metrics": {"stage2_rank_owner": 0.0, "stage2_write_once": 1.0}}

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("non-rank0 must not enter P2 write path")

    monkeypatch.setattr(stages_mod, "setup_torch_distributed", fake_setup)
    monkeypatch.setattr(stages_mod, "broadcast_object", fake_broadcast)
    monkeypatch.setattr(stages_mod, "barrier_if_distributed", lambda: None)
    monkeypatch.setattr(stages_mod, "JsonlStore", fail_if_called)

    result = run_stage2_semi_supervised(steps=2, out_dir=tmp_path / "data", n_cards=3)
    assert result.metrics["stage2_write_once"] == 1.0
    assert not (tmp_path / "data" / "teacher_dataset.jsonl").exists()


def test_stage1_stage3_data_shards_depend_on_rank(monkeypatch):
    from goofspiel.training import stages as stages_mod

    rt0 = type("Runtime", (), {"rank": 0, "world_size": 2})()
    rt1 = type("Runtime", (), {"rank": 1, "world_size": 2})()
    corpus_states = [
        state_record_from_game_state(GameState.initial(3, current_prize=(i % 3) + 1))
        for i in range(6)
    ]

    assert stages_mod._ranked_step(3, rt0) == 6
    assert stages_mod._ranked_step(3, rt1) == 7
    shard0 = stages_mod._corpus_batch_for_rank(corpus_states, step=0, batch_size=2, runtime=rt0)
    shard1 = stages_mod._corpus_batch_for_rank(corpus_states, step=0, batch_size=2, runtime=rt1)
    assert [s.state_hash for s in shard0] != [s.state_hash for s in shard1]


def test_stage6_stage7_eval_axis_nonzero_rank_do_not_write(monkeypatch, tmp_path: Path):
    from goofspiel.training import stages as stages_mod
    from goofspiel.training.stages import (
        run_axis_promotion_selection,
        run_evaluation_suite,
        run_stage6_league,
        run_stage7_redteam,
    )

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")

    def fake_setup(device="auto"):
        runtime = stages_mod.current_runtime()
        return runtime, device

    payloads = iter([
        {"metrics": {"stage6_write_once": 1.0}, "checkpoint": None},
        {"metrics": {"stage7_write_once": 1.0}, "checkpoint": str(tmp_path / "redteam" / "stage7_corrected.pt")},
        {"rank_owner": 0, "write_once": True, "reports": []},
        {"rank_owner": 0, "write_once": True, "selected": {}, "table": {}},
    ])

    def fake_broadcast(obj, *, src=0):
        assert obj is None
        return next(payloads)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("non-rank0 must not enter write/training path")

    monkeypatch.setattr(stages_mod, "setup_torch_distributed", fake_setup)
    monkeypatch.setattr(stages_mod, "broadcast_object", fake_broadcast)
    monkeypatch.setattr(stages_mod, "barrier_if_distributed", lambda: None)
    monkeypatch.setattr(stages_mod, "LeagueRegistry", fail_if_called)
    monkeypatch.setattr(stages_mod, "FailureBuffer", fail_if_called)

    assert run_stage6_league(out_dir=tmp_path).metrics["stage6_write_once"] == 1.0
    assert run_stage7_redteam(out_dir=tmp_path).metrics["stage7_write_once"] == 1.0
    assert run_evaluation_suite(out_dir=tmp_path)["write_once"] is True
    assert run_axis_promotion_selection(out_dir=tmp_path, candidates={})["write_once"] is True
    assert not (tmp_path / "league" / "league_report.json").exists()
    assert not (tmp_path / "redteam" / "focused_correction_report.json").exists()


def test_stage6_fresh_run_rebuilds_league_registry(tmp_path: Path):
    from goofspiel.training.league import ROLE_AGGRESSIVE, ROLE_EXPLOITER, ROLE_ROBUST
    from goofspiel.training.stages import run_stage1_pretrain, run_stage6_league

    seed_ckpt = run_stage1_pretrain(
        steps=1,
        batch_size=2,
        out_dir=tmp_path / "seed",
        n_cards=3,
        local_only=True,
    ).checkpoint
    assert seed_ckpt is not None
    role_checkpoints = {
        ROLE_ROBUST: seed_ckpt,
        ROLE_AGGRESSIVE: seed_ckpt,
        ROLE_EXPLOITER: seed_ckpt,
    }

    run_stage6_league(out_dir=tmp_path, role_checkpoints=role_checkpoints, n_cards=3, seed=1)
    registry_path = tmp_path / "league" / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["agents"].append(
        {
            "agent_id": "stale_extra_robust",
            "role": ROLE_ROBUST,
            "checkpoint_path": seed_ckpt,
            "policy_version": 99,
            "metrics": {"priority": 1.0},
            "frozen": True,
            "created_at": 0.0,
        }
    )
    registry_path.write_text(json.dumps(registry, ensure_ascii=False), encoding="utf-8")

    result = run_stage6_league(out_dir=tmp_path, role_checkpoints=role_checkpoints, n_cards=3, seed=1)
    rebuilt = json.loads(registry_path.read_text(encoding="utf-8"))
    report = json.loads((tmp_path / "league" / "league_report.json").read_text(encoding="utf-8"))
    assert result.metrics["league_agents"] == 3.0
    assert [row["agent_id"] for row in rebuilt["agents"]] == [
        "seed_initial_aggressive",
        "seed_initial_exploiter",
        "seed_initial_robust",
    ]
    assert "stale_extra_robust" not in report["agent_checkpoints"]
    assert len(report["cross_play"]) == 9


def test_stage7_fresh_run_rebuilds_failure_and_correction_jsonl(tmp_path: Path):
    from goofspiel.training.stages import run_stage1_pretrain, run_stage7_redteam

    seed_ckpt = run_stage1_pretrain(
        steps=1,
        batch_size=2,
        out_dir=tmp_path / "seed",
        n_cards=3,
        local_only=True,
    ).checkpoint
    assert seed_ckpt is not None

    run_stage7_redteam(out_dir=tmp_path, init_from_checkpoint=seed_ckpt, correction_steps=1, n_cards=3, seed=41)
    run_stage7_redteam(out_dir=tmp_path, init_from_checkpoint=seed_ckpt, correction_steps=1, n_cards=3, seed=43)

    failure_rows = [
        json.loads(line)
        for line in (tmp_path / "redteam" / "failures.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    correction_rows = [
        json.loads(line)
        for line in (tmp_path / "redteam" / "corrections.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(failure_rows) == 3
    assert len(correction_rows) == 3
    assert {row["failure_id"] for row in failure_rows} == {
        "redteam_attack_0_seed43_n3",
        "redteam_attack_1_seed43_n3",
        "redteam_attack_2_seed43_n3",
    }
    assert {row["sample_id"] for row in correction_rows} == {
        "correction_0_seed43_n3",
        "correction_1_seed43_n3",
        "correction_2_seed43_n3",
    }


def test_smoke_pipeline_nonzero_rank_runs_without_local_writes(monkeypatch, tmp_path: Path):
    from types import SimpleNamespace

    from goofspiel.training import stages as stages_mod

    monkeypatch.setenv("RANK", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "2")

    def fake_setup(device="auto"):
        return stages_mod.current_runtime(), device

    def fake_stage0(*_args, **_kwargs):
        return SimpleNamespace(ok=True, checks={}, errors=[])

    def fake_metrics(stage: str, checkpoint: str | None = None):
        return stages_mod.StageMetrics(stage, 1, {"write_once": 1.0}, checkpoint)

    def fake_broadcast(obj, *, src=0):
        assert obj is None
        return {"ok": True, "summary_path": str(tmp_path / "training_smoke_summary.json")}

    monkeypatch.setattr(stages_mod, "setup_torch_distributed", fake_setup)
    monkeypatch.setattr(stages_mod, "broadcast_object", fake_broadcast)
    monkeypatch.setattr(stages_mod, "barrier_if_distributed", lambda: None)
    monkeypatch.setattr(stages_mod, "collect_system_metrics", lambda: {})
    monkeypatch.setattr(stages_mod, "run_stage0_verify", fake_stage0)
    monkeypatch.setattr(stages_mod, "generate_random_game_corpus", lambda **_kwargs: {"samples": 0})
    monkeypatch.setattr(stages_mod, "run_stage1_pretrain", lambda **_kwargs: fake_metrics("P1_PRETRAIN", "p1.pt"))
    monkeypatch.setattr(stages_mod, "run_stage2_semi_supervised", lambda **_kwargs: fake_metrics("P2_SEMI_SUPERVISED"))
    monkeypatch.setattr(stages_mod, "run_stage3_sft", lambda **_kwargs: fake_metrics("P3_STRATEGIC_SFT", "p3.pt"))
    monkeypatch.setattr(stages_mod, "run_stage4_robust_rl", lambda **_kwargs: fake_metrics("P4_ROBUST_RL", "p4.pt"))
    monkeypatch.setattr(stages_mod, "run_stage5_adaptive", lambda **_kwargs: fake_metrics("P5_ADAPTIVE", "p5.pt"))
    monkeypatch.setattr(stages_mod, "run_stage6_league", lambda **_kwargs: fake_metrics("P6_LEAGUE"))
    monkeypatch.setattr(stages_mod, "run_stage7_redteam", lambda **_kwargs: fake_metrics("P7_REDTEAM", "p7.pt"))
    monkeypatch.setattr(stages_mod, "run_evaluation_suite", lambda **_kwargs: {"write_once": True})
    monkeypatch.setattr(stages_mod, "run_axis_promotion_selection", lambda **_kwargs: {"write_once": True})
    monkeypatch.setattr(stages_mod, "_smoke_algorithmic_check", lambda *_args, **_kwargs: {"algorithmic_ok": True})

    result = stages_mod.run_smoke_pipeline(
        out_dir=tmp_path,
        steps=1,
        batch_size=1,
        n_cards=3,
        num_corpus_games=1,
    )
    assert result["ok"] is True
    assert not (tmp_path / "events" / "training_smoke.jsonl").exists()
    assert not (tmp_path / "training_smoke_summary.json").exists()


def test_stage7_ids_are_seed_deterministic(tmp_path: Path):
    from goofspiel.training.stages import run_stage7_redteam

    first = run_stage7_redteam(out_dir=tmp_path / "a", n_cards=3, correction_steps=1, seed=77)
    second = run_stage7_redteam(out_dir=tmp_path / "b", n_cards=3, correction_steps=1, seed=77)

    def ids(root: Path) -> tuple[list[str], list[str]]:
        failures = [
            json.loads(line)["failure_id"]
            for line in (root / "redteam" / "failures.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        corrections = [
            json.loads(line)["sample_id"]
            for line in (root / "redteam" / "corrections.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return failures, corrections

    assert first.metrics["stage7_write_once"] == 1.0
    assert second.metrics["stage7_write_once"] == 1.0
    assert ids(tmp_path / "a") == ids(tmp_path / "b")
