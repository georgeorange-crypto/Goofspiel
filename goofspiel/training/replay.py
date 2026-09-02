"""Replay buffer helpers for self-play trajectories."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from goofspiel.training.data import JsonlStore, PublicStateRecord, RobustTrajectorySample, RoundRecord


class TrajectoryReplayBuffer:
    """Small append-only replay buffer backed by JSONL for reproducible smoke runs."""

    def __init__(self, path: str | Path, *, max_size: int = 10_000) -> None:
        self.store = JsonlStore[RobustTrajectorySample](path)
        self.max_size = int(max_size)
        self.items: list[RobustTrajectorySample] = []

    @property
    def path(self) -> Path:
        return self.store.path

    def append_many(self, samples: list[RobustTrajectorySample]) -> None:
        self.items.extend(samples)
        if len(self.items) > self.max_size:
            self.items = self.items[-self.max_size:]
        self.store.extend(samples)

    def sample(self, k: int, rng: random.Random | None = None) -> list[RobustTrajectorySample]:
        if not self.items:
            return []
        rng = rng or random.Random()
        k = min(int(k), len(self.items))
        return rng.sample(self.items, k)

    def count(self) -> int:
        return len(self.items)

    def persisted_count(self) -> int:
        return self.store.count()

    def snapshot(self) -> dict[str, Any]:
        """Stable in-memory replay snapshot for crash-resume checkpoints.

        This intentionally serializes ``self.items`` rather than reading the
        append-only JSONL backing file, because the JSONL may contain samples
        written after the last periodic checkpoint.
        """
        return {
            "schema": "goofspiel.stage4.replay_snapshot.v1",
            "max_size": self.max_size,
            "items": [_trajectory_to_dict(item) for item in self.items],
        }

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        if snapshot.get("schema") != "goofspiel.stage4.replay_snapshot.v1":
            raise ValueError(f"unsupported replay snapshot schema={snapshot.get('schema')!r}")
        self.max_size = int(snapshot.get("max_size", self.max_size))
        self.items = [_trajectory_from_dict(row) for row in snapshot.get("items", [])]

    def rewrite_store_from_memory(self) -> None:
        """Rewrite the JSONL mirror from the current in-memory replay state."""
        self.store.path.unlink(missing_ok=True)
        self.store.extend(self.items)


def replay_summary(buffer: TrajectoryReplayBuffer) -> dict[str, Any]:
    transitions = sum(len(item.rounds) for item in buffer.items)
    return {
        "path": str(buffer.path),
        "in_memory_samples": buffer.count(),
        "persisted_samples": buffer.persisted_count(),
        "transitions": transitions,
    }


def _state_from_dict(row: dict[str, Any]) -> PublicStateRecord:
    return PublicStateRecord(
        n=int(row["n"]),
        self_mask=int(row["self_mask"]),
        opponent_mask=int(row["opponent_mask"]),
        prize_mask=int(row["prize_mask"]),
        current_prize=int(row["current_prize"]),
        self_score=int(row["self_score"]),
        opponent_score=int(row["opponent_score"]),
        round_index=int(row["round_index"]),
        carry_pool=int(row.get("carry_pool", 0)),
        done=bool(row.get("done", False)),
        state_hash=row.get("state_hash"),
    )


def _round_from_dict(row: dict[str, Any]) -> RoundRecord:
    return RoundRecord(
        round_index=int(row["round_index"]),
        prize=int(row["prize"]),
        self_action=int(row["self_action"]),
        opponent_action=int(row["opponent_action"]),
        reward_self=int(row["reward_self"]),
        reward_opponent=int(row["reward_opponent"]),
        carry_in=int(row.get("carry_in", 0)),
        carry_out=int(row.get("carry_out", 0)),
        done=bool(row.get("done", False)),
    )


def _trajectory_to_dict(sample: RobustTrajectorySample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "states": [state.__dict__ for state in sample.states],
        "rounds": [round_record.__dict__ for round_record in sample.rounds],
        "behavior_policy_self": sample.behavior_policy_self,
        "behavior_policy_opponent": sample.behavior_policy_opponent,
        "action_prob_self": sample.action_prob_self,
        "action_prob_opponent": sample.action_prob_opponent,
        "final_score_diff": int(sample.final_score_diff),
        "model_version": sample.model_version,
        "opponent_version": sample.opponent_version,
        "n": int(sample.n),
        "mode": sample.mode,
        "schema_version": sample.schema_version,
    }


def _trajectory_from_dict(row: dict[str, Any]) -> RobustTrajectorySample:
    return RobustTrajectorySample(
        sample_id=str(row["sample_id"]),
        states=[_state_from_dict(state) for state in row.get("states", [])],
        rounds=[_round_from_dict(round_record) for round_record in row.get("rounds", [])],
        behavior_policy_self=[[float(x) for x in pol] for pol in row.get("behavior_policy_self", [])],
        behavior_policy_opponent=[[float(x) for x in pol] for pol in row.get("behavior_policy_opponent", [])],
        action_prob_self=[float(x) for x in row.get("action_prob_self", [])],
        action_prob_opponent=[float(x) for x in row.get("action_prob_opponent", [])],
        final_score_diff=int(row["final_score_diff"]),
        model_version=str(row["model_version"]),
        opponent_version=str(row["opponent_version"]),
        n=int(row["n"]),
        mode=str(row.get("mode", "ROBUST")),
        schema_version=str(row.get("schema_version", "goofspiel.training.v1")),
    )
