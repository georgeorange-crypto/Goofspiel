"""Persistent data schemas and JSONL stores for the training pipeline."""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Generic, Iterable, Iterator, TypeVar

from goofspiel.training.schema import SCHEMA_VERSION, stable_state_hash
T = TypeVar("T")


@dataclass
class PublicStateRecord:
    n: int
    self_mask: int
    opponent_mask: int
    prize_mask: int
    current_prize: int
    self_score: int
    opponent_score: int
    round_index: int
    carry_pool: int = 0
    done: bool = False
    state_hash: str | None = None

    def __post_init__(self) -> None:
        if self.state_hash is None:
            payload = {
                "n": self.n,
                "self_mask": self.self_mask,
                "opponent_mask": self.opponent_mask,
                "prize_mask": self.prize_mask,
                "current_prize": self.current_prize,
                "self_score": self.self_score,
                "opponent_score": self.opponent_score,
                "round_index": self.round_index,
                "carry_pool": self.carry_pool,
                "done": self.done,
            }
            self.state_hash = stable_state_hash(payload)


@dataclass
class RoundRecord:
    round_index: int
    prize: int
    self_action: int
    opponent_action: int
    reward_self: int
    reward_opponent: int
    carry_in: int = 0
    carry_out: int = 0
    done: bool = False


@dataclass
class GameCorpusSample:
    sample_id: str
    state: PublicStateRecord
    round_event: RoundRecord | None
    opponent_id: str = "synthetic"
    session_id: str = "default"
    source: str = "generated"
    schema_version: str = SCHEMA_VERSION
    created_at: float = field(default_factory=time.time)


@dataclass
class ExactSample:
    sample_id: str
    state: PublicStateRecord
    q_matrix: list[list[float]]
    row_policy: list[float]
    column_policy: list[float]
    value: float
    solver_precision: str = "fp64_scipy_highs"
    source: str = "EXACT"
    schema_version: str = SCHEMA_VERSION


@dataclass
class TeacherSample:
    sample_id: str
    state: PublicStateRecord
    teacher_q: list[list[float]] | None
    teacher_policy: list[float] | None
    teacher_value: float | None
    teacher_source: str
    teacher_confidence: float
    teacher_disagreement: float = 0.0
    schema_version: str = SCHEMA_VERSION


@dataclass
class RobustTrajectorySample:
    sample_id: str
    states: list[PublicStateRecord]
    rounds: list[RoundRecord]
    behavior_policy_self: list[list[float]]
    behavior_policy_opponent: list[list[float]]
    action_prob_self: list[float]
    action_prob_opponent: list[float]
    final_score_diff: int
    model_version: str
    opponent_version: str
    n: int
    mode: str = "ROBUST"
    schema_version: str = SCHEMA_VERSION


@dataclass
class OpponentSession:
    session_id: str
    opponent_id: str
    strategy_regime_id: str
    games: list[list[RoundRecord]]
    schema_version: str = SCHEMA_VERSION


@dataclass
class AdaptiveTrajectorySample:
    sample_id: str
    states: list[PublicStateRecord]
    rounds: list[RoundRecord]
    opponent_context_id: str
    final_score_diff: int
    mode: str = "ADAPTIVE"
    schema_version: str = SCHEMA_VERSION


@dataclass
class FailureRecord:
    failure_id: str
    failure_type: str
    state: PublicStateRecord
    model_version: str
    teacher_source: str | None = None
    details: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION


@dataclass
class ReanalysisRecord:
    sample_id: str
    original_sample_id: str
    state: PublicStateRecord
    new_teacher_source: str
    teacher_q: list[list[float]] | None = None
    teacher_policy: list[float] | None = None
    teacher_value: float | None = None
    schema_version: str = SCHEMA_VERSION


def _to_jsonable(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    return obj


class JsonlStore(Generic[T]):
    """Append-only JSONL store with optional gzip read support."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, item: T) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_to_jsonable(item), ensure_ascii=False, sort_keys=True))
            handle.write("\n")

    def extend(self, items: Iterable[T]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            for item in items:
                handle.write(json.dumps(_to_jsonable(item), ensure_ascii=False, sort_keys=True))
                handle.write("\n")

    def iter_dicts(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        opener = gzip.open if self.path.suffix == ".gz" else open
        with opener(self.path, "rt", encoding="utf-8") as handle:  # type: ignore[arg-type]
            for line in handle:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if data.get("schema_version") != SCHEMA_VERSION:
                        raise ValueError(
                            f"unsupported schema_version={data.get('schema_version')!r} "
                            f"in {self.path}"
                        )
                    yield data

    def count(self) -> int:
        return sum(1 for _ in self.iter_dicts())


def state_record_from_game_state(state: Any) -> PublicStateRecord:
    return PublicStateRecord(
        n=int(state.n),
        self_mask=int(state.self_mask),
        opponent_mask=int(state.opp_mask),
        prize_mask=int(state.prize_mask),
        current_prize=int(state.current_prize),
        self_score=int(state.self_score),
        opponent_score=int(state.opp_score),
        round_index=int(state.round_index),
        carry_pool=int(getattr(state, "carry_pool", 0)),
        done=bool(state.done),
    )
