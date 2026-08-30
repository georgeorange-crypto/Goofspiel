from __future__ import annotations

from goofspiel.game import GameState
from goofspiel.training.data import GameCorpusSample, state_record_from_game_state
from goofspiel.training.datasets import (
    AdaptiveTrajectoryBuffer,
    ExactDataset,
    GameCorpus,
    OpponentSessionBuffer,
    ReanalysisBuffer,
    RobustTrajectoryBuffer,
    TeacherDataset,
)


def test_dataset_pools_are_separate(tmp_path):
    pools = [
        GameCorpus(tmp_path / "game.jsonl"),
        ExactDataset(tmp_path / "exact.jsonl"),
        TeacherDataset(tmp_path / "teacher.jsonl"),
        RobustTrajectoryBuffer(tmp_path / "robust.jsonl"),
        OpponentSessionBuffer(tmp_path / "session.jsonl"),
        AdaptiveTrajectoryBuffer(tmp_path / "adaptive.jsonl"),
        ReanalysisBuffer(tmp_path / "reanalysis.jsonl"),
    ]
    assert len({pool.path for pool in pools}) == len(pools)
    corpus = pools[0]
    corpus.add(GameCorpusSample("s", state_record_from_game_state(GameState.initial(3)), None))
    assert corpus.count() == 1
    assert pools[1].count() == 0
