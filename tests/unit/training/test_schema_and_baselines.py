from __future__ import annotations

from goofspiel.training.baselines import PRIMARY, baselines_by_arena, default_baselines
from goofspiel.training.data import PublicStateRecord
from goofspiel.training.schema import index_to_rank, rank_to_index, stable_state_hash


def test_rank_index_conversion_is_centralized():
    assert rank_to_index(1) == 0
    assert rank_to_index(13) == 12
    assert index_to_rank(0) == 1
    assert index_to_rank(12) == 13


def test_public_state_gets_stable_hash():
    state = PublicStateRecord(
        n=3,
        self_mask=7,
        opponent_mask=7,
        prize_mask=6,
        current_prize=1,
        self_score=0,
        opponent_score=0,
        round_index=1,
    )
    assert state.state_hash == stable_state_hash(
        {
            "n": 3,
            "self_mask": 7,
            "opponent_mask": 7,
            "prize_mask": 6,
            "current_prize": 1,
            "self_score": 0,
            "opponent_score": 0,
            "round_index": 1,
            "carry_pool": 0,
            "done": False,
        }
    )


def test_baseline_registry_contains_required_primary_groups():
    names = {baseline.name for baseline in default_baselines() if baseline.tier == PRIMARY}
    assert {"Random", "Exact Nash", "Minimax-Q", "CFR", "CFR+", "NeuRD", "R-NaD", "SM-MCTS"} <= names
    assert any(baseline.name == "Exact Nash" for baseline in baselines_by_arena("A"))
    assert all(baseline.implemented for baseline in default_baselines())
    assert all(baseline.entrypoint != "planned" for baseline in default_baselines())


def test_all_policy_baselines_are_callable():
    from goofspiel.game import GameState
    from goofspiel.training.baseline_algorithms import create_baseline

    state = GameState.initial(3, current_prize=1)
    for name in ["Minimax-Q", "R-NaD", "IPPO", "NFSP", "Deep CFR"]:
        policy = create_baseline(name).policy_for_state(state)
        assert len(policy) == 13
        assert abs(sum(policy) - 1.0) < 1e-6
