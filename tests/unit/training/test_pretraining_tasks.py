from __future__ import annotations

from goofspiel.game import GameState
from goofspiel.training.pretraining import build_pretraining_targets


def test_pretraining_targets_cover_required_tasks():
    state = GameState.initial(3, current_prize=1)
    targets = build_pretraining_targets(state, self_action=1, opponent_action=2)
    assert targets.player_swap_state.self_mask == state.opp_mask
    assert targets.next_state.round_index == 2
    assert len(targets.immediate_q) == 3
    assert targets.masked_history_action == 1
    assert targets.future_opponent_action == 2
    assert targets.style_pair_id
