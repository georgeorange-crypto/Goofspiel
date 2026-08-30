"""Pretraining task builders for P1."""

from __future__ import annotations

from dataclasses import dataclass

from goofspiel.game import GameState, transition
from goofspiel.training.teachers import immediate_q_matrix


@dataclass(frozen=True)
class PretrainingTargets:
    player_swap_state: GameState
    next_state: GameState
    immediate_q: list[list[float]]
    masked_history_action: int
    future_opponent_action: int
    style_pair_id: str


def build_pretraining_targets(state: GameState, *, self_action: int, opponent_action: int) -> PretrainingTargets:
    next_prize = state.self_actions[0] if state.prize_mask & 1 else (state.opponent_actions[0] if state.prize_mask else None)
    if next_prize is not None and not (state.prize_mask & (1 << (next_prize - 1))):
        next_prize = next((card for card in range(1, state.n + 1) if state.prize_mask & (1 << (card - 1))), None)
    out = transition(state, self_action, opponent_action, next_prize=next_prize)
    swapped = GameState(
        n=state.n,
        self_mask=state.opp_mask,
        opp_mask=state.self_mask,
        prize_mask=state.prize_mask,
        current_prize=state.current_prize,
        self_score=state.opp_score,
        opp_score=state.self_score,
        round_index=state.round_index,
        done=state.done,
        carry_pool=state.carry_pool,
    )
    q, _self_cards, _opp_cards = immediate_q_matrix(state)
    return PretrainingTargets(
        player_swap_state=swapped,
        next_state=out.state,
        immediate_q=q.tolist(),
        masked_history_action=self_action,
        future_opponent_action=opponent_action,
        style_pair_id=f"style:n{state.n}:round{state.round_index}",
    )
