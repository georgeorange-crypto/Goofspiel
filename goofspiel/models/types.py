"""Tensor dataclasses used by the Goofspiel neural model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from goofspiel.game import GameState, legal_cards


@dataclass
class PublicStateBatch:
    n_cards: Tensor
    self_cards: Tensor
    opponent_cards: Tensor
    remaining_prizes: Tensor
    current_prize: Tensor
    self_score: Tensor
    opponent_score: Tensor
    round_idx: Tensor
    rank_mask: Tensor
    self_action_mask: Tensor
    opponent_action_mask: Tensor
    carry_pool: Tensor

    @property
    def device(self) -> torch.device:
        return self.self_cards.device

    @property
    def batch_size(self) -> int:
        return int(self.self_cards.shape[0])

    @property
    def max_cards(self) -> int:
        return int(self.self_cards.shape[1])


@dataclass
class HistoryBatch:
    prize: Tensor
    self_action: Tensor
    opponent_action: Tensor
    score_diff: Tensor
    outcome: Tensor
    round_idx: Tensor
    valid_mask: Tensor


@dataclass
class OpponentMemoryBatch:
    game_summary_sequence: Tensor
    valid_mask: Tensor


def public_state_from_game(
    states: list[GameState],
    *,
    max_cards: int = 13,
    device: torch.device | str | None = None,
) -> PublicStateBatch:
    device = torch.device(device or "cpu")
    batch = len(states)
    self_cards = torch.zeros(batch, max_cards, device=device)
    opp_cards = torch.zeros_like(self_cards)
    prizes = torch.zeros_like(self_cards)
    rank_mask = torch.zeros_like(self_cards, dtype=torch.bool)
    self_mask = torch.zeros_like(self_cards, dtype=torch.bool)
    opp_mask = torch.zeros_like(self_cards, dtype=torch.bool)
    n_cards = torch.zeros(batch, device=device, dtype=torch.long)
    current_prize = torch.zeros(batch, device=device, dtype=torch.long)
    self_score = torch.zeros(batch, device=device)
    opp_score = torch.zeros(batch, device=device)
    round_idx = torch.zeros(batch, device=device)
    carry_pool = torch.zeros(batch, device=device)

    for row, state in enumerate(states):
        n_cards[row] = state.n
        current_prize[row] = state.current_prize
        self_score[row] = float(state.self_score)
        opp_score[row] = float(state.opp_score)
        round_idx[row] = float(state.round_index)
        carry_pool[row] = float(state.carry_pool)
        rank_mask[row, : state.n] = True
        for card in legal_cards(state.self_mask, state.n):
            self_cards[row, card - 1] = 1.0
            self_mask[row, card - 1] = True
        for card in legal_cards(state.opp_mask, state.n):
            opp_cards[row, card - 1] = 1.0
            opp_mask[row, card - 1] = True
        for card in legal_cards(state.prize_mask, state.n):
            prizes[row, card - 1] = 1.0
        if state.current_prize:
            prizes[row, state.current_prize - 1] = 1.0

    return PublicStateBatch(
        n_cards=n_cards,
        self_cards=self_cards,
        opponent_cards=opp_cards,
        remaining_prizes=prizes,
        current_prize=current_prize,
        self_score=self_score,
        opponent_score=opp_score,
        round_idx=round_idx,
        rank_mask=rank_mask,
        self_action_mask=self_mask,
        opponent_action_mask=opp_mask,
        carry_pool=carry_pool,
    )
