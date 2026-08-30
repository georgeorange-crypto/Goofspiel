"""Pure compact Goofspiel state and deterministic transition.

This module is intentionally independent from the mutable web/training
environment.  Search, exact teachers, and learning targets can use the same
bitmask transition semantics without cloning an environment object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from goofspiel.env import PLAYER_0, PLAYER_1, GoofspielEnv


def full_mask(n: int) -> int:
    if n < 1 or n > 13:
        raise ValueError(f"n must be in [1,13], got {n}")
    return (1 << n) - 1


def card_bit(card: int) -> int:
    if card < 1 or card > 13:
        raise ValueError(f"card must be in [1,13], got {card}")
    return 1 << (card - 1)


def mask_from_cards(cards: Iterable[int]) -> int:
    mask = 0
    for card in cards:
        mask |= card_bit(int(card))
    return mask


def legal_cards(mask: int, n: int | None = None) -> list[int]:
    max_card = n if n is not None else max(1, int(mask).bit_length())
    return [card for card in range(1, max_card + 1) if mask & card_bit(card)]


@dataclass(frozen=True)
class GameState:
    n: int
    self_mask: int
    opp_mask: int
    prize_mask: int
    current_prize: int
    self_score: int = 0
    opp_score: int = 0
    round_index: int = 1
    done: bool = False
    carry_pool: int = 0

    @classmethod
    def initial(cls, n: int, current_prize: int | None = None) -> "GameState":
        mask = full_mask(n)
        prize = int(current_prize if current_prize is not None else 1)
        if not (mask & card_bit(prize)):
            raise ValueError(f"current_prize {prize} is outside n={n}")
        return cls(
            n=n,
            self_mask=mask,
            opp_mask=mask,
            prize_mask=mask & ~card_bit(prize),
            current_prize=prize,
            round_index=1,
        )

    @property
    def total_prize_mass(self) -> int:
        return self.n * (self.n + 1) // 2

    @property
    def stake(self) -> int:
        return 0 if self.done else self.current_prize + self.carry_pool

    @property
    def self_actions(self) -> list[int]:
        return legal_cards(self.self_mask, self.n)

    @property
    def opponent_actions(self) -> list[int]:
        return legal_cards(self.opp_mask, self.n)


@dataclass(frozen=True)
class TransitionResult:
    state: GameState
    reward_self: int
    reward_opp: int
    normalized_reward: float
    winner: str | None
    discarded: int


def _choose_next_prize(prize_mask: int, next_prize: int | None) -> tuple[int, int]:
    if prize_mask == 0:
        return 0, 0
    if next_prize is None:
        next_prize = legal_cards(prize_mask)[0]
    bit = card_bit(next_prize)
    if not (prize_mask & bit):
        raise ValueError(f"next_prize={next_prize} is not in remaining prize mask")
    return int(next_prize), prize_mask & ~bit


def transition(
    state: GameState,
    self_action: int,
    opp_action: int,
    *,
    next_prize: int | None = None,
) -> TransitionResult:
    """Apply one simultaneous action pair and return a new immutable state.

    The transition implements the project's carry-over rule: a non-final tie
    rolls the whole stake into the next round; a final tie discards it.
    No randomness is used.  Chance is represented only by the caller supplied
    `next_prize`.
    """
    if state.done:
        raise RuntimeError("cannot transition from a terminal state")
    if not (state.self_mask & card_bit(self_action)):
        raise ValueError(f"illegal self_action={self_action}")
    if not (state.opp_mask & card_bit(opp_action)):
        raise ValueError(f"illegal opp_action={opp_action}")

    stake = state.current_prize + state.carry_pool
    next_self_mask = state.self_mask & ~card_bit(self_action)
    next_opp_mask = state.opp_mask & ~card_bit(opp_action)
    is_final = state.prize_mask == 0

    reward_self = 0
    reward_opp = 0
    carry_next = 0
    discarded = 0
    winner: str | None = None

    if self_action > opp_action:
        reward_self = stake
        winner = "self"
    elif self_action < opp_action:
        reward_opp = stake
        winner = "opponent"
    elif is_final:
        discarded = stake
    else:
        carry_next = stake

    if is_final:
        next_state = GameState(
            n=state.n,
            self_mask=next_self_mask,
            opp_mask=next_opp_mask,
            prize_mask=0,
            current_prize=0,
            self_score=state.self_score + reward_self,
            opp_score=state.opp_score + reward_opp,
            round_index=state.round_index,
            done=True,
            carry_pool=0,
        )
    else:
        current_prize, prize_mask = _choose_next_prize(state.prize_mask, next_prize)
        next_state = GameState(
            n=state.n,
            self_mask=next_self_mask,
            opp_mask=next_opp_mask,
            prize_mask=prize_mask,
            current_prize=current_prize,
            self_score=state.self_score + reward_self,
            opp_score=state.opp_score + reward_opp,
            round_index=state.round_index + 1,
            done=False,
            carry_pool=carry_next,
        )

    return TransitionResult(
        state=next_state,
        reward_self=reward_self,
        reward_opp=reward_opp,
        normalized_reward=(reward_self - reward_opp) / state.total_prize_mass,
        winner=winner,
        discarded=discarded,
    )


def state_from_env(env: GoofspielEnv, player: str = PLAYER_0) -> GameState:
    opponent = PLAYER_1 if player == PLAYER_0 else PLAYER_0
    current = int(env.current_prize or 0)
    prizes = [int(p) for p in env.remaining_prizes]
    return GameState(
        n=env.num_cards,
        self_mask=mask_from_cards(env.remaining_cards[player]),
        opp_mask=mask_from_cards(env.remaining_cards[opponent]),
        prize_mask=mask_from_cards(prizes),
        current_prize=current,
        self_score=int(env.scores[player]),
        opp_score=int(env.scores[opponent]),
        round_index=int(env.round),
        done=bool(env.done),
        carry_pool=int(getattr(env, "carry_pool", 0)),
    )
