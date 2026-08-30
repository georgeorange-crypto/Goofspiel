"""Game corpus generation utilities."""

from __future__ import annotations

import random
from pathlib import Path

from goofspiel.env import PLAYER_0, PLAYER_1, GoofspielEnv
from goofspiel.game import state_from_env
from goofspiel.training.data import GameCorpusSample, JsonlStore, RoundRecord, state_record_from_game_state


def generate_random_game_corpus(
    *,
    out_path: str | Path,
    num_games: int,
    n_min: int = 3,
    n_max: int = 13,
    seed: int = 1,
) -> dict[str, int]:
    rng = random.Random(seed)
    store = JsonlStore(out_path)
    written = 0
    for game_idx in range(num_games):
        n = rng.randint(n_min, n_max)
        env = GoofspielEnv(num_cards=n, rng=random.Random(rng.randint(0, 2**31 - 1)))
        env.reset()
        round_idx = 0
        while not env.done:
            before = state_from_env(env, PLAYER_0)
            a0 = rng.choice(env.legal_actions(PLAYER_0))
            a1 = rng.choice(env.legal_actions(PLAYER_1))
            _obs, rewards, done, info = env.step({PLAYER_0: a0, PLAYER_1: a1})
            round_idx += 1
            event = RoundRecord(
                round_index=round_idx,
                prize=int(info["round_prize"]),
                self_action=int(a0),
                opponent_action=int(a1),
                reward_self=int(rewards[PLAYER_0]),
                reward_opponent=int(rewards[PLAYER_1]),
                carry_in=int(info["carry_in"]),
                carry_out=int(info["carry_out"]),
                done=bool(done),
            )
            store.append(
                GameCorpusSample(
                    sample_id=f"random:{seed}:{game_idx}:{round_idx}",
                    state=state_record_from_game_state(before),
                    round_event=event,
                    opponent_id="random",
                    session_id=f"random:{seed}:{game_idx}",
                    source="random_legal_play",
                )
            )
            written += 1
    return {"games": int(num_games), "samples": written}
