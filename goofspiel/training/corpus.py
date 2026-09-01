"""Game corpus generation utilities."""

from __future__ import annotations

import random
from pathlib import Path

from goofspiel.env import PLAYER_0, PLAYER_1, GoofspielEnv
from goofspiel.game import state_from_env
from goofspiel.training.data import GameCorpusSample, JsonlStore, RoundRecord, state_record_from_game_state
from goofspiel.training.distributed import barrier_if_distributed, broadcast_object, current_runtime, setup_torch_distributed


def generate_random_game_corpus(
    *,
    out_path: str | Path,
    num_games: int,
    n_min: int = 3,
    n_max: int = 13,
    seed: int = 1,
    append: bool = False,
) -> dict[str, int]:
    runtime = current_runtime()
    if runtime.is_distributed:
        runtime, _ = setup_torch_distributed("auto")
    if not runtime.is_rank0:
        payload = broadcast_object(None, src=0)
        if payload is None:
            raise RuntimeError("corpus payload broadcast failed")
        barrier_if_distributed()
        return payload

    rng = random.Random(seed)
    path = Path(out_path)
    if not append:
        path.unlink(missing_ok=True)
    store = JsonlStore(path)
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
    payload = {"games": int(num_games), "samples": written, "rank_owner": 0, "append": int(bool(append))}
    payload = broadcast_object(payload, src=0)
    if payload is None:
        raise RuntimeError("corpus payload broadcast failed")
    barrier_if_distributed()
    return payload
