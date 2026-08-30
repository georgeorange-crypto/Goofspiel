"""Evaluation and baseline utilities for training promotion gates."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from goofspiel.bots import BOT_HEURISTIC, BOT_RANDOM, create_bot
from goofspiel.env import PLAYER_0, PLAYER_1, GoofspielEnv
from goofspiel.solver import estimate_complexity


@dataclass
class EvaluationReport:
    name: str
    metrics: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    passed: bool = True


def evaluate_bot_matchup(
    *,
    bot_a: str = BOT_HEURISTIC,
    bot_b: str = BOT_RANDOM,
    num_games: int = 32,
    n_cards: int = 13,
    seed: int = 1,
) -> EvaluationReport:
    rng = random.Random(seed)
    diffs: list[int] = []
    wins = draws = 0
    for idx in range(num_games):
        env = GoofspielEnv(num_cards=n_cards, rng=random.Random(rng.randint(0, 2**31 - 1)))
        a = create_bot(bot_a, seed=rng.randint(0, 2**31 - 1))
        b = create_bot(bot_b, seed=rng.randint(0, 2**31 - 1))
        env.reset()
        while not env.done:
            aa = a.choose_action(env, PLAYER_0)
            bb = b.choose_action(env, PLAYER_1)
            env.step({PLAYER_0: aa, PLAYER_1: bb})
        diff = int(env.scores[PLAYER_0] - env.scores[PLAYER_1])
        diffs.append(diff)
        if diff > 0:
            wins += 1
        elif diff == 0:
            draws += 1
    mean = sum(diffs) / max(1, len(diffs))
    return EvaluationReport(
        name="bot_matchup",
        metrics={
            "games": float(num_games),
            "mean_score_diff": float(mean),
            "win_rate": wins / max(1, num_games),
            "draw_rate": draws / max(1, num_games),
        },
        details={"bot_a": bot_a, "bot_b": bot_b, "n_cards": n_cards},
    )


def exact_feasibility_sweep(max_n: int = 13) -> EvaluationReport:
    risks: dict[str, str] = {}
    states: dict[str, int] = {}
    for n in range(1, max_n + 1):
        rpt = estimate_complexity(n)
        risks[str(n)] = rpt.risk
        states[str(n)] = int(rpt.chance_states)
    return EvaluationReport(
        name="exact_feasibility_sweep",
        metrics={"max_n": float(max_n)},
        details={"risk_by_n": risks, "states_by_n": states},
    )
