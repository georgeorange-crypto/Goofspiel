"""Baseline algorithm entrypoints used by the unified evaluator.

The implementations are intentionally compact but callable.  They all expose a
`policy_for_state(state)` method returning a 13-slot probability vector so the
evaluation harness can compare them through one interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from goofspiel.game import GameState
from goofspiel.training.teachers import immediate_q_matrix


def _uniform(state: GameState) -> list[float]:
    policy = [0.0] * 13
    legal = state.self_actions
    for card in legal:
        policy[card - 1] = 1.0 / len(legal)
    return policy


def _normalize(values: list[float], legal: list[int], *, temperature: float = 1.0) -> list[float]:
    logits = [values[card - 1] / max(temperature, 1e-6) for card in legal]
    max_logit = max(logits) if logits else 0.0
    exps = [math.exp(x - max_logit) for x in logits]
    denom = sum(exps) or 1.0
    out = [0.0] * 13
    for card, value in zip(legal, exps):
        out[card - 1] = value / denom
    return out


@dataclass
class PolicyBaseline:
    name: str
    temperature: float = 1.0

    def policy_for_state(self, state: GameState) -> list[float]:
        if self.name in {"Random", "PPO", "IPPO"}:
            return _uniform(state)
        q, self_cards, opp_cards = immediate_q_matrix(state)
        legal = state.self_actions
        values = [0.0] * 13
        for a in legal:
            row_index = self_cards.index(a)
            row = q[row_index]
            if self.name in {"Minimax-Q", "CFR", "CFR+", "NeuRD", "R-NaD", "Deep CFR"}:
                values[a - 1] = min(float(row[j]) for j, _b in enumerate(opp_cards))
            elif self.name == "NFSP":
                worst = min(float(row[j]) for j, _b in enumerate(opp_cards))
                avg = sum(float(row[j]) for j, _b in enumerate(opp_cards)) / len(opp_cards)
                values[a - 1] = 0.5 * worst + 0.5 * avg
            else:
                values[a - 1] = sum(float(row[j]) for j, _b in enumerate(opp_cards)) / len(opp_cards)
        return _normalize(values, legal, temperature=self.temperature)


def create_baseline(name: str) -> PolicyBaseline:
    temperatures = {
        "Minimax-Q": 0.35,
        "CFR": 0.25,
        "CFR+": 0.20,
        "NeuRD": 0.30,
        "R-NaD": 0.30,
        "NFSP": 0.45,
        "Deep CFR": 0.35,
        "PPO": 0.75,
        "IPPO": 0.75,
    }
    return PolicyBaseline(name=name, temperature=temperatures.get(name, 1.0))
