"""Baseline registry required by comparative evaluation specs."""

from __future__ import annotations

from dataclasses import dataclass

PRIMARY = "PRIMARY"
REFERENCE = "REFERENCE"


@dataclass(frozen=True)
class BaselineCard:
    name: str
    tier: str
    arena: str
    category: str
    implemented: bool
    entrypoint: str
    notes: str = ""


def default_baselines() -> list[BaselineCard]:
    return [
        BaselineCard("Random", PRIMARY, "A/B/D", "non_learning", True, "goofspiel.bots.RandomBot"),
        BaselineCard("Exact Nash", PRIMARY, "A", "exact", True, "goofspiel.solver.GoofspielCarrySolver"),
        BaselineCard("Minimax-Q", PRIMARY, "A", "joint_action_rl", True, "goofspiel.training.baseline_algorithms.create_baseline"),
        BaselineCard("CFR", PRIMARY, "A/C", "game_theoretic", True, "goofspiel.reasoning.run_gt_cfr"),
        BaselineCard("CFR+", PRIMARY, "A/C", "game_theoretic", True, "goofspiel.learning.game_theory.regret_matching_plus"),
        BaselineCard("NeuRD", PRIMARY, "A/B", "neural_game_theoretic", True, "goofspiel.learning.game_theory.neurd"),
        BaselineCard("R-NaD", PRIMARY, "B", "neural_game_theoretic", True, "goofspiel.training.baseline_algorithms.create_baseline"),
        BaselineCard("Heuristic Suite", REFERENCE, "A/B/D", "non_learning", True, "goofspiel.bots.HeuristicBot"),
        BaselineCard("PPO", REFERENCE, "A/B", "model_free_rl", True, "scripts.train_n5_ppo"),
        BaselineCard("IPPO", REFERENCE, "A/B", "model_free_rl", True, "goofspiel.training.baseline_algorithms.create_baseline"),
        BaselineCard("NFSP", REFERENCE, "A", "neural_game_theoretic", True, "goofspiel.training.baseline_algorithms.create_baseline"),
        BaselineCard("Deep CFR", REFERENCE, "A/C", "neural_game_theoretic", True, "goofspiel.training.baseline_algorithms.create_baseline"),
        BaselineCard("SM-MCTS", PRIMARY, "C", "search", True, "goofspiel.reasoning.run_sm_mcts"),
        BaselineCard("Adaptive BR", PRIMARY, "D", "opponent_adaptation", True, "goofspiel.reasoning.final_decision"),
    ]


def baselines_by_arena(arena: str) -> list[BaselineCard]:
    needle = arena.upper()
    return [baseline for baseline in default_baselines() if needle in baseline.arena.split("/")]
