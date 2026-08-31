"""Honest checkpoint evaluation harness (Phase 0.1 / 0.3).

The rest of the training code never actually evaluates a *trained* checkpoint:
``run_unified_benchmark`` takes no model and ``E2_N13_ROBUST`` resolves to
``Heuristic vs Random``.  This module fixes the thermometer.  It:

* loads a real ``GoofspielModel`` checkpoint and exposes its robust policy as a
  plain ``policy_fn: GameState -> {card: prob}``;
* plays that policy against the ground-truth bots (``Random`` / ``Heuristic`` /
  ``Nash``) through the real :class:`~goofspiel.env.GoofspielEnv`, reporting
  **win-rate** and **mean score-diff**;
* reports two clearly-distinct exploitability figures, never conflated
  (Phase 0.3 naming discipline):

  ``full_game_exploitability``
      The true best-response gap against the recursive carry-over game value,
      computed exactly over the whole game tree.  Small ``N`` only (the tree is
      enumerated), and named to mean exactly what it computes.

  ``one_step_matrix_nash_gap``
      A cheap proxy: the best-response gap of the policy's row strategy on the
      *current-state immediate* payoff matrix only.  Valid for any ``N`` but it
      is **not** full-game exploitability and is never labelled as such.

Everything below is deterministic given a seed.  Nothing here hard-codes a
success value; every number is computed from the checkpoint under test.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from goofspiel.env import PLAYER_0, PLAYER_1, GoofspielEnv
from goofspiel.game import GameState, legal_cards, transition
from goofspiel.game.state import card_bit, mask_from_cards, state_from_env

# A policy maps a concrete game state to a probability over that state's legal
# self actions.  Probabilities are keyed by card value (1..n) and sum to 1.
PolicyFn = Callable[[GameState], Mapping[int, float]]

# Above this many cards the full-game best-response enumeration is refused so we
# never silently return a truncated (and therefore dishonest) exploitability.
DEFAULT_MAX_FULL_GAME_N = 6


# ======================================================================
# Model / checkpoint -> policy_fn
# ======================================================================
def load_model_from_checkpoint(path: str | Path, *, device: str = "cpu") -> tuple[Any, dict[str, Any]]:
    """Load a ``GoofspielModel`` checkpoint and return ``(model, metadata)``.

    The checkpoint format is the one written by
    :func:`goofspiel.training.checkpoint.save_checkpoint`: a dict with
    ``model_state`` and ``metadata``.  ``max_cards`` is taken from the stored
    config when present so the reconstructed module matches the saved weights.
    """
    import torch

    from goofspiel.models import GoofspielModel
    from goofspiel.training.checkpoint import load_checkpoint

    payload = load_checkpoint(path)
    metadata = dict(payload.get("metadata", {}))
    config = dict(metadata.get("config", {}))
    # ``max_cards`` governs the input/output width; default 13 matches every
    # stage runner (they all build ``GoofspielModel(max_cards=13)``).
    max_cards = int(config.get("max_cards", 13))
    model = GoofspielModel(max_cards=max_cards).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, metadata


def robust_policy_fn(
    model: Any,
    *,
    device: str = "cpu",
    greedy: bool = True,
    temperature: float = 1.0,
) -> PolicyFn:
    """Wrap a model's ``robust_policy_logits`` as a ``PolicyFn``.

    ``greedy=True`` puts all mass on the argmax legal card (deterministic, the
    right default for evaluation).  ``greedy=False`` returns the softmax over
    legal cards at the given ``temperature``.
    """
    import torch

    from goofspiel.models import public_state_from_game

    max_cards = int(getattr(model, "max_cards", 13))

    def policy(state: GameState) -> dict[int, float]:
        legal = state.self_actions
        if not legal:
            return {}
        batch = public_state_from_game([state], max_cards=max_cards, device=device)
        with torch.no_grad():
            logits = model(batch).robust_policy_logits[0]
        # Restrict to legal cards, then softmax (or argmax) over just those.
        legal_logits = {card: float(logits[card - 1]) for card in legal}
        if greedy:
            best = max(legal_logits, key=legal_logits.get)
            return {card: (1.0 if card == best else 0.0) for card in legal}
        temp = max(float(temperature), 1e-6)
        max_logit = max(legal_logits.values())
        exps = {card: math.exp((v - max_logit) / temp) for card, v in legal_logits.items()}
        denom = sum(exps.values()) or 1.0
        return {card: v / denom for card, v in exps.items()}

    return policy


def uniform_policy_fn() -> PolicyFn:
    """Reference policy: uniform over legal self actions (a ``Random`` player)."""

    def policy(state: GameState) -> dict[int, float]:
        legal = state.self_actions
        if not legal:
            return {}
        p = 1.0 / len(legal)
        return {card: p for card in legal}

    return policy


def carry_nash_policy_fn(n_cards: int) -> PolicyFn:
    """Exact carry-over Nash policy (the equilibrium of the game ``transition``
    implements).  Used to validate the exploitability metric: a Nash policy must
    score ``full_game_exploitability`` ~ 0.  Small ``N`` only (solver budget).
    """
    from goofspiel.solver import GoofspielCarrySolver, SolverConfig

    # ``use_symmetry`` / ``short_cut_equal_hand`` populate a *canonicalised*
    # policy map that misses many raw reachable keys; those misses would fall
    # back to uniform and make an equilibrium policy look exploitable.  For the
    # reference policy we want a complete raw map, so both are disabled.
    solver = GoofspielCarrySolver(
        SolverConfig(
            use_symmetry=False,
            short_cut_equal_hand=False,
            skip_benchmark=True,
            skip_calibration_solve=True,
        )
    )
    result = solver.solve_with_policy(n_cards, force=False)
    policy_map = result.policy_map or {}

    def policy(state: GameState) -> dict[int, float]:
        legal = state.self_actions
        if not legal:
            return {}
        r_mask = state.prize_mask | (card_bit(state.current_prize) if state.current_prize else 0)
        key = (state.self_mask, state.opp_mask, r_mask, state.carry_pool, state.current_prize)
        entry = policy_map.get(key)
        if entry is None:
            # Off the solved support (e.g. an unreachable state) -> uniform.
            p = 1.0 / len(legal)
            return {card: p for card in legal}
        _value, x, _y = entry
        total = float(sum(max(0.0, float(v)) for v in x)) or 1.0
        return {card: max(0.0, float(x[i])) / total for i, card in enumerate(legal)}

    return policy


# ======================================================================
# Match play (win-rate / mean score-diff) against the ground-truth bots
# ======================================================================
def _sample_action(dist: Mapping[int, float], rng: random.Random) -> int:
    cards = list(dist.keys())
    weights = [max(0.0, float(dist[c])) for c in cards]
    total = sum(weights)
    if total <= 0:
        return rng.choice(cards)
    r = rng.random() * total
    upto = 0.0
    for card, w in zip(cards, weights):
        upto += w
        if r <= upto:
            return card
    return cards[-1]


def play_policy_scripted_game(
    policy: PolicyFn,
    *,
    n_cards: int,
    prize_order: list[int],
    opponent_action: Callable[[GameState], int],
    normalized: bool = True,
) -> float:
    """Play one fully-deterministic game and return the candidate's score-diff.

    Every source of randomness is removed: the prize sequence is fixed by
    ``prize_order`` (``prize_order[0]`` is the opening prize), the opponent is a
    deterministic function of the state, and the candidate plays its policy's
    argmax card.  This is the primitive the discrimination test uses to
    *re-execute* an exactly hand-computable outcome rather than read a metric.
    """
    if len(prize_order) != n_cards:
        raise ValueError(f"prize_order must list all {n_cards} prizes, got {prize_order!r}")
    state = GameState.initial(n_cards, current_prize=prize_order[0])
    next_idx = 1
    while not state.done:
        dist = policy(state)
        # Deterministic: highest-probability legal card (ties -> lowest card).
        a = max(state.self_actions, key=lambda c: (dist.get(c, 0.0), -c))
        b = opponent_action(state)
        nxt = prize_order[next_idx] if state.prize_mask else None
        if state.prize_mask:
            next_idx += 1
        state = transition(state, a, b, next_prize=nxt).state
    diff = state.self_score - state.opp_score
    return diff / state.total_prize_mass if normalized else float(diff)


def play_policy_vs_bot(
    policy: PolicyFn,
    bot_type: str,
    *,
    n_cards: int,
    num_games: int = 32,
    seed: int = 1,
) -> dict[str, float]:
    """Play ``policy`` (as player 0) against a named bot over ``num_games``.

    Returns real, computed win-rate / draw-rate / mean & worst score-diff.  The
    policy plays the role of ``PLAYER_0``; the env is the true carry-over game.
    """
    from goofspiel.bots import create_bot

    rng = random.Random(seed)
    diffs: list[int] = []
    wins = draws = 0
    for _ in range(num_games):
        env = GoofspielEnv(num_cards=n_cards, rng=random.Random(rng.randint(0, 2**31 - 1)))
        bot = create_bot(bot_type, seed=rng.randint(0, 2**31 - 1))
        env.reset()
        while not env.done:
            state = state_from_env(env, PLAYER_0)
            self_action = _sample_action(policy(state), rng)
            if self_action not in env.legal_actions(PLAYER_0):
                self_action = rng.choice(env.legal_actions(PLAYER_0))
            opp_action = bot.choose_action(env, PLAYER_1)
            env.step({PLAYER_0: self_action, PLAYER_1: opp_action})
        diff = int(env.scores[PLAYER_0] - env.scores[PLAYER_1])
        diffs.append(diff)
        if diff > 0:
            wins += 1
        elif diff == 0:
            draws += 1
    games = max(1, num_games)
    return {
        "games": float(num_games),
        "win_rate": wins / games,
        "draw_rate": draws / games,
        "mean_score_diff": sum(diffs) / games,
        "worst_score_diff": float(min(diffs)) if diffs else 0.0,
    }


# ======================================================================
# 0.3 (a) full-game exploitability — exact best response over the whole tree
# ======================================================================
def full_game_exploitability(
    policy: PolicyFn,
    *,
    n_cards: int,
    max_n: int = DEFAULT_MAX_FULL_GAME_N,
) -> float | None:
    """Exact full-game exploitability of ``policy`` under the carry-over rule.

    Definition.  The candidate plays the fixed strategy ``policy`` at every one
    of its decision nodes; an unconstrained adversary best-responds to minimise
    the candidate's expected normalised score-diff; chance averages uniformly
    over prize reveals.  The full symmetric game value is 0, so

        exploitability = game_value - worst_case_value(policy) = -worst_case_value

    which is >= 0 for any policy (a best-responder can never do worse than the
    game value).  A Nash policy gives ~0; a weak policy gives a clearly positive
    number.

    Returns ``None`` for ``n_cards > max_n`` rather than silently truncating the
    tree (which would make the figure dishonest).
    """
    if n_cards > max_n:
        return None

    memo: dict[GameState, float] = {}
    policy_cache: dict[GameState, Mapping[int, float]] = {}

    def _policy(state: GameState) -> Mapping[int, float]:
        cached = policy_cache.get(state)
        if cached is None:
            cached = policy(state)
            policy_cache[state] = cached
        return cached

    def _chance_value(state: GameState, a: int, b: int) -> float:
        if state.prize_mask == 0:
            # Final round: next prize is ignored by ``transition``.
            return _br(transition(state, a, b).state)
        prizes = legal_cards(state.prize_mask, state.n)
        return sum(_br(transition(state, a, b, next_prize=p).state) for p in prizes) / len(prizes)

    def _br(state: GameState) -> float:
        if state.done:
            return (state.self_score - state.opp_score) / state.total_prize_mass
        cached = memo.get(state)
        if cached is not None:
            return cached
        dist = _policy(state)
        best: float | None = None
        for b in state.opponent_actions:
            ev = 0.0
            for a, pa in dist.items():
                if pa <= 0.0:
                    continue
                ev += pa * _chance_value(state, a, b)
            if best is None or ev < best:
                best = ev
        result = 0.0 if best is None else best
        memo[state] = result
        return result

    prizes = list(range(1, n_cards + 1))
    worst_case_value = sum(_br(GameState.initial(n_cards, current_prize=p)) for p in prizes) / len(prizes)
    exploit = -worst_case_value
    # Clamp float noise; worst_case_value <= 0 in exact arithmetic.
    return max(0.0, exploit)


# ======================================================================
# 0.3 (b) one-step matrix Nash gap — proxy, valid for any N
# ======================================================================
def one_step_matrix_nash_gap(policy: PolicyFn, states: Iterable[GameState]) -> float:
    """Mean best-response gap of ``policy`` on each state's *immediate* matrix.

    For the immediate self-payoff matrix ``Q`` at a state, with the policy's row
    distribution ``x``, the adversary's best response value is
    ``min_b (x @ Q)[b]``; the matrix game value is ``v*``.  The gap
    ``v* - min_b (x @ Q)[b] >= 0`` measures how exploitable ``x`` is *on this one
    matrix only*.  This is a cheap proxy, explicitly **not** full-game
    exploitability.
    """
    from goofspiel.solver import solve_zero_sum_matrix
    from goofspiel.training.teachers import immediate_q_matrix

    gaps: list[float] = []
    for state in states:
        legal = state.self_actions
        if len(legal) < 1:
            continue
        q, a_cards, b_cards = immediate_q_matrix(state)
        if q.size == 0:
            continue
        dist = policy(state)
        x = [float(dist.get(card, 0.0)) for card in a_cards]
        sx = sum(x) or 1.0
        x = [v / sx for v in x]
        # Adversary best-responds against x on this matrix (column minimises the
        # row player's payoff): value under x = min_b sum_a x[a] * Q[a,b].
        col_values = [sum(x[i] * float(q[i][j]) for i in range(len(a_cards))) for j in range(len(b_cards))]
        row_value_under_x = min(col_values) if col_values else 0.0
        v_star, _row, _col = solve_zero_sum_matrix(q)
        gaps.append(max(0.0, float(v_star) - row_value_under_x))
    return sum(gaps) / len(gaps) if gaps else 0.0


def _reachable_opening_states(n_cards: int) -> list[GameState]:
    """The n opening states (one per possible first prize) — a small, cheap,
    any-N set of states for the one-step proxy."""
    return [GameState.initial(n_cards, current_prize=p) for p in range(1, n_cards + 1)]


# ======================================================================
# Top-level report
# ======================================================================
@dataclass
class CheckpointEvaluation:
    checkpoint: str
    checkpoint_id: str
    training_stage: str
    matchups: dict[str, dict[str, float]] = field(default_factory=dict)
    full_game_exploitability: dict[str, float | None] = field(default_factory=dict)
    one_step_matrix_nash_gap: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def evaluate_checkpoint(
    path: str | Path,
    *,
    device: str = "cpu",
    n_cards: tuple[int, ...] = (5, 7),
    num_games: int = 32,
    seed: int = 1,
    opponents: tuple[str, ...] = ("random", "heuristic", "nash"),
    greedy: bool = True,
    full_game_max_n: int = DEFAULT_MAX_FULL_GAME_N,
) -> CheckpointEvaluation:
    """Load ``path`` and produce a fully computed honest evaluation report."""
    model, metadata = load_model_from_checkpoint(path, device=device)
    policy = robust_policy_fn(model, device=device, greedy=greedy)

    report = CheckpointEvaluation(
        checkpoint=str(path),
        checkpoint_id=str(metadata.get("checkpoint_id", "")),
        training_stage=str(metadata.get("training_stage", "")),
    )
    for n in n_cards:
        for opp in opponents:
            report.matchups[f"N{n}_vs_{opp}"] = play_policy_vs_bot(
                policy, opp, n_cards=n, num_games=num_games, seed=seed
            )
        # Full-game exploitability: honest only for small N.
        report.full_game_exploitability[f"N{n}"] = full_game_exploitability(
            policy, n_cards=n, max_n=full_game_max_n
        )
        # One-step proxy: any N.
        report.one_step_matrix_nash_gap[f"N{n}"] = one_step_matrix_nash_gap(
            policy, _reachable_opening_states(n)
        )
    if any(v is None for v in report.full_game_exploitability.values()):
        report.notes.append(
            f"full_game_exploitability=None for N>{full_game_max_n}: tree enumeration refused "
            f"(use one_step_matrix_nash_gap as the any-N proxy)."
        )
    return report


__all__ = [
    "PolicyFn",
    "load_model_from_checkpoint",
    "robust_policy_fn",
    "uniform_policy_fn",
    "carry_nash_policy_fn",
    "play_policy_scripted_game",
    "play_policy_vs_bot",
    "full_game_exploitability",
    "one_step_matrix_nash_gap",
    "evaluate_checkpoint",
    "CheckpointEvaluation",
]
