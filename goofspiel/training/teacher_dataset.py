"""Multi-source robust teacher dataset for P2 → P3 (Phase 2.2b).

Before this, `run_stage2_semi_supervised` wrote a `teacher_dataset.jsonl` that
**no one consumed**: P3 (`run_stage3_sft`) ignored the file and re-derived targets
from `_sample_states` + the immediate matrix, and its four "SFT source" metrics
were all the *same* `exact_anchor_count`.

This module produces a teacher dataset from **four genuinely distinct, robust-only
sources**, so P3's four counts measure four different computations and its loss
depends on the file's content.  The sources differ by *algorithm and search
depth*, and none of them uses opponent behaviour (that stays in P5, preserving
`Q_R ⊥ Q_A`):

| Source  | What it computes                                             | Coverage |
|---------|--------------------------------------------------------------|----------|
| `CFR`   | RM+ regret-matching equilibrium of the immediate matrix      | all      |
| `SEARCH`| depth-1 lookahead (folds one ply, immediate-value leaves)    | budget   |
| `EXACT` | full-game recursive Nash (carry solver, folds continuation)  | small N  |
| `PSEUDO`| confident (low-entropy) subset of robust self-labels         | subset   |

`CFR` and `EXACT` genuinely differ (immediate vs full continuation); `SEARCH`
sits strictly between them (one ply of folded continuation); `PSEUDO` is a
confidence-gated self-label subset.  The four sample counts therefore differ by
both algorithm and coverage — an honest four-way split, not one aliased number.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from goofspiel.game import GameState, legal_cards, transition
from goofspiel.solver import solve_with_policy_carry, solve_zero_sum_matrix
from goofspiel.training.data import JsonlStore, TeacherSample, state_record_from_game_state
from goofspiel.training.teachers import immediate_q_matrix

# Distinct robust-only teacher sources.  Deliberately excludes any
# opponent-behaviour source (that is P5's responsibility).
SOURCE_CFR = "CFR"
SOURCE_SEARCH = "SEARCH"
SOURCE_EXACT = "EXACT"
SOURCE_PSEUDO = "PSEUDO"
ROBUST_TEACHER_SOURCES = (SOURCE_CFR, SOURCE_SEARCH, SOURCE_EXACT, SOURCE_PSEUDO)

# Budgets / gates that make the four coverages genuinely differ.
_SEARCH_MAX_REMAINING = 4   # depth-1 lookahead only when the branching is cheap
_EXACT_MAX_N = 6            # full recursive solve only for small games
_PSEUDO_MAX_ENTROPY_FRAC = 0.55  # accept self-label only when clearly concentrated


def _padded_policy(cards: list[int], probs: np.ndarray) -> list[float]:
    """Scatter a policy over legal `cards` into a length-13 vector (card i -> i-1)."""
    out = [0.0] * 13
    for card, prob in zip(cards, probs.tolist()):
        out[card - 1] = float(prob)
    return out


# ----------------------------------------------------------------------------
# Depth-limited robust value/policy (shared engine for CFR and SEARCH).
# ----------------------------------------------------------------------------
def _rm_plus_solve(matrix: np.ndarray, iterations: int = 256) -> tuple[float, np.ndarray]:
    """Regret-matching+ equilibrium of a zero-sum matrix (self is the row player).

    Returns (game value from the row player's view, row policy).  This is the
    CFR-family solver, distinct from the FP64 LP used by the exact recursion.
    """
    rows, cols = matrix.shape
    row_regret = np.zeros(rows)
    col_regret = np.zeros(cols)
    row_sum = np.zeros(rows)
    col_sum = np.zeros(cols)

    def strat(regret: np.ndarray) -> np.ndarray:
        pos = np.maximum(regret, 0.0)
        total = pos.sum()
        return pos / total if total > 1e-12 else np.full_like(regret, 1.0 / len(regret))

    for _ in range(iterations):
        row = strat(row_regret)
        col = strat(col_regret)
        row_sum += row
        col_sum += col
        row_util = matrix @ col
        value = float(row @ row_util)
        col_util = (-matrix).T @ row
        col_value = float(col @ col_util)
        row_regret = np.maximum(row_regret + (row_util - value), 0.0)
        col_regret = np.maximum(col_regret + (col_util - col_value), 0.0)

    row_policy = row_sum / row_sum.sum() if row_sum.sum() > 1e-12 else np.full(rows, 1.0 / rows)
    col_policy = col_sum / col_sum.sum() if col_sum.sum() > 1e-12 else np.full(cols, 1.0 / cols)
    value = float(row_policy @ matrix @ col_policy)
    return value, row_policy


def _immediate_matrix_over_cards(state: GameState) -> tuple[np.ndarray, list[int], list[int]]:
    q, self_cards, opp_cards = immediate_q_matrix(state)
    return q, self_cards, opp_cards


def _remaining_value(state: GameState, depth: int) -> float:
    """Value of the *remaining* rewards from `state` (score-diff / S_N), robust.

    Past banked score is intentionally excluded: it is a constant offset that
    does not change the equilibrium policy, so the recursion stays clean.  At
    `depth == 0` the immediate matrix value is used as the leaf bootstrap (this
    is exactly SM-MCTS/GT-CFR-style truncated search, not a full solve).
    """
    if state.done or state.prize_mask == 0 and depth <= 0:
        # Terminal or leaf at the final round: value is just this round's matrix.
        q, _, _ = _immediate_matrix_over_cards(state)
        value, _ = _rm_plus_solve(q)
        return value
    q, self_cards, opp_cards = _immediate_matrix_over_cards(state)
    if depth <= 0:
        value, _ = _rm_plus_solve(q)
        return value
    # Fold one ply: payoff[a,b] = immediate reward + E_chance[value(child, depth-1)].
    remaining_prizes = legal_cards(state.prize_mask, state.n)
    matrix = np.array(q, dtype=np.float64, copy=True)
    for i, a in enumerate(self_cards):
        for j, b in enumerate(opp_cards):
            if remaining_prizes:
                child_vals = []
                for np_prize in remaining_prizes:
                    child = transition(state, a, b, next_prize=np_prize).state
                    child_vals.append(0.0 if child.done else _remaining_value(child, depth - 1))
                matrix[i, j] += float(np.mean(child_vals))
    value, _ = _rm_plus_solve(matrix)
    return value


def _search_policy(state: GameState, depth: int = 1) -> tuple[float, np.ndarray, list[int]]:
    """Depth-limited robust root policy (folds `depth` plies of continuation)."""
    q, self_cards, opp_cards = _immediate_matrix_over_cards(state)
    matrix = np.array(q, dtype=np.float64, copy=True)
    remaining_prizes = legal_cards(state.prize_mask, state.n)
    if depth > 0 and remaining_prizes:
        for i, a in enumerate(self_cards):
            for j, b in enumerate(opp_cards):
                child_vals = []
                for np_prize in remaining_prizes:
                    child = transition(state, a, b, next_prize=np_prize).state
                    child_vals.append(0.0 if child.done else _remaining_value(child, depth - 1))
                matrix[i, j] += float(np.mean(child_vals))
    value, row_policy = _rm_plus_solve(matrix)
    return value, row_policy, self_cards


# ----------------------------------------------------------------------------
# Per-source label builders.  Each returns a TeacherSample or None (not applicable).
# ----------------------------------------------------------------------------
def cfr_label(state: GameState) -> TeacherSample | None:
    """RM+ regret-matching equilibrium of the immediate matrix (depth 0)."""
    q, self_cards, _ = _immediate_matrix_over_cards(state)
    value, row_policy = _rm_plus_solve(np.asarray(q, dtype=np.float64))
    return TeacherSample(
        sample_id=f"{SOURCE_CFR}:{state.n}:{state.self_mask}:{state.opp_mask}:{state.prize_mask}:{state.current_prize}:{state.carry_pool}",
        state=state_record_from_game_state(state),
        teacher_q=np.asarray(q, dtype=np.float64).tolist(),
        teacher_policy=_padded_policy(self_cards, row_policy),
        teacher_value=float(value),
        teacher_source=SOURCE_CFR,
        teacher_confidence=1.0,
    )


def search_label(state: GameState) -> TeacherSample | None:
    """Depth-1 lookahead robust policy — budget-gated to small remaining hands."""
    if len(legal_cards(state.self_mask, state.n)) > _SEARCH_MAX_REMAINING:
        return None
    value, row_policy, self_cards = _search_policy(state, depth=1)
    q, _, _ = _immediate_matrix_over_cards(state)
    return TeacherSample(
        sample_id=f"{SOURCE_SEARCH}:{state.n}:{state.self_mask}:{state.opp_mask}:{state.prize_mask}:{state.current_prize}:{state.carry_pool}",
        state=state_record_from_game_state(state),
        teacher_q=np.asarray(q, dtype=np.float64).tolist(),
        teacher_policy=_padded_policy(self_cards, row_policy),
        teacher_value=float(value),
        teacher_source=SOURCE_SEARCH,
        teacher_confidence=1.0,
    )


@lru_cache(maxsize=None)
def _exact_policy_map(n: int):
    """Full-game Nash policy map from the recursive carry solver (cached per N)."""
    result = solve_with_policy_carry(n, force=True)
    return result.policy_map or {}


def exact_label(state: GameState) -> TeacherSample | None:
    """Full-game recursive Nash (carry solver) — small N only, policy-map lookup."""
    if state.n > _EXACT_MAX_N:
        return None
    policy_map = _exact_policy_map(state.n)
    r_mask = state.prize_mask | (1 << (state.current_prize - 1))
    key = (state.self_mask, state.opp_mask, r_mask, state.carry_pool, state.current_prize)
    entry = policy_map.get(key)
    if entry is None:
        return None
    value, policy_a, _policy_b = entry
    self_cards = legal_cards(state.self_mask, state.n)
    q, _, _ = _immediate_matrix_over_cards(state)
    # policy_a is indexed over the legal cards in ascending order (its length is
    # the number of legal cards), matching `self_cards`.
    probs = np.asarray(policy_a, dtype=np.float64)
    if probs.shape[0] != len(self_cards):  # pragma: no cover - solver contract guard
        return None
    total = probs.sum()
    probs = probs / total if total > 1e-12 else np.full(len(self_cards), 1.0 / len(self_cards))
    return TeacherSample(
        sample_id=f"{SOURCE_EXACT}:{state.n}:{state.self_mask}:{state.opp_mask}:{state.prize_mask}:{state.current_prize}:{state.carry_pool}",
        state=state_record_from_game_state(state),
        teacher_q=np.asarray(q, dtype=np.float64).tolist(),
        teacher_policy=_padded_policy(self_cards, probs),
        teacher_value=float(value),
        teacher_source=SOURCE_EXACT,
        teacher_confidence=1.0,
    )


def pseudo_label(state: GameState) -> TeacherSample | None:
    """Confident robust self-label: keep the immediate equilibrium only when it is
    clearly concentrated (low normalized entropy).  This is the subset of states
    where the robust decision is unambiguous — a high-confidence self-label — and
    is deliberately narrower than CFR's full coverage so the counts differ."""
    q, self_cards, _ = _immediate_matrix_over_cards(state)
    value, row_policy = _rm_plus_solve(np.asarray(q, dtype=np.float64))
    k = len(self_cards)
    if k <= 1:
        entropy_frac = 0.0
    else:
        probs = np.clip(row_policy, 1e-12, 1.0)
        entropy = float(-(probs * np.log(probs)).sum())
        entropy_frac = entropy / math.log(k)
    if entropy_frac > _PSEUDO_MAX_ENTROPY_FRAC:
        return None  # too uncertain to be a confident self-label
    return TeacherSample(
        sample_id=f"{SOURCE_PSEUDO}:{state.n}:{state.self_mask}:{state.opp_mask}:{state.prize_mask}:{state.current_prize}:{state.carry_pool}",
        state=state_record_from_game_state(state),
        teacher_q=np.asarray(q, dtype=np.float64).tolist(),
        teacher_policy=_padded_policy(self_cards, row_policy),
        teacher_value=float(value),
        teacher_source=SOURCE_PSEUDO,
        teacher_confidence=float(1.0 - entropy_frac),
    )


_SOURCE_BUILDERS = {
    SOURCE_CFR: cfr_label,
    SOURCE_SEARCH: search_label,
    SOURCE_EXACT: exact_label,
    SOURCE_PSEUDO: pseudo_label,
}


def build_teacher_dataset(states: list[GameState], store: JsonlStore) -> dict[str, int]:
    """Label every state with each applicable source and append to `store`.

    Returns a per-source accepted-count dict.  A state may yield up to four
    samples (one per source), and the four totals differ because the sources
    have different applicability gates (small-N for EXACT, budget for SEARCH,
    confidence for PSEUDO, all-states for CFR).
    """
    counts = {src: 0 for src in ROBUST_TEACHER_SOURCES}
    for state in states:
        if state.done:
            continue
        for src, builder in _SOURCE_BUILDERS.items():
            sample = builder(state)
            if sample is not None:
                store.append(sample)
                counts[src] += 1
    return counts
