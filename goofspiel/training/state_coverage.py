"""Reachable-state sampling and coverage accounting for P1/P3 (Phase 2.1).

The original `_sample_states` (`stages.py`) produced only full-hand, score-0:0,
round-1, carry-0 *opening* states.  P1/P3 therefore never saw endgames, score
crises, sustained carry, or asymmetric remaining hands — exactly the regimes
where the model was observed to misplay (e.g. failing to protect a large carried
stake).  No number of gradient steps can teach a state that is never sampled.

This module fixes the *data*: it samples genuinely **reachable** states (produced
only via `transition`, never hand-built into an illegal configuration) spanning
the game, and it makes coverage a **first-class artifact** — labelled buckets and
histograms a training report can be audited against — instead of a single opaque
`teacher_samples=N`.

Reachability facts used below (all provable from `game/state.py`):
- Each round both players play exactly one card, so `|self_mask| == |opp_mask|`
  always; "asymmetric hand" therefore means different card *sets* of equal size,
  which arises after any non-tie round.
- `carry_pool` grows only through non-final ties (both play the same card).
- `self_score`/`opp_score` are sums of stakes actually won.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from goofspiel.game import GameState, transition
from goofspiel.game.state import full_mask, legal_cards

# ----------------------------------------------------------------------------
# Coverage buckets
# ----------------------------------------------------------------------------
BUCKETS = (
    "OPENING",
    "MIDGAME",
    "ENDGAME",
    "HIGH_CARRY",
    "MUST_WIN",
    "MUST_NOT_LOSE",
    "SCORE_AHEAD",
    "SCORE_BEHIND",
    "ASYMMETRIC_HAND",
)


def _popcount(mask: int) -> int:
    return int(mask).bit_count()


def classify_state(state: GameState) -> set[str]:
    """Return the set of coverage buckets a (non-terminal) state belongs to.

    A state can occupy several buckets at once (e.g. an ENDGAME state can also be
    MUST_WIN and ASYMMETRIC_HAND); the buckets are descriptive facets, not a
    partition.  Definitions are chosen to be interpretable and directly checkable
    against the state fields.
    """
    tags: set[str] = set()
    n = state.n
    cards_left = _popcount(state.self_mask)
    stake = state.current_prize + state.carry_pool
    diff = state.self_score - state.opp_score

    is_opening = (
        state.round_index == 1
        and state.self_mask == full_mask(n)
        and state.opp_mask == full_mask(n)
        and state.self_score == 0
        and state.opp_score == 0
        and state.carry_pool == 0
    )
    # Final prize on the board (prize_mask == 0) or only a small tail of cards
    # remains: this is the endgame where crises are decided.
    is_endgame = state.prize_mask == 0 or cards_left <= max(1, n // 4)

    if is_opening:
        tags.add("OPENING")
    if is_endgame:
        tags.add("ENDGAME")
    if not is_opening and not is_endgame:
        tags.add("MIDGAME")

    # Sustained carry: the carried pool is at least as large as the prize on the
    # board, so the stake is dominated by history — the regime the model missed.
    if state.carry_pool > 0 and state.carry_pool >= state.current_prize:
        tags.add("HIGH_CARRY")

    if diff > 0:
        tags.add("SCORE_AHEAD")
    if diff < 0:
        tags.add("SCORE_BEHIND")

    if state.self_mask != state.opp_mask:
        tags.add("ASYMMETRIC_HAND")

    # MUST_WIN / MUST_NOT_LOSE are only well-defined on the final round, where the
    # current stake alone flips the game outcome:
    #   final round, self behind or level (-stake < diff <= 0) -> must win the last
    #   prize or lose the game;
    #   final round, self ahead by less than the stake (0 <= diff < stake) -> must
    #   not lose it or fall behind.
    if state.prize_mask == 0 and not state.done:
        if -stake < diff <= 0:
            tags.add("MUST_WIN")
        if 0 <= diff < stake:
            tags.add("MUST_NOT_LOSE")

    return tags


# ----------------------------------------------------------------------------
# Reachable-state generators (everything routed through `transition`)
# ----------------------------------------------------------------------------
def _outcome_actions(state: GameState, outcome: str) -> tuple[int, int]:
    """Pick a legal (self, opp) action pair realizing the desired round outcome.

    'tie'  -> both play the lowest legal card (identical -> tie);
    'self' -> self plays its highest, opp its lowest (self wins unless 1 card);
    'opp'  -> opp plays its highest, self its lowest (opp wins unless 1 card).
    With a single card left, both are forced to play it (a tie), which the caller
    tolerates.
    """
    self_cards = legal_cards(state.self_mask, state.n)
    opp_cards = legal_cards(state.opp_mask, state.n)
    if outcome == "tie":
        # Prefer a card both still hold so the masks stay symmetric on a tie.
        common = sorted(set(self_cards) & set(opp_cards))
        card = common[0] if common else self_cards[0]
        opp_card = card if card in opp_cards else opp_cards[0]
        return card, opp_card
    if outcome == "self":
        return max(self_cards), min(opp_cards)
    if outcome == "opp":
        return min(self_cards), max(opp_cards)
    raise ValueError(f"unknown outcome {outcome!r}")


def _rollout(
    n: int,
    prize_order: list[int],
    outcomes: list[str],
) -> list[GameState]:
    """Play a scripted game and return every non-terminal state visited.

    `prize_order` is the exact sequence of prizes revealed (first entry is the
    opening `current_prize`).  `outcomes[i]` scripts round i.  The rollout stops
    when the game ends or the script/prizes are exhausted.
    """
    state = GameState.initial(n, current_prize=prize_order[0])
    visited = [state]
    for i, outcome in enumerate(outcomes):
        if state.done:
            break
        next_prize = prize_order[i + 1] if i + 1 < len(prize_order) else None
        self_action, opp_action = _outcome_actions(state, outcome)
        state = transition(state, self_action, opp_action, next_prize=next_prize).state
        if not state.done:
            visited.append(state)
    return visited


def _random_reachable_states(n: int, rng: random.Random, games: int) -> list[GameState]:
    """Collect intermediate states from fully random reachable games."""
    out: list[GameState] = []
    for _ in range(games):
        prizes = list(range(1, n + 1))
        rng.shuffle(prizes)
        state = GameState.initial(n, current_prize=prizes[0])
        idx = 1
        while not state.done:
            out.append(state)
            self_action = rng.choice(state.self_actions)
            opp_action = rng.choice(state.opponent_actions)
            next_prize = prizes[idx] if idx < len(prizes) else None
            idx += 1
            state = transition(state, self_action, opp_action, next_prize=next_prize).state
    return out


def _high_carry_states(n: int, rng: random.Random) -> list[GameState]:
    """Force early non-final ties to accumulate a large carried stake.

    Prizes are revealed high-to-low so the carry (sum of early large prizes)
    dominates the small prize currently on the board -> HIGH_CARRY.
    """
    prize_order = list(range(n, 0, -1))  # n, n-1, ..., 1
    outcomes = ["tie"] * (n - 1)
    return _rollout(n, prize_order, outcomes)


def _must_win_states(n: int, rng: random.Random) -> list[GameState]:
    """Drive self behind, then reach the final round where it must win.

    Opp wins the earliest round(s) by a margin smaller than the final stake, so
    at the final round `-stake < diff <= 0`.
    """
    # Reveal a small prize first (opp wins it -> small deficit), a large prize
    # last (large final stake) so the deficit is recoverable only by winning it.
    prize_order = [1] + list(range(2, n + 1))  # last prize is n (largest)
    outcomes = ["opp"] + ["tie"] * (n - 2)  # opp wins round 1, ties afterwards
    return _rollout(n, prize_order, outcomes)


def _must_not_lose_states(n: int, rng: random.Random) -> list[GameState]:
    """Drive self ahead by less than the final stake, then reach the final round.

    Self wins the earliest (small) round; the final prize is the largest, so at
    the final round `0 <= diff < stake` and self must not lose it.
    """
    prize_order = [1] + list(range(2, n + 1))  # last prize is n (largest)
    outcomes = ["self"] + ["tie"] * (n - 2)  # self wins round 1, ties afterwards
    return _rollout(n, prize_order, outcomes)


def _score_gap_states(n: int, rng: random.Random) -> list[GameState]:
    """A lopsided mid-game producing both SCORE_AHEAD and SCORE_BEHIND views."""
    prize_order = list(range(1, n + 1))
    ahead = _rollout(n, prize_order, ["self"] * (n - 1))
    behind = _rollout(n, prize_order, ["opp"] * (n - 1))
    return ahead + behind


def _midgame_states(n: int, rng: random.Random) -> list[GameState]:
    """States drawn from the middle of a mixed reachable game.

    Guarantees a MIDGAME representative even in a very small batch at large N,
    where the crisis rollouts otherwise crowd out the interior of the game.  The
    scripted outcomes alternate so the state is neither an opening nor an endgame
    and the hands are asymmetric.
    """
    prize_order = list(range(1, n + 1))
    outcomes = [("self" if i % 2 == 0 else "opp") for i in range(n - 1)]
    visited = _rollout(n, prize_order, outcomes)
    # Return the interior slice (drop the opening and the final states) ordered
    # from the middle outward, so the first element is a canonical mid-game state.
    interior = visited[1:-1] if len(visited) > 2 else visited
    mid = len(interior) // 2
    return interior[mid:] + interior[:mid]


@dataclass(frozen=True)
class CoverageReport:
    total: int
    bucket_counts: dict[str, int]
    histograms: dict[str, dict[int, int]]

    def as_metrics(self) -> dict[str, float]:
        """Flatten to scalar StageMetrics entries (bucket counts only)."""
        return {f"coverage_bucket_{name.lower()}": float(count) for name, count in self.bucket_counts.items()}

    def missing_buckets(self) -> list[str]:
        return [name for name in BUCKETS if self.bucket_counts.get(name, 0) == 0]


def coverage_report(states: list[GameState]) -> CoverageReport:
    """Compute bucket counts + audit histograms over a batch of states."""
    bucket_counts = {name: 0 for name in BUCKETS}
    hist_keys = ("round_index", "carry", "score_diff", "remaining_cards", "stake", "hand_asymmetry")
    histograms: dict[str, dict[int, int]] = {key: {} for key in hist_keys}

    for state in states:
        for tag in classify_state(state):
            bucket_counts[tag] += 1
        hand_asymmetry = _popcount(state.self_mask ^ state.opp_mask)
        values = {
            "round_index": state.round_index,
            "carry": state.carry_pool,
            "score_diff": state.self_score - state.opp_score,
            "remaining_cards": _popcount(state.self_mask),
            "stake": state.current_prize + state.carry_pool,
            "hand_asymmetry": hand_asymmetry,
        }
        for key, value in values.items():
            histograms[key][value] = histograms[key].get(value, 0) + 1

    return CoverageReport(total=len(states), bucket_counts=bucket_counts, histograms=histograms)


def sample_reachable_states(batch_size: int, *, n: int, step: int, seed: int = 0) -> list[GameState]:
    """Return `batch_size` reachable states spanning every coverage bucket.

    Deterministic in `(n, step, seed)`.  The batch always contains at least one
    representative of each crisis bucket (high carry, must-win, must-not-lose,
    score ahead/behind, asymmetric hand, endgame) constructed by scripted
    reachable rollouts, with the remainder filled by random reachable games.  A
    tiny batch is padded by cycling the guaranteed pool so coverage never drops.
    """
    rng = random.Random((seed * 1_000_003) ^ (step * 31 + n))

    # Build one scripted-rollout list per crisis regime, then INTERLEAVE them
    # round-robin so that even a batch far smaller than the full pool still lands
    # at least one representative of every bucket before it is truncated.
    opening = [GameState.initial(n, current_prize=rng.choice(list(range(1, n + 1))))]  # OPENING
    sources: list[list[GameState]] = [opening]
    if n >= 2:
        sources.append(_must_win_states(n, rng))       # MUST_WIN (final round of its rollout)
        sources.append(_must_not_lose_states(n, rng))  # MUST_NOT_LOSE
        sources.append(_high_carry_states(n, rng))     # HIGH_CARRY
        sources.append(_score_gap_states(n, rng))      # SCORE_AHEAD / SCORE_BEHIND / ENDGAME
        sources.append(_midgame_states(n, rng))        # MIDGAME (interior, asymmetric)
    # Put the outcome-deciding state of each scripted rollout first (crisis buckets
    # live on the final states), then the earlier states, so truncation keeps them.
    ordered_sources = [list(reversed(src)) for src in sources]
    guaranteed: list[GameState] = []
    for column in range(max((len(src) for src in ordered_sources), default=0)):
        for src in ordered_sources:
            if column < len(src):
                guaranteed.append(src[column])

    # Fill the rest with random reachable games (covers MIDGAME/ENDGAME/asymmetry
    # densely and adds distributional variety).
    needed = max(batch_size - len(guaranteed), 0)
    random_games = max(2, (needed // max(n - 1, 1)) + 2)
    pool = guaranteed + _random_reachable_states(n, rng, random_games)

    if not pool:  # pragma: no cover - n>=2 always yields states
        pool = [GameState.initial(n, current_prize=1)]

    if len(pool) >= batch_size:
        # Keep every guaranteed crisis state, then sample the remainder from the
        # random tail so the guaranteed buckets are never dropped.
        chosen = list(guaranteed)
        tail = pool[len(guaranteed):]
        rng.shuffle(tail)
        chosen += tail[: max(batch_size - len(chosen), 0)]
        # If the guaranteed pool alone already exceeds batch_size, truncate but
        # keep the leading opening + crisis coverage (they are ordered first).
        return chosen[:batch_size]

    # Batch larger than the pool: cycle the pool to reach batch_size.
    return [pool[i % len(pool)] for i in range(batch_size)]
