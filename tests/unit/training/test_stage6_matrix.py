"""Stage6 statistical matrix: block/paired bootstrap + workload accounting.

Runs Stage6 at a small QUICK-like budget (multi-seed, multi-sequence, so the CI
is non-degenerate) and:
  1. asserts every statistical column the matrix promises is present on all 9
     ordered pairs, with self-consistent values (rates in range, CI ordering),
     the self-play diagonal flagged, and ``raw_games`` ≠ ``bootstrap_blocks``
     (delta #2/#15: a block is a ``(seed, prize_sequence)`` group, so with
     games_per_block > 1 the block count is strictly below the raw game count);
  2. RE-EXECUTES the BLOCK bootstrap — the report stores each cell's per-block
     game rows, so we rebuild the block means from real play and reproduce the
     exact ci_low/ci_high the report recorded from the SAME deterministic
     ``_bootstrap_ci`` seed the runner uses.  This proves the interval is a
     block-resample re-run fact, not an IID-over-games literal;
  3. RE-EXECUTES the paired seat-symmetrized statistic (delta #3) straight from
     the two ordered cells' stored block rows — no re-play — proving the
     ``mean(diff_A_vs_B, −diff_B_vs_A)`` construction is recomputable.

The legacy 9-row ``cross_play`` block and its ``crossplay_seed_base`` invariant
are unaffected (a separate test re-executes those); here we exercise the
statistical ``matrix`` / ``paired_matrix`` blocks.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from goofspiel.training import TrainingCoordinator, TrainingRunConfig

MATRIX_COLUMNS = {
    "row_agent", "col_agent", "row_role", "col_role",
    "self_play", "relationship",
    "games", "raw_games", "bootstrap_blocks", "blocks",
    "seeds", "seeds_used", "prize_sequences",
    "win_rate", "draw_rate", "mean_score_diff", "std", "median",
    "worst_seed", "worst_seed_mean",
    "ci_low", "ci_high", "ci_halfwidth",
    "block_bootstrap_ci_low", "block_bootstrap_ci_high", "block_rows",
}


def test_stage6_matrix_columns_present_and_ci_reproducible(tmp_path: Path):
    try:
        __import__("torch")
    except Exception as exc:  # pragma: no cover - depends on local torch install
        pytest.skip(f"torch cannot be imported: {exc}")

    from goofspiel.training.stages import _bootstrap_ci

    # A small multi-seed / multi-sequence budget so CIs are non-degenerate but the
    # test stays fast.  We stash the budget in extra["budgets"] the same way the
    # CLI does, so the coordinator resolves and threads it into Stage6.
    from goofspiel.training.budgets import resolve_budgets
    from dataclasses import asdict

    budgets = resolve_budgets(
        profile="SMOKE",
        steps_fallback=10,
        overrides={"stage6_games_per_matchup": 2, "stage6_seeds": 3, "stage6_prize_sequences": 2},
    )
    result = TrainingCoordinator(
        TrainingRunConfig(
            artifact_dir=str(tmp_path / "stage6"),
            stage="stage6_league",
            n_cards=3,
            seed=1,
            extra={"budgets": asdict(budgets)},
        )
    ).run()
    assert result["ok"] is True

    report = json.loads((tmp_path / "stage6" / "league" / "league_report.json").read_text(encoding="utf-8"))
    matrix = report["matrix"]
    assert len(matrix) == 9, "3 agents × 3 agents = 9 ordered matrix cells"

    crossplay_seed_base = int(report["crossplay_seed_base"])
    n_agents = 3
    # Re-derive the ordered agent list exactly as the runner does (sorted by id).
    agent_ids = sorted({row["row_agent"] for row in report["cross_play"]})
    assert len(agent_ids) == n_agents
    index_of = {aid: i for i, aid in enumerate(agent_ids)}

    for row in matrix:
        # 1. Every promised column is present.
        assert MATRIX_COLUMNS.issubset(row.keys()), f"missing columns: {MATRIX_COLUMNS - set(row.keys())}"
        # Self-consistency of the aggregates.
        assert row["games"] == 2 * 3 * 2, "games == games_per_matchup × seeds × prize_sequences"
        assert row["seeds"] == 3 and row["prize_sequences"] == 2
        assert 0.0 <= row["win_rate"] <= 1.0 and 0.0 <= row["draw_rate"] <= 1.0
        assert row["ci_low"] <= row["ci_high"]
        assert row["ci_halfwidth"] == pytest.approx((row["ci_high"] - row["ci_low"]) / 2.0)
        assert row["std"] >= 0.0
        assert row["worst_seed_mean"] <= row["mean_score_diff"] + 1e-9
        # delta #15: raw_games and bootstrap_blocks are BOTH reported and, with
        # games_per_block > 1, they DIFFER (blocks = seeds × prize_sequences = 6,
        # raw games = 12).  This is the whole point of the block bootstrap.
        assert row["raw_games"] == 2 * 3 * 2
        assert row["bootstrap_blocks"] == 3 * 2
        assert row["blocks"] == 3 * 2
        assert row["raw_games"] != row["bootstrap_blocks"], "acc#4: raw_games must differ from blocks when games/block>1"
        # The canonical block-bootstrap names alias the legacy ci_* fields.
        assert row["block_bootstrap_ci_low"] == pytest.approx(row["ci_low"])
        assert row["block_bootstrap_ci_high"] == pytest.approx(row["ci_high"])
        # delta #3 (acc#5): the diagonal is self-play, off-diagonal competitive.
        expect_self = row["row_agent"] == row["col_agent"]
        assert row["self_play"] is expect_self
        assert row["relationship"] == ("self_play" if expect_self else "competitive")
        # Each block row holds exactly games_per_matchup games and a matching mean.
        assert len(row["block_rows"]) == row["bootstrap_blocks"]
        for br in row["block_rows"]:
            assert len(br["games"]) == 2
            assert br["block_mean"] == pytest.approx(sum(br["games"]) / len(br["games"]))

    # The diagonal (3 self-play) and off-diagonal (6 competitive) partition the 9.
    assert sum(1 for r in matrix if r["self_play"]) == 3
    assert sum(1 for r in matrix if not r["self_play"]) == 6

    # 2. RE-EXECUTE the BLOCK bootstrap for one cell (delta #2).  A block is a
    #    (seed, prize_sequence) group; the bootstrap resamples WHOLE blocks, which
    #    for equal-size blocks is exactly a bootstrap over the per-block MEANS.  We
    #    rebuild the diffs by replaying the same primitive over the same CRN
    #    schedule, group them into blocks, and feed the block means to
    #    `_bootstrap_ci` with the SAME canonical-pair seed to reproduce ci_low/high.
    from goofspiel.training.stages import _CheckpointPolicy, _play_policy_match_seq, _prize_sequences

    ckpts = report["agent_checkpoints"]
    # Pick an off-diagonal pair (row != col) so the diffs are a real distribution.
    target = next(r for r in matrix if r["row_agent"] != r["col_agent"])
    ri, ci = index_of[target["row_agent"]], index_of[target["col_agent"]]
    lo_idx, hi_idx = (ri, ci) if ri <= ci else (ci, ri)
    pair_key = lo_idx * n_agents + hi_idx
    sequences = _prize_sequences(3, seed=crossplay_seed_base + pair_key, k=2)
    row_pol = _CheckpointPolicy(ckpts[target["row_agent"]], temperature=0.5)
    col_pol = _CheckpointPolicy(ckpts[target["col_agent"]], temperature=0.5)
    diffs: list[float] = []
    block_means: list[float] = []
    for s in range(3):
        for q in range(2):
            block: list[float] = []
            for g in range(2):
                cell_seed = crossplay_seed_base + pair_key * 100000 + s * 1000 + q * 100 + g
                block.append(
                    _play_policy_match_seq(row_pol, col_pol, n_cards=3, prize_order=sequences[q], seed=cell_seed)
                )
            diffs.extend(block)
            block_means.append(sum(block) / len(block))
    # The re-executed aggregate reproduces the reported mean, and the BLOCK
    # bootstrap (over block means) reproduces the reported CI exactly.
    assert sum(diffs) / len(diffs) == pytest.approx(target["mean_score_diff"]), "matrix mean is not reproducible play"
    lo, hi = _bootstrap_ci(block_means, seed=crossplay_seed_base + pair_key)
    assert lo == pytest.approx(target["ci_low"]), "block-bootstrap ci_low is not reproducible"
    assert hi == pytest.approx(target["ci_high"]), "block-bootstrap ci_high is not reproducible"
    # The stored block rows carry the SAME block means we just recomputed (acc#2:
    # a (seed, prize_sequence) block was never split by the bootstrap).
    stored_block_means = [br["block_mean"] for br in target["block_rows"]]
    assert stored_block_means == pytest.approx(block_means)

    # 3. acc#3: the paired seat-reversal statistic is recomputable from the stored
    #    block rows.  For the unordered pair, pair cell (lo,hi) with (hi,lo)
    #    game-for-game on their shared CRN and average mean(d, -d_rev)/... — this is
    #    the mean(diff_A_vs_B, -diff_B_vs_A) construction (delta #3).  We read the
    #    two ordered cells' block rows straight from the report; no re-play needed.
    a_id, b_id = agent_ids[lo_idx], agent_ids[hi_idx]
    fwd_row = next(r for r in matrix if r["row_agent"] == a_id and r["col_agent"] == b_id)
    rev_row = next(r for r in matrix if r["row_agent"] == b_id and r["col_agent"] == a_id)
    fwd_by_key = {(br["seed"], br["prize_sequence"]): br["games"] for br in fwd_row["block_rows"]}
    rev_by_key = {(br["seed"], br["prize_sequence"]): br["games"] for br in rev_row["block_rows"]}
    paired_games: list[float] = []
    for key in sorted(set(fwd_by_key) & set(rev_by_key)):
        fg, rg = fwd_by_key[key], rev_by_key[key]
        paired_games.extend(0.5 * (fg[g] - rg[g]) for g in range(min(len(fg), len(rg))))
    paired = next(
        p for p in report["paired_matrix"]
        if {p["agent_a"], p["agent_b"]} == {a_id, b_id}
    )
    assert sum(paired_games) / len(paired_games) == pytest.approx(paired["paired_mean_score_diff"]), (
        "paired seat-symmetrized statistic is not recomputable from the stored block rows"
    )
    assert paired["paired_block_bootstrap_ci_low"] <= paired["paired_block_bootstrap_ci_high"]
    assert paired["raw_games"] == len(paired_games)

    # The workload block records the same statistical budget + block accounting.
    workload = report["workload"]
    assert workload["games_per_matchup"] == 2 and workload["seeds"] == 3 and workload["prize_sequences"] == 2
    assert workload["total_games"] == 9 * 2 * 3 * 2
    # delta #15 run-level accounting: raw_games vs bootstrap_blocks, and the
    # ordered-matchup partition into self-play + competitive.
    assert workload["raw_games"] == 9 * 2 * 3 * 2
    assert workload["bootstrap_blocks"] == 9 * 3 * 2
    assert workload["ordered_matchups"] == 9
    assert workload["self_play_matchups"] == 3
    assert workload["competitive_matchups"] == 6
    # delta #4: fixed budget on every profile — never early-stops.
    assert workload["sequential_ci_stop"] is False


# --------------------------------------------------------------------------
# Focused, torch-free unit facts about the block bootstrap primitive itself
# (acceptance #1 determinism, #2 whole-block resampling).  These construct the
# inputs directly so they exercise the statistic, not the pipeline.
# --------------------------------------------------------------------------


def test_acc1_block_bootstrap_is_deterministic_given_a_fixed_seed():
    """#1: the block bootstrap (``_bootstrap_ci`` over per-block means) is fully
    determined by (values, seed): the same inputs reproduce byte-identical bounds
    on every call, and the seed genuinely drives the resampling (it is not
    ignored)."""
    from goofspiel.training.stages import _bootstrap_ci

    block_means = [0.5, -1.0, 2.0, 0.0, -0.5, 1.5, -2.0, 0.25]
    fixed = _bootstrap_ci(block_means, seed=1234)
    for _ in range(5):
        assert _bootstrap_ci(block_means, seed=1234) == fixed, "must be deterministic for a fixed seed"
    # The seed is actually consumed: over a spread of seeds on a continuous input
    # (so percentile bounds are not forced to coincide), the CI is not constant.
    continuous = [i * 0.137 - 3.0 for i in range(40)]
    results = {_bootstrap_ci(continuous, seed=s) for s in range(20)}
    assert len(results) > 1, "the bootstrap must depend on its seed, not ignore it"


def test_acc2_bootstrap_resamples_whole_blocks_not_individual_games():
    """#2: resampling WHOLE (seed, prize_sequence) blocks is not the same as
    resampling individual games.  We build blocks whose per-block means have a
    much narrower spread than the raw games, so a block bootstrap of the block
    means yields a strictly tighter interval than an IID bootstrap over all games
    with the same seed — proving the block structure is respected, not flattened.
    """
    from goofspiel.training.stages import _bootstrap_ci

    # 4 blocks × 4 games.  Inside each block the games swing widely (±10) but each
    # block's MEAN is ~0, so block means cluster tightly while raw games do not.
    blocks = [
        [10.0, -10.0, 10.0, -10.0],
        [-10.0, 10.0, -10.0, 10.0],
        [10.0, -10.0, 10.0, -10.0],
        [-10.0, 10.0, -10.0, 10.0],
    ]
    block_means = [sum(b) / len(b) for b in blocks]          # all 0.0
    all_games = [g for b in blocks for g in b]               # wide ±10 spread
    block_ci = _bootstrap_ci(block_means, seed=7)
    iid_ci = _bootstrap_ci(all_games, seed=7)
    block_hw = (block_ci[1] - block_ci[0]) / 2.0
    iid_hw = (iid_ci[1] - iid_ci[0]) / 2.0
    assert block_hw < iid_hw, (
        "a block bootstrap over block means must not equal an IID bootstrap over "
        "games — the CRN block correlation has to be preserved"
    )
