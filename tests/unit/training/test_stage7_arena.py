"""Stage7 arena records an HONEST search fallback (no fabricated search bot).

``create_bot`` supports only random/heuristic/nash/nash_carry; there is no
"Search" bot.  Above the exact-Nash cap (``NASH_MAX_N``) the strong slot must
record the truth — ``opponent_requested="nash"`` but
``opponent_effective="heuristic_fallback"`` — rather than pretend a search
result.  This test RE-EXECUTES the cap decision (``_strong_bot_for``) and then
runs a real Stage7 arena at n=13 to confirm the report carries the honest
fallback on every strong row, for both the robust and corrected policies.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_strong_bot_for_reports_honest_cap():
    from goofspiel.bots import NASH_MAX_N
    from goofspiel.training.stages import _strong_bot_for

    # At/under the exact cap the strong slot is genuinely nash.
    req, eff = _strong_bot_for(NASH_MAX_N)
    assert (req, eff) == ("nash", "nash")
    # Above the cap it HONESTLY degrades — never a fabricated search win.
    req, eff = _strong_bot_for(NASH_MAX_N + 1)
    assert req == "nash" and eff == "heuristic_fallback"
    req, eff = _strong_bot_for(13)
    assert req == "nash" and eff == "heuristic_fallback"


def test_stage7_arena_records_honest_search_fallback(tmp_path: Path):
    try:
        __import__("torch")
    except Exception as exc:  # pragma: no cover - depends on local torch install
        pytest.skip(f"torch cannot be imported: {exc}")

    from goofspiel.training.budgets import Stage7Budget
    from goofspiel.training.stages import run_stage7_redteam

    # Minimal arena (1 game × 1 seed) at n=13 so the strong slot is above the
    # exact-Nash cap and must fall back honestly.
    budget = Stage7Budget(
        attack_cases=3,
        correction_steps=2,
        correction_train_cases=3,
        heldout_attack_cases=0,
        arena_games=1,
        arena_seeds=1,
    )
    metrics = run_stage7_redteam(
        out_dir=tmp_path / "s7",
        correction_steps=2,
        n_cards=13,
        seed=5,
        budget=budget,
    )
    report = json.loads((tmp_path / "s7" / "redteam" / "focused_correction_report.json").read_text(encoding="utf-8"))
    arena = report["arena"]
    assert arena is not None, "arena_games>0 must produce an arena block"
    assert arena["n_cards"] == 13

    # The arena-level summary records the honest fallback.
    assert arena["strong_opponent_requested"] == "nash"
    assert arena["strong_opponent_effective"] == "heuristic_fallback"

    # Every strong row (robust AND corrected policy) carries the honest fallback;
    # a genuinely-Nash-capable slot would never appear at n=13.
    for policy_key in ("robust", "corrected"):
        strong_rows = [r for r in arena[policy_key] if r["opponent_slot"] == "strong"]
        assert strong_rows, f"no strong row for {policy_key}"
        for row in strong_rows:
            assert row["opponent_requested"] == "nash"
            assert row["opponent_effective"] == "heuristic_fallback"

    # Workload accounting reflects the arena actually ran.
    assert metrics.metrics["arena_games_played"] == 1.0
