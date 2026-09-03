"""Budget/profile resolver precedence (Step 1 of the Stage6/7/eval upgrade).

Pure-python — no torch, no stage execution.  Pins the four-tier precedence the
resolver promises, per field:

    1. an explicit per-stage override (non-None flag)  — highest
    2. the profile preset
    3. the ``--steps`` fallback (ONLY for stage4_steps / stage5_sessions)
    4. the dataclass default (== the SMOKE preset)      — lowest

Every assertion RE-EXECUTES ``resolve_budgets`` and reads the returned tree;
nothing trusts a stored constant.
"""
from __future__ import annotations

from goofspiel.training.budgets import (
    PROFILE_FULL,
    PROFILE_QUICK,
    PROFILE_SMOKE,
    resolve_budgets,
)


def test_resolve_budgets_precedence():
    # ---- Tier 4: no profile, no overrides -> SMOKE defaults --------------------
    smoke = resolve_budgets(profile=None, steps_fallback=10, overrides={})
    assert smoke.profile == PROFILE_SMOKE
    assert (smoke.stage6.games_per_matchup, smoke.stage6.seeds, smoke.stage6.prize_sequences) == (1, 1, 1)
    assert smoke.stage7.attack_cases == 3
    assert (smoke.evaluate.games_per_matchup, smoke.evaluate.seeds) == (16, 3)

    # ---- Tier 2: the profile preset supplies the stage6/7/eval numbers ---------
    quick = resolve_budgets(profile=PROFILE_QUICK, steps_fallback=10, overrides={})
    assert quick.profile == PROFILE_QUICK
    assert (quick.stage6.games_per_matchup, quick.stage6.seeds, quick.stage6.prize_sequences) == (8, 3, 2)
    assert quick.stage7.attack_cases == 24
    assert quick.stage7.heldout_attack_cases == 12
    assert quick.stage7.arena_games == 8
    full = resolve_budgets(profile=PROFILE_FULL, steps_fallback=10, overrides={})
    assert full.profile == PROFILE_FULL
    # delta #4: Arena v1 is fixed-budget on EVERY profile — sequential CI stopping
    # is disabled everywhere, FULL included.  Re-execute the resolver for all three
    # profiles and pin that none enables it and the target half-width stays 0.
    for prof in (PROFILE_SMOKE, PROFILE_QUICK, PROFILE_FULL):
        resolved = resolve_budgets(profile=prof, steps_fallback=10, overrides={})
        assert resolved.stage6.sequential_ci_stop is False
        assert resolved.stage6.ci_target_halfwidth == 0.0

    # ---- Tier 1 beats Tier 2: an explicit flag overrides the profile preset ----
    overridden = resolve_budgets(
        profile=PROFILE_QUICK,
        steps_fallback=10,
        overrides={"stage6_games_per_matchup": 50, "stage7_attack_cases": 7},
    )
    assert overridden.stage6.games_per_matchup == 50   # flag wins over QUICK's 8
    assert overridden.stage6.seeds == 3                 # unset field still QUICK's 3
    assert overridden.stage7.attack_cases == 7          # flag wins over QUICK's 24
    assert overridden.stage7.heldout_attack_cases == 12  # unset -> QUICK preset

    # ---- Tier 3: --steps fallback fills stage4_steps / stage5_sessions ONLY -----
    fell_back = resolve_budgets(profile=PROFILE_SMOKE, steps_fallback=999, overrides={})
    assert fell_back.stage4_steps == 999
    assert fell_back.stage5_sessions == 999
    # The fallback must NOT leak into the stage6/7/eval budgets (they never
    # consumed --steps); those stay on the SMOKE preset even at steps_fallback=999.
    assert fell_back.stage6.games_per_matchup == 1
    assert fell_back.evaluate.games_per_matchup == 16

    # ---- Tier 1 beats Tier 3: an explicit θ-flag overrides the --steps fallback -
    theta = resolve_budgets(
        profile=PROFILE_SMOKE,
        steps_fallback=999,
        overrides={"stage4_steps": 123},
    )
    assert theta.stage4_steps == 123      # explicit flag
    assert theta.stage5_sessions == 999   # unset -> --steps fallback
