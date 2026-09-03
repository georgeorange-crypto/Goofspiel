"""Stage budget & evaluation-profile resolution (Step 1 of the Stage6/7/eval
workload+budget upgrade).

Historically a single global ``--steps`` was threaded into every stage, but
Stage6/Stage7/evaluate never actually consumed it — they hardcoded their own
workload (9 single-game cross-play pairs, 3 fixed attack states + 40 optimizer
steps, 16 evaluation games).  This module gives each stage a *typed* budget with
clear semantics, resolved ONCE from a profile preset plus optional per-stage
overrides, and carried in ``TrainingRunConfig.extra["budgets"]``.

Three profiles:

* ``SMOKE`` — correctness / tiny / CI default.  Its presets are calibrated so the
  existing smoke path is byte-preserved (9 cross-play pairs, exactly 3 Stage7
  train attacks, 16 evaluation games).  This is the default when no profile and
  no override is supplied, so CI behaviour is unchanged.
* ``QUICK`` — moderate diagnostic.  Never emits a binding PASS/FAIL (the
  benchmark's ``decide_promotion`` returns ``NOT_EVALUATED`` for any non-FULL
  profile).
* ``FULL`` — statistical / paper-grade.  Its numbers here are *provisional*
  (TBD): the real values are chosen in Step 2 after a FULL-lite throughput
  benchmark on a formal checkpoint, per the two-step rollout.

Precedence, resolved per field (highest wins):

1. an explicit per-stage override (a non-``None`` value in ``overrides``);
2. the profile preset;
3. the ``--steps`` fallback — **only** for ``stage4_steps`` / ``stage5_sessions``
   (the θ stages that historically consumed the global ``--steps``); it is
   deliberately NOT a fallback for the Stage6/7/eval evaluation budgets, which
   never consumed ``--steps``;
4. the dataclass default (= the SMOKE preset value).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

PROFILE_SMOKE = "SMOKE"
PROFILE_QUICK = "QUICK"
PROFILE_FULL = "FULL"
PROFILES = (PROFILE_SMOKE, PROFILE_QUICK, PROFILE_FULL)


@dataclass
class Stage6Budget:
    """League cross-play workload.

    A matchup is aggregated over ``games_per_matchup × seeds × prize_sequences``
    games played on common random numbers (the same prize schedule is reused
    across both seat orders of a pair).

    ``sequential_ci_stop`` is **disabled in Arena v1** (delta #4): every profile,
    SMOKE/QUICK/FULL, is fixed-budget so a matchup's game count is a property of
    the profile, not of the data it happened to see.  The field (and
    ``ci_target_halfwidth``) are retained for forward-compatibility and mapping
    round-trips, but no preset sets them and the Stage6 runner never early-stops.
    """

    games_per_matchup: int = 1
    seeds: int = 1
    prize_sequences: int = 1
    sequential_ci_stop: bool = False  # delta #4: forced off in v1 (no preset enables it)
    ci_target_halfwidth: float = 0.0


@dataclass
class Stage7Budget:
    """Red-team discovery / correction / regression workload.

    ``attack_cases`` attacks are discovered; the first ``correction_train_cases``
    of them form the TRAIN set the focused correction optimises on; any remainder
    plus ``heldout_attack_cases`` additional attacks are held out for a
    generalization / memorization measurement.  ``arena_games`` > 0 enables the
    corrected-vs-robust arena across the bot suite.  ``correction_steps`` is
    carried here for reporting but the runner's own ``correction_steps`` parameter
    is authoritative for the optimizer loop (the coordinator passes the two
    consistently).
    """

    attack_cases: int = 3
    correction_steps: int = 40
    heldout_attack_cases: int = 0
    correction_train_cases: int = 3
    arena_games: int = 0
    arena_seeds: int = 0


@dataclass
class EvaluateBudget:
    """Evaluation-suite workload: games per bot matchup and number of seeds."""

    games_per_matchup: int = 16
    seeds: int = 3


@dataclass
class StageBudgets:
    """The resolved budget tree for one run.

    ``stage4_steps`` / ``stage5_sessions`` / ``stage5_adaptation_steps`` are the
    θ-training budgets (``None`` means "inherit the global ``--steps``").
    ``stage5_adaptation_steps`` is carried for forward-compatibility and
    reporting; it has no home in the current Stage5 firewall algorithm and is
    intentionally NOT wired into training (wiring it would touch Stage5, which is
    out of scope for this change).
    """

    profile: str = PROFILE_SMOKE
    stage4_steps: int | None = None
    stage5_sessions: int | None = None
    stage5_adaptation_steps: int | None = None
    stage6: Stage6Budget = field(default_factory=Stage6Budget)
    stage7: Stage7Budget = field(default_factory=Stage7Budget)
    evaluate: EvaluateBudget = field(default_factory=EvaluateBudget)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "StageBudgets":
        """Reconstruct a ``StageBudgets`` from a (possibly JSON-round-tripped)
        nested mapping such as ``asdict(budgets)`` — tolerant of missing keys,
        which fall back to the SMOKE defaults."""

        def _sub(cls_, key):
            raw = data.get(key)
            if isinstance(raw, cls_):
                return raw
            raw = raw or {}
            allowed = {f.name for f in fields(cls_)}
            return cls_(**{k: v for k, v in raw.items() if k in allowed})

        return cls(
            profile=str(data.get("profile", PROFILE_SMOKE)),
            stage4_steps=data.get("stage4_steps"),
            stage5_sessions=data.get("stage5_sessions"),
            stage5_adaptation_steps=data.get("stage5_adaptation_steps"),
            stage6=_sub(Stage6Budget, "stage6"),
            stage7=_sub(Stage7Budget, "stage7"),
            evaluate=_sub(EvaluateBudget, "evaluate"),
        )


# ---------------------------------------------------------------------------
# Profile presets.  These are the ONLY hardcoded budget numbers in the codebase.
# FULL is provisional (TBD): its numbers are finalised in Step 2 after a
# throughput benchmark, per the two-step rollout — do not treat them as a
# committed statistical budget yet.
# ---------------------------------------------------------------------------
_PRESETS: dict[str, StageBudgets] = {
    PROFILE_SMOKE: StageBudgets(
        profile=PROFILE_SMOKE,
        stage6=Stage6Budget(games_per_matchup=1, seeds=1, prize_sequences=1),
        stage7=Stage7Budget(
            attack_cases=3, correction_steps=40, heldout_attack_cases=0,
            correction_train_cases=3, arena_games=0, arena_seeds=0,
        ),
        evaluate=EvaluateBudget(games_per_matchup=16, seeds=3),
    ),
    PROFILE_QUICK: StageBudgets(
        profile=PROFILE_QUICK,
        stage6=Stage6Budget(games_per_matchup=8, seeds=3, prize_sequences=2),
        stage7=Stage7Budget(
            attack_cases=24, correction_steps=40, heldout_attack_cases=12,
            correction_train_cases=8, arena_games=8, arena_seeds=2,
        ),
        evaluate=EvaluateBudget(games_per_matchup=16, seeds=3),
    ),
    PROFILE_FULL: StageBudgets(
        profile=PROFILE_FULL,
        stage6=Stage6Budget(
            games_per_matchup=100, seeds=20, prize_sequences=5,
            sequential_ci_stop=False, ci_target_halfwidth=0.0,  # delta #4: fixed budget, no early stop
        ),
        stage7=Stage7Budget(
            attack_cases=500, correction_steps=40, heldout_attack_cases=200,
            correction_train_cases=300, arena_games=100, arena_seeds=20,
        ),
        evaluate=EvaluateBudget(games_per_matchup=100, seeds=20),
    ),
}


def _preset_for(profile: str | None) -> StageBudgets:
    name = (profile or PROFILE_SMOKE).strip().upper()
    if name not in _PRESETS:
        raise ValueError(f"unknown profile {profile!r}; allowed={list(PROFILES)}")
    # Deep-copy the preset so callers never mutate the shared template.
    return StageBudgets.from_mapping(_asdict_budgets(_PRESETS[name]))


def _asdict_budgets(b: StageBudgets) -> dict[str, Any]:
    from dataclasses import asdict

    return asdict(b)


def _pick(override: Any, preset: Any) -> Any:
    """Explicit per-stage override wins over the profile preset value."""
    return preset if override is None else override


def resolve_budgets(
    *,
    profile: str | None,
    steps_fallback: int,
    overrides: dict[str, Any] | None = None,
) -> StageBudgets:
    """Resolve the full budget tree for a run.

    ``profile`` selects the preset (default SMOKE).  ``overrides`` carries the
    per-stage CLI flags (``None`` for any flag the user did not pass).
    ``steps_fallback`` is the legacy global ``--steps`` value, used ONLY as the
    default for ``stage4_steps`` / ``stage5_sessions``.
    """
    overrides = overrides or {}
    base = _preset_for(profile)

    def ov(key: str) -> Any:
        return overrides.get(key)

    # θ-training budgets: explicit flag > --steps fallback.  (The presets leave
    # these None: they are not a per-profile evaluation concept.)
    stage4_steps = ov("stage4_steps")
    stage4_steps = int(stage4_steps) if stage4_steps is not None else int(steps_fallback)
    stage5_sessions = ov("stage5_sessions")
    stage5_sessions = int(stage5_sessions) if stage5_sessions is not None else int(steps_fallback)
    stage5_adaptation_steps = ov("stage5_adaptation_steps")
    stage5_adaptation_steps = (
        int(stage5_adaptation_steps) if stage5_adaptation_steps is not None else None
    )

    stage6 = Stage6Budget(
        games_per_matchup=int(_pick(ov("stage6_games_per_matchup"), base.stage6.games_per_matchup)),
        seeds=int(_pick(ov("stage6_seeds"), base.stage6.seeds)),
        prize_sequences=int(_pick(ov("stage6_prize_sequences"), base.stage6.prize_sequences)),
        # delta #4: Arena v1 is fixed-budget on EVERY profile — sequential CI
        # stopping is force-disabled here, so it cannot re-enter via a preset or a
        # stale on-disk mapping.
        sequential_ci_stop=False,
        ci_target_halfwidth=0.0,
    )
    stage7 = Stage7Budget(
        attack_cases=int(_pick(ov("stage7_attack_cases"), base.stage7.attack_cases)),
        correction_steps=int(_pick(ov("stage7_correction_steps"), base.stage7.correction_steps)),
        heldout_attack_cases=int(
            _pick(ov("stage7_heldout_attack_cases"), base.stage7.heldout_attack_cases)
        ),
        correction_train_cases=int(
            _pick(ov("stage7_correction_train_cases"), base.stage7.correction_train_cases)
        ),
        arena_games=int(_pick(ov("stage7_arena_games"), base.stage7.arena_games)),
        arena_seeds=int(_pick(ov("stage7_arena_seeds"), base.stage7.arena_seeds)),
    )
    evaluate = EvaluateBudget(
        games_per_matchup=int(_pick(ov("eval_games_per_matchup"), base.evaluate.games_per_matchup)),
        seeds=int(_pick(ov("eval_seeds"), base.evaluate.seeds)),
    )
    return StageBudgets(
        profile=base.profile,
        stage4_steps=stage4_steps,
        stage5_sessions=stage5_sessions,
        stage5_adaptation_steps=stage5_adaptation_steps,
        stage6=stage6,
        stage7=stage7,
        evaluate=evaluate,
    )


def coerce_budgets(raw: Any, *, steps_fallback: int) -> StageBudgets:
    """Return a ``StageBudgets`` from whatever is stored in ``config.extra``.

    Accepts an already-built ``StageBudgets``, a nested mapping (e.g. after a
    JSON round-trip through ``resolved_config.json``), or ``None`` — in which
    case the SMOKE profile is resolved with the given ``--steps`` fallback so a
    config built without any budget (every existing test) behaves exactly as
    before.
    """
    if isinstance(raw, StageBudgets):
        return raw
    if isinstance(raw, dict):
        budgets = StageBudgets.from_mapping(raw)
        # A mapping that omitted the θ budgets inherits the --steps fallback.
        if budgets.stage4_steps is None:
            budgets.stage4_steps = int(steps_fallback)
        if budgets.stage5_sessions is None:
            budgets.stage5_sessions = int(steps_fallback)
        return budgets
    return resolve_budgets(profile=PROFILE_SMOKE, steps_fallback=steps_fallback, overrides={})


# The override keys the CLI populates (used by scripts/train_goofspiel_full.py).
OVERRIDE_KEYS = (
    "stage4_steps",
    "stage5_sessions",
    "stage5_adaptation_steps",
    "stage6_games_per_matchup",
    "stage6_seeds",
    "stage6_prize_sequences",
    "stage7_attack_cases",
    "stage7_correction_steps",
    "stage7_heldout_attack_cases",
    "stage7_correction_train_cases",
    "stage7_arena_games",
    "stage7_arena_seeds",
    "eval_games_per_matchup",
    "eval_seeds",
)
