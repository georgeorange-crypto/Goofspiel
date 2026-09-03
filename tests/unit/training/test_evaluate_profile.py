"""Evaluate profile discipline (Section 9): only FULL certifies.

The evaluate suite threads the RESOLVED profile name into the benchmark, and the
benchmark's promotion decision must honour it:

  * SMOKE / QUICK (any non-FULL profile) -> ``NOT_EVALUATED`` — never a binding
    PROMOTE/REJECT, no matter what the gates say;
  * FULL -> a binding decision (INCOMPLETE / PROMOTE_CANDIDATE / REJECT_CANDIDATE),
    never ``NOT_EVALUATED``.

Both halves RE-EXECUTE the fact: we read the ``summary.json`` the suite actually
wrote, and we independently recompute the decision with ``decide_promotion`` from
the report's own tri-state gates and profile, proving the persisted decision is
reproducible from its inputs.
"""
from __future__ import annotations

import json
from pathlib import Path

from goofspiel.training.benchmark import (
    PROMOTION_NOT_EVALUATED,
    EvaluationProfile,
    decide_promotion,
)
from goofspiel.training.stages import run_evaluation_suite


def _run(out_dir: Path, profile_name: str) -> dict:
    run_evaluation_suite(out_dir=out_dir, num_games=4, seeds=[1, 2], profile_name=profile_name)
    return json.loads((out_dir / "reports" / "quick" / "summary.json").read_text(encoding="utf-8"))


def test_quick_profile_never_certifies(tmp_path: Path):
    summary = _run(tmp_path / "q", "QUICK")
    assert summary["profile"]["name"] == "QUICK"
    assert summary["promotion_decision"] == PROMOTION_NOT_EVALUATED

    # RE-EXECUTE: recompute the decision from the report's own gates + profile.
    profile = EvaluationProfile(name="QUICK", seeds=[1, 2], num_games=4, include_e7=False)
    recomputed = decide_promotion(profile, summary["hard_gates"])
    assert recomputed == PROMOTION_NOT_EVALUATED
    assert not profile.emits_binding_promotion


def test_smoke_profile_never_certifies(tmp_path: Path):
    summary = _run(tmp_path / "s", "SMOKE")
    assert summary["profile"]["name"] == "SMOKE"
    assert summary["promotion_decision"] == PROMOTION_NOT_EVALUATED


def test_full_profile_emits_binding_decision(tmp_path: Path):
    summary = _run(tmp_path / "f", "FULL")
    assert summary["profile"]["name"] == "FULL"
    # FULL is the ONLY profile that certifies — its decision is binding, so it is
    # never NOT_EVALUATED (here INCOMPLETE, since the smoke-sized gates left some
    # required gate NOT_RUN — but decisively not the non-binding sentinel).
    assert summary["promotion_decision"] != PROMOTION_NOT_EVALUATED

    profile = EvaluationProfile(name="FULL", seeds=[1, 2], num_games=4, include_e7=False)
    assert profile.emits_binding_promotion
    recomputed = decide_promotion(profile, summary["hard_gates"])
    assert recomputed == summary["promotion_decision"], "FULL decision is not reproducible from its gates"
    assert recomputed != PROMOTION_NOT_EVALUATED
