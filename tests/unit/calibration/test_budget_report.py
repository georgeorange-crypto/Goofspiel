"""Re-execution tests for the F6 budget report.

Asserts the report *content* by re-reading what was written to disk (not by
trusting the in-memory object), and pins the two safety properties: the
NOT-FOR-PROMOTION banner is always present, and the writer refuses any output
name that looks like the real budget schema.
"""

from __future__ import annotations

import json

import pytest

from goofspiel.calibration.budget_report import (
    NOT_FOR_PROMOTION_BANNER,
    REPORT_BASENAME,
    BudgetLine,
    BudgetReport,
    FullCandidate,
    make_full_candidates,
)
from goofspiel.calibration.provenance import RunProvenance
from goofspiel.calibration.scaling import LinearFit, Projection, ProjectionKind


def _prov(**over):
    base = dict(
        git_commit="a" * 40,
        git_branch="calibration/F",
        dirty=False,
        device="cuda:0",
        world_size=2,
        seed=1,
        n_cards=5,
    )
    base.update(over)
    return RunProvenance(**base)


def _fit():
    return LinearFit(fixed_overhead_s=10.0, per_game_s=0.5, r_squared=0.99, n_points=4, cv_games_per_sec=0.02)


def test_make_full_candidates_are_all_projected():
    cands = make_full_candidates(_fit(), n_cards=5, small_games=100, medium_games=1000, large_games=10000)
    assert [c.tier for c in cands] == ["FULL_S", "FULL_M", "FULL_L"]
    for c in cands:
        assert c.projection.kind is ProjectionKind.PROJECTED
    # FULL_L projected wall recomputed: 10 + 0.5*10000.
    full_l = cands[-1]
    assert full_l.projection.wall_clock_s == pytest.approx(10.0 + 0.5 * 10000)


def test_report_markdown_has_banner_and_projected_labels():
    fit = _fit()
    report = BudgetReport(
        provenance=_prov(),
        unit_costs=[BudgetLine(component="stage6_play", unit="games", unit_cost_s=0.5)],
        candidates=make_full_candidates(fit, n_cards=13, small_games=100, medium_games=1000, large_games=10000),
        fit=fit,
    )
    md = report.render_markdown()
    assert NOT_FOR_PROMOTION_BANNER in md
    assert "PROJECTED" in md
    assert "BEYOND ANCHOR n=13" in md  # n=13 candidates flagged
    # The report must explicitly disclaim writing the schema file (the footer
    # states F does NOT write budgets.py); that disclaimer is the desired content.
    assert "does not write `budgets.py`" in md
    # And it must never present budgets.py as an artifact it *produced*.
    assert "wrote budgets.py" not in md
    assert "writing budgets.py" not in md


def test_report_json_roundtrips_from_disk(tmp_path):
    fit = _fit()
    report = BudgetReport(
        provenance=_prov(),
        unit_costs=[BudgetLine(component="stage6_play", unit="games", unit_cost_s=0.5)],
        candidates=make_full_candidates(fit, n_cards=5, small_games=100, medium_games=1000, large_games=10000),
        fit=fit,
        notes=["measured at anchor n=5"],
    )
    paths = report.save_report(tmp_path)
    # Re-read from disk and re-verify content, not the in-memory object.
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert data["not_for_promotion"] is True
    assert data["banner"] == NOT_FOR_PROMOTION_BANNER
    assert data["provenance"]["binding_promotion"] is False
    assert data["provenance"]["profile_name"] in ("FULL_LITE", "CALIBRATION")
    tiers = [c["tier"] for c in data["candidates"]]
    assert tiers == ["FULL_S", "FULL_M", "FULL_L"]
    # All candidates carried as PROJECTED.
    assert all(c["kind"] == "PROJECTED" for c in data["candidates"])


def test_saved_filenames_are_report_not_schema(tmp_path):
    report = BudgetReport(provenance=_prov())
    paths = report.save_report(tmp_path)
    assert paths["markdown"].name == f"{REPORT_BASENAME}.md"
    assert paths["json"].name == f"{REPORT_BASENAME}.json"
    assert "budgets" not in paths["markdown"].name
    assert "budgets" not in paths["json"].name


def test_duplicate_tier_rejected():
    fit = _fit()
    p = Projection.from_fit(fit, target_games=100, n_cards=5)
    with pytest.raises(ValueError):
        BudgetReport(
            provenance=_prov(),
            candidates=[
                FullCandidate(tier="FULL_S", purpose="dev", workload_games=100, projection=p),
                FullCandidate(tier="FULL_S", purpose="dev", workload_games=100, projection=p),
            ],
        )


def test_dirty_provenance_shows_dev_disposition_in_report():
    report = BudgetReport(provenance=_prov(dirty=True))
    md = report.render_markdown()
    assert "NOT FOR PROMOTION" in md
    assert "DEV" in md  # dirty disposition surfaced
