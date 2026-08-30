from __future__ import annotations

from pathlib import Path


def test_unified_benchmark_writes_required_reports(tmp_path):
    from goofspiel.training.benchmark import EvaluationProfile, run_unified_benchmark, write_benchmark_report

    report = run_unified_benchmark(EvaluationProfile(name="QUICK", seeds=[1], num_games=1))
    paths = write_benchmark_report(report, tmp_path / "reports" / "candidate")
    assert report.benchmark_version
    assert "E0_MATHEMATICAL_CORRECTNESS" in report.arenas
    assert "E5_SEARCH_COMPUTE" in report.arenas
    assert report.arenas["E0_MATHEMATICAL_CORRECTNESS"]["passed"] is True
    assert report.arenas["E5_SEARCH_COMPUTE"]["rows"]
    assert (tmp_path / "reports" / "candidate" / "summary.json").exists()
    assert paths["summary_md"].endswith("summary.md")
    for key in ("main_table", "search_table", "adaptive_table", "opponent_table", "generalization_table"):
        assert Path(paths[key]).exists()
