"""End-to-end: one full sequence, every wired artifact re-executed and verified.

The existing suite splits full-sequence coverage across several tests, each
checking one seam (θ chaining, downstream wiring, lineage tree, dataset
provenance).  None runs the whole ``STAGE_SEQUENCE`` once and then walks every
stage asserting its product exists, loads, and is chained correctly — and none
asserts that the new dual-channel run logger actually wrote both channels.

This test does exactly that, in ONE run, and (per the project testing principle)
re-executes every fact:

  * each θ product is re-loaded from disk and its parent-SHA is re-hashed and
    matched against the child's recorded ``parent_checkpoint_sha256``;
  * stage2's teacher dataset is read back and required non-empty (a seam the
    in-sequence path never checked before);
  * stage6 / stage7 / evaluate products are re-loaded / re-parsed;
  * ``full_sequence_summary.json`` is re-read from disk and its own lineage
    verdict is required True;
  * ``run.log`` and ``events/run.jsonl`` are re-read and required to contain the
    full RUN_START -> STAGE_* -> LINEAGE_VERDICT -> RUN_END event arc — the
    log-was-written fact, re-executed, not a status flag.

Marked ``integration``: it runs the real sequence (~1-2 min on CPU).  Exclude
with ``-m "not integration"``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    import torch  # noqa: F401
except OSError as exc:  # pragma: no cover - environment guard
    pytest.skip(f"torch cannot be imported in this environment: {exc}", allow_module_level=True)

from goofspiel.training import TrainingCoordinator
from goofspiel.training.checkpoint import load_checkpoint, sha256_file
from goofspiel.training.distributed import STAGE_SEQUENCE

# The θ chain, as child -> parent, that must hold byte-for-byte on disk.
_THETA_PARENT_FILES = {
    "checkpoints/stage3_sft.pt": "checkpoints/stage1_pretrain.pt",
    "checkpoints/stage4_robust_rl.pt": "checkpoints/stage3_sft.pt",
    "stage5_adaptive.pt": "checkpoints/stage4_robust_rl.pt",
}
_THETA_FILES = {
    "stage1_pretrain": "checkpoints/stage1_pretrain.pt",
    "stage3_sft": "checkpoints/stage3_sft.pt",
    "stage4_robust_rl": "checkpoints/stage4_robust_rl.pt",
    "stage5_adaptive": "stage5_adaptive.pt",
}


@pytest.mark.integration
def test_full_sequence_produces_every_wired_artifact(tmp_path, tiny_config):
    cfg = tiny_config(tmp_path, stage="all", steps=1, batch_size=2, n_cards=3, num_corpus_games=1)
    summary = TrainingCoordinator(cfg).run_full_sequence()
    root = Path(tmp_path)

    # ---- 1. Every θ product exists, LOADS, and is chained by re-hashed SHA ----
    for child_rel, parent_rel in _THETA_PARENT_FILES.items():
        child = root / child_rel
        parent = root / parent_rel
        assert child.exists(), f"missing θ product {child_rel}"
        assert parent.exists(), f"missing θ parent {parent_rel}"
        child_meta = load_checkpoint(str(child))["metadata"]
        recorded_parent_sha = child_meta.get("parent_checkpoint_sha256")
        assert recorded_parent_sha, f"{child_rel} recorded no parent_checkpoint_sha256"
        # Re-hash the parent file on disk NOW and require it equals what the child
        # stamped when it inherited — proves the child descends from THIS parent,
        # not merely that it names one.
        assert recorded_parent_sha == sha256_file(parent), (
            f"{child_rel} parent SHA does not match the {parent_rel} on disk"
        )

    # The chain head (stage1) loads and names no θ parent.
    p1_meta = load_checkpoint(str(root / _THETA_FILES["stage1_pretrain"]))["metadata"]
    assert p1_meta.get("parent_checkpoint_sha256") in (None, ""), "chain head must have no θ parent"

    # ---- 2. stage2 teacher dataset exists and is non-empty (in-sequence gap) --
    teacher = root / "data" / "teacher_dataset.jsonl"
    assert teacher.exists(), "stage2 wrote no teacher_dataset.jsonl"
    teacher_lines = [ln for ln in teacher.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert teacher_lines, "teacher dataset is empty"
    # every line is valid JSON carrying a state — re-parse, don't trust a count
    assert all("state" in json.loads(ln) for ln in teacher_lines)

    # ---- 3. stage6 / stage7 / evaluate products land and are structurally valid
    league_report = root / "league" / "league_report.json"
    assert league_report.exists()
    assert json.loads(league_report.read_text(encoding="utf-8")), "empty league report"

    corrected = root / "redteam" / "stage7_corrected.pt"
    assert corrected.exists()
    # re-load the corrected checkpoint: it must be a loadable model, not a stub
    corrected_payload = load_checkpoint(str(corrected))
    assert corrected_payload["metadata"]["training_stage"], "stage7 checkpoint has no stage tag"

    evaluation = root / "evaluation_report.json"
    assert evaluation.exists()
    assert json.loads(evaluation.read_text(encoding="utf-8")), "empty evaluation report"

    # ---- 4. summary round-trips from DISK and its own lineage verdict is green
    summary_on_disk = json.loads(
        (root / "full_sequence_summary.json").read_text(encoding="utf-8")
    )
    assert summary_on_disk["ok"] is True
    expected_stages = [s for s in STAGE_SEQUENCE if s != "smoke_pipeline"]
    assert summary_on_disk["stages_run"] == expected_stages
    assert summary_on_disk["lineage_consistent"] is True, summary_on_disk.get(
        "lineage_inconsistencies"
    )
    # the in-memory return and the on-disk file agree on the verdict
    assert summary["lineage_consistent"] == summary_on_disk["lineage_consistent"]

    # ---- 5. the run logger wrote BOTH channels — re-execute that fact ---------
    run_log = root / "run.log"
    assert run_log.exists(), "run.log was not written"
    log_text = run_log.read_text(encoding="utf-8")
    for stage in expected_stages:
        assert stage in log_text, f"run.log never mentions {stage}"
    # summary points at the log products (mirrors smoke's event_log convention)
    assert Path(summary_on_disk["run_log"]) == run_log
    assert summary_on_disk["event_count"] > 0

    event_log = root / "events" / "run.jsonl"
    assert event_log.exists(), "structured event log was not written"
    events = []
    for line in event_log.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))  # re-parse: every line must be valid JSON
    types = [e["event_type"] for e in events]
    # the full arc is present and correctly ordered
    assert types[0] == "RUN_START", types[:3]
    assert types[-1] == "RUN_END", types[-3:]
    assert "LINEAGE_VERDICT" in types
    # every θ stage emitted both a START and an END event
    for stage in expected_stages:
        starts = [e for e in events if e["event_type"] == "STAGE_START" and e["payload"].get("stage") == stage]
        ends = [e for e in events if e["event_type"] == "STAGE_END" and e["payload"].get("stage") == stage]
        assert starts, f"no STAGE_START event for {stage}"
        assert ends, f"no STAGE_END event for {stage}"
        assert ends[-1]["payload"]["ok"] is True, f"{stage} ended not-ok in the event log"
    # STEP_METRICS actually captured per-step loss for the θ stages (not dropped)
    step_events = [e for e in events if e["event_type"] == "STEP_METRICS"]
    assert step_events, "no STEP_METRICS were logged"
    assert all("metrics" in e["payload"] and e["payload"]["metrics"] for e in step_events)
    # θ wiring was logged for each inheriting stage
    wired = {e["payload"]["child"] for e in events if e["event_type"] == "THETA_WIRED"}
    assert {"stage3_sft", "stage4_robust_rl", "stage5_adaptive"} <= wired, wired
