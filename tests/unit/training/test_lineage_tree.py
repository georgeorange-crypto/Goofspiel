"""Priority ⑥ — the lineage tree's consistency check actually catches breakage.

`build_lineage_tree(...).is_consistent()` is only worth anything if it goes RED
when the lineage is genuinely broken and GREEN when it is sound.  These tests
prove both directions by RE-EXECUTING the fact — building the tree from real
checkpoints on disk and, for the negative case, physically corrupting the parent
file and rebuilding — never by reading a stored 'consistent' flag.

  1. A real chained run (P1→P3→P4) builds a tree that IS consistent, and its
     chain order follows the parent links head-to-tail.
  2. Tampering the PARENT file after the child inherited makes the tree
     INCONSISTENT, and names the offending node with reason
     'parent_content_changed' — the exact failure a bare parent-id string cannot
     detect.
  3. A child naming a parent that is absent from the tree is flagged
     'dangling_parent'.
"""

from __future__ import annotations

import pytest

try:
    import torch
except OSError as exc:  # pragma: no cover - machine environment guard
    pytest.skip(f"torch cannot be imported in this environment: {exc}", allow_module_level=True)

from goofspiel.training.checkpoint import load_checkpoint, sha256_file
from goofspiel.training.lineage import build_lineage_tree


def _chain(tmp_path):
    from goofspiel.training.stages import (
        run_stage1_pretrain,
        run_stage3_sft,
        run_stage4_robust_rl,
    )

    p1 = run_stage1_pretrain(steps=1, batch_size=4, out_dir=tmp_path / "ck", n_cards=3)
    p3 = run_stage3_sft(
        steps=1, batch_size=8, out_dir=tmp_path / "ck", n_cards=5,
        init_from_checkpoint=p1.checkpoint,
    )
    p4 = run_stage4_robust_rl(
        steps=1, batch_size=8, out_dir=tmp_path / "ck", n_cards=5,
        init_from_checkpoint=p3.checkpoint,
    )
    return p1, p3, p4


def test_real_chain_is_consistent_and_ordered(tmp_path):
    p1, p3, p4 = _chain(tmp_path)
    tree = build_lineage_tree([p1.checkpoint, p3.checkpoint, p4.checkpoint])

    assert tree.is_consistent(), tree.inconsistencies()
    # The recorded parent sha256 on each child equals the parent file's real hash.
    for child, parent in ((p3.checkpoint, p1.checkpoint), (p4.checkpoint, p3.checkpoint)):
        child_meta = load_checkpoint(child)["metadata"]
        assert child_meta["parent_checkpoint_sha256"] == sha256_file(parent)
    # Chain order follows the parent links head→tail.
    order = tree.chain_order()
    assert order == ["stage1_pretrain", "stage3_sft", "stage4_robust_rl"]


def test_tampered_parent_breaks_consistency(tmp_path):
    """Re-run the parent stage after the child inherited → tree must flag the child.

    The realistic breakage: P4 inherits P3, then P3 is regenerated (a new, VALID
    checkpoint at the same path with different bytes — e.g. someone re-ran that
    stage).  P4's stored ``parent_checkpoint_sha256`` no longer matches P3's
    current content, so the lineage is no longer sound — the exact drift a bare
    ``parent_checkpoint_id`` string cannot see.  Using a valid replacement (not
    appended garbage) keeps the file loadable, so the tree still SEES P3 and can
    diagnose 'parent_content_changed' rather than losing it to a load error.
    """
    from pathlib import Path

    from goofspiel.training.stages import run_stage1_pretrain, run_stage3_sft

    p1, p3, p4 = _chain(tmp_path)
    tree_ok = build_lineage_tree([p1.checkpoint, p3.checkpoint, p4.checkpoint])
    assert tree_ok.is_consistent()
    old_p3_sha = sha256_file(p3.checkpoint)

    # Regenerate P3 in place with more steps → a valid but DIFFERENT file.
    p3b = run_stage3_sft(
        steps=3, batch_size=8, out_dir=tmp_path / "ck", n_cards=5,
        init_from_checkpoint=p1.checkpoint,
    )
    assert Path(p3b.checkpoint) == Path(p3.checkpoint), "expected in-place overwrite"
    new_sha = sha256_file(p3.checkpoint)
    assert new_sha != old_p3_sha, "regeneration did not change the file bytes"

    tree_bad = build_lineage_tree([p1.checkpoint, p3.checkpoint, p4.checkpoint])
    assert not tree_bad.is_consistent()
    problems = {p["node"]: p for p in tree_bad.inconsistencies()}
    assert "stage4_robust_rl" in problems
    assert problems["stage4_robust_rl"]["reason"] == "parent_content_changed"
    assert problems["stage4_robust_rl"]["recorded_parent_sha256"] == old_p3_sha
    assert problems["stage4_robust_rl"]["actual_parent_sha256"] == new_sha


def test_dangling_parent_is_flagged(tmp_path):
    """A child whose parent file is absent from the tree → 'dangling_parent'."""
    p1, p3, p4 = _chain(tmp_path)
    # Build the tree WITHOUT P3: P4 names stage3_sft, which is now missing.
    tree = build_lineage_tree([p1.checkpoint, p4.checkpoint])
    assert not tree.is_consistent()
    problems = {p["node"]: p for p in tree.inconsistencies()}
    assert problems["stage4_robust_rl"]["reason"] == "dangling_parent"
    assert problems["stage4_robust_rl"]["parent_checkpoint_id"] == "stage3_sft"


def test_full_sequence_summary_reports_consistent_lineage(tmp_path):
    """A real full-sequence run stamps a GREEN lineage verdict in its summary."""
    from goofspiel.training import TrainingCoordinator, TrainingRunConfig

    cfg = TrainingRunConfig(
        artifact_dir=str(tmp_path), stage="all", steps=1, batch_size=2,
        n_cards=2, num_corpus_games=1, device="cpu",
    )
    summary = TrainingCoordinator(cfg).run_full_sequence()
    assert summary["lineage_consistent"] is True, summary["lineage_inconsistencies"]
    # Head-to-tail order over the θ producers this run actually wrote.
    assert summary["lineage_order"][0] == "stage1_pretrain"
    assert summary["lineage_order"][-1] == "stage5_adaptive"
