"""Re-execution tests for calibration provenance (§4/§12/§15 label enforcement).

These build a *real* temporary git repo so the git helpers are exercised against
git itself, not a mock — the re-execute principle: assert the fact by recomputing
it, never by reading back a field we set.
"""

from __future__ import annotations

import subprocess

import pytest

from goofspiel.calibration import (
    ALLOWED_PROFILE_NAMES,
    ANCHOR_N_CARDS,
    BINDING_PROMOTION,
    EVALUATION_PURPOSE,
)
from goofspiel.calibration.provenance import (
    ProvenanceError,
    RunProvenance,
    capture,
    git_branch,
    git_commit,
    git_is_dirty,
    sha256_file,
)


def _run(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)


@pytest.fixture
def temp_repo(tmp_path):
    """A real git repo with one commit; returns (path, head_sha)."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "init")
    _run(repo, "config", "user.email", "t@t")
    _run(repo, "config", "user.name", "t")
    (repo / "a.txt").write_text("hello", encoding="utf-8")
    _run(repo, "add", "a.txt")
    _run(repo, "commit", "-m", "init")
    head = git_commit(repo)
    return repo, head


def test_git_commit_is_real_40_hex(temp_repo):
    repo, head = temp_repo
    # Re-execute: ask git directly and compare.
    direct = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert git_commit(repo) == direct
    assert len(head) == 40 and all(c in "0123456789abcdef" for c in head)


def test_git_is_dirty_tracks_modifications_not_untracked(temp_repo):
    repo, _ = temp_repo
    assert git_is_dirty(repo) is False
    # Untracked file alone → still not dirty (reproducible checkout).
    (repo / "new_untracked.txt").write_text("x", encoding="utf-8")
    assert git_is_dirty(repo) is False
    # Modifying a tracked file → dirty.
    (repo / "a.txt").write_text("changed", encoding="utf-8")
    assert git_is_dirty(repo) is True


def test_git_branch_reports_current(temp_repo):
    repo, _ = temp_repo
    _run(repo, "checkout", "-b", "calibration/F")
    assert git_branch(repo) == "calibration/F"


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib

    p = tmp_path / "ckpt.bin"
    payload = b"weights" * 5000
    p.write_bytes(payload)
    assert sha256_file(p) == hashlib.sha256(payload).hexdigest()


def test_capture_hashes_checkpoint_and_binds_sha(temp_repo, tmp_path):
    repo, head = temp_repo
    ckpt = tmp_path / "model.pt"
    ckpt.write_bytes(b"\x00\x01\x02\x03")
    prov = capture(repo, device="cpu", world_size=1, seed=7, n_cards=5, checkpoint_path=ckpt)
    assert prov.git_commit == head
    assert prov.checkpoint_sha256 == sha256_file(ckpt)
    assert prov.is_at_anchor is True
    assert prov.formal_ready is True  # clean tree


# -- label invariants -------------------------------------------------------- #
def _valid_kwargs(**over):
    base = dict(
        git_commit="a" * 40,
        git_branch="calibration/F",
        dirty=False,
        device="cpu",
        world_size=1,
        seed=1,
        n_cards=5,
    )
    base.update(over)
    return base


def test_defaults_are_the_only_legal_labels():
    prov = RunProvenance(**_valid_kwargs())
    assert prov.evaluation_purpose == EVALUATION_PURPOSE == "budget_calibration"
    assert prov.binding_promotion is BINDING_PROMOTION is False
    assert prov.profile_name in ALLOWED_PROFILE_NAMES
    assert "FULL" not in ALLOWED_PROFILE_NAMES  # never a bare binding name


def test_short_sha_rejected():
    with pytest.raises(ProvenanceError):
        RunProvenance(**_valid_kwargs(git_commit="abc123"))


def test_binding_promotion_true_rejected():
    with pytest.raises(ProvenanceError):
        RunProvenance(**_valid_kwargs(binding_promotion=True))


def test_full_profile_name_rejected():
    with pytest.raises(ProvenanceError):
        RunProvenance(**_valid_kwargs(profile_name="FULL"))


def test_wrong_purpose_rejected():
    with pytest.raises(ProvenanceError):
        RunProvenance(**_valid_kwargs(evaluation_purpose="binding_eval"))


def test_dirty_tree_downgrades_disposition():
    clean = RunProvenance(**_valid_kwargs(dirty=False))
    dirty = RunProvenance(**_valid_kwargs(dirty=True))
    assert clean.formal_ready is True
    assert "CALIBRATION" in clean.disposition()
    assert dirty.formal_ready is False
    assert "NOT FOR PROMOTION" in dirty.disposition()


def test_anchor_flag_reflects_n_cards():
    assert RunProvenance(**_valid_kwargs(n_cards=ANCHOR_N_CARDS)).is_at_anchor is True
    assert RunProvenance(**_valid_kwargs(n_cards=13)).is_at_anchor is False
