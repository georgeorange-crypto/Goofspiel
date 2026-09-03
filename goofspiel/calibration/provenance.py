"""Run provenance for F calibration measurements (§4 / §12 / §15).

Every calibration measurement must be reproducible from its record alone: the
exact 40-char commit it ran on, whether the tree was dirty, the checkpoint's
content hash, and the config/seed/topology.  This module captures that and
enforces the non-binding calibration labels so an F measurement can never be
mistaken for — or promoted to — a binding FULL result.

Pure stdlib (subprocess + hashlib): no torch, no Arena imports.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from . import (
    ALLOWED_PROFILE_NAMES,
    ANCHOR_N_CARDS,
    BINDING_PROMOTION,
    EVALUATION_PURPOSE,
)

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class ProvenanceError(RuntimeError):
    """Raised when a provenance record violates a calibration invariant."""


def _git(repo: str | Path, *args: str) -> str:
    """Run ``git -C <repo> <args>`` and return stripped stdout.

    Uses ``-C`` (not cwd juggling) so it is safe to call against any worktree.
    """
    out = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def git_commit(repo: str | Path) -> str:
    """Full 40-char HEAD SHA of ``repo``."""
    return _git(repo, "rev-parse", "HEAD")


def git_branch(repo: str | Path) -> str:
    """Current branch name (or ``HEAD`` when detached)."""
    return _git(repo, "rev-parse", "--abbrev-ref", "HEAD")


def git_is_dirty(repo: str | Path) -> bool:
    """True if the worktree has staged or unstaged changes to tracked files.

    Untracked files alone do NOT count as dirty for reproducibility of a
    committed checkout — but a modified tracked file does.  ``--untracked-files=no``
    makes that distinction explicit.
    """
    status = _git(repo, "status", "--porcelain", "--untracked-files=no")
    return bool(status.strip())


def sha256_file(path: str | Path, *, chunk: int = 1 << 20) -> str:
    """Streaming SHA256 of a file's bytes (checkpoint content hash)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class RunProvenance:
    """Immutable-ish record binding one measurement to its exact substrate.

    The three calibration invariants (purpose, non-binding, profile name) are
    enforced in :meth:`validate`, called from :meth:`__post_init__`, so an
    invalid record cannot be constructed silently.
    """

    git_commit: str
    git_branch: str
    dirty: bool
    device: str
    world_size: int
    seed: int
    n_cards: int
    checkpoint_path: str | None = None
    checkpoint_sha256: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    # Calibration labels — defaulted to the only legal values; overriding them
    # to something binding raises in validate().
    evaluation_purpose: str = EVALUATION_PURPOSE
    binding_promotion: bool = BINDING_PROMOTION
    profile_name: str = "CALIBRATION"

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not _SHA_RE.match(self.git_commit or ""):
            raise ProvenanceError(
                f"git_commit must be a full 40-char SHA, got {self.git_commit!r}"
            )
        if self.evaluation_purpose != EVALUATION_PURPOSE:
            raise ProvenanceError(
                "calibration runs must set evaluation_purpose="
                f"{EVALUATION_PURPOSE!r}, got {self.evaluation_purpose!r}"
            )
        if self.binding_promotion is not False:
            raise ProvenanceError(
                "binding_promotion must be False for a calibration run "
                "(F never promotes)"
            )
        if self.profile_name not in ALLOWED_PROFILE_NAMES:
            raise ProvenanceError(
                f"profile_name must be one of {ALLOWED_PROFILE_NAMES} "
                f"(never 'FULL'), got {self.profile_name!r}"
            )
        if self.world_size < 1:
            raise ProvenanceError(f"world_size must be >=1, got {self.world_size}")

    @property
    def is_at_anchor(self) -> bool:
        """True when this ran at the measured anchor board size (n=5)."""
        return self.n_cards == ANCHOR_N_CARDS

    @property
    def formal_ready(self) -> bool:
        """A measurement is trustworthy as an anchor only from a clean tree.

        Not a promotion — F never promotes — but a dirty tree means the numbers
        cannot be reproduced from the recorded SHA, so they may only be reported
        as DEV / NON-BINDING (see :meth:`disposition`).
        """
        return not self.dirty

    def disposition(self) -> str:
        """Human-facing tag for reports: clean → CALIBRATION, dirty → DEV."""
        return "CALIBRATION (NON-BINDING)" if self.formal_ready else "DEV / NON-BINDING / NOT FOR PROMOTION"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def capture(
    repo: str | Path,
    *,
    device: str,
    world_size: int,
    seed: int,
    n_cards: int,
    checkpoint_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
    profile_name: str = "CALIBRATION",
) -> RunProvenance:
    """Snapshot the live provenance of ``repo`` into a :class:`RunProvenance`.

    Hashes the checkpoint file if given.  Raises via ``validate`` if the labels
    are illegal.  Does NOT enforce cleanliness here — the caller decides whether
    a dirty tree is acceptable (drivers refuse it; ad-hoc probes may allow it and
    be tagged DEV).
    """
    ckpt_sha = sha256_file(checkpoint_path) if checkpoint_path is not None else None
    return RunProvenance(
        git_commit=git_commit(repo),
        git_branch=git_branch(repo),
        dirty=git_is_dirty(repo),
        device=device,
        world_size=world_size,
        seed=seed,
        n_cards=n_cards,
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        checkpoint_sha256=ckpt_sha,
        config=dict(config or {}),
        profile_name=profile_name,
    )


__all__ = [
    "ProvenanceError",
    "RunProvenance",
    "capture",
    "git_commit",
    "git_branch",
    "git_is_dirty",
    "sha256_file",
]
