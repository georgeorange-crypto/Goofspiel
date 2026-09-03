"""Workstream F — Arena budget-calibration harness (MEASUREMENT ONLY).

This package is the calibration/F workstream's measurement infrastructure.  Its
job is to *measure* the cost of the Arena runners (Stage6 league, Stage7
red-team, evaluate) at a fixed workload, fit how that cost scales, and propose
candidate FULL budgets — nothing else.

Hard invariants baked in here (see the F directive + [[git-worktree-commit-protocol]]):

  * **Measurement only.**  Nothing in this package changes an algorithm, edits
    the Arena code, applies the recovery patch, or writes ``budgets.py``.  The
    F6 output is a *report*, never a schema mutation.
  * **No binding promotion.**  Every run is stamped
    ``evaluation_purpose="budget_calibration"``, ``binding_promotion=False`` and
    named ``FULL_LITE`` / ``CALIBRATION`` — never ``FULL``.
  * **n=5 is the measured anchor.**  formal1k trained at n_cards=5.  Any
    workload/board size that was not *directly executed* is emitted as
    ``PROJECTED`` (never ``MEASURED``), and n≠5 (especially n=13) is additionally
    flagged as beyond the anchor.  A projection can never be presented as a
    measured FULL runtime — the type system here forbids it.
  * **No sequential stopping.**  F uses a fixed workload sweep; the harness never
    early-stops on a CI target.
  * **Provenance or it didn't happen.**  Every measurement binds to an exact
    40-char commit SHA + ``dirty`` flag + checkpoint SHA256 (§4/§12/§15).  The
    F1–F4 drivers refuse to run against a dirty tree or a SHA that isn't the
    committed Arena integration SHA.

The package deliberately imports no Arena code and no torch at module load: the
analysis core (records / scaling / budget_report) runs on ingested numbers, and
the drivers import the Arena runners lazily only at execution time.
"""

from __future__ import annotations

# Fixed labels — importable so callers and tests reference the single source of
# truth rather than re-typing the strings.
EVALUATION_PURPOSE = "budget_calibration"
BINDING_PROMOTION = False
#: Profile names the calibration harness is allowed to emit.  "FULL" is
#: intentionally absent: a calibration run must never be named like a binding run.
ALLOWED_PROFILE_NAMES = ("FULL_LITE", "CALIBRATION")
#: The measured empirical anchor: formal1k is trained at n_cards=5.
ANCHOR_N_CARDS = 5

__all__ = [
    "EVALUATION_PURPOSE",
    "BINDING_PROMOTION",
    "ALLOWED_PROFILE_NAMES",
    "ANCHOR_N_CARDS",
]
