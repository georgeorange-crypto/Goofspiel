# S7-F1 -- Fix-Only Production Rerun: MANIFEST + RECIPE (NON-CERTIFYING, BLOCKED)

**Work unit:** S7-F1 -- Minimal Teacher-Policy Value-Slot Fix + Fix-Only Production
Regression. **Date:** 2026-09-04.

**Status of the exact-parent rerun (S7-F1 sec.26-33):**
`DIAGNOSTIC / EXTERNAL-PARENT NON-CERTIFYING LINEAGE -- BLOCKED (parents absent)`.

The single-variable fix-only production rerun that would fill the "Fixed" column of
the sec.32 comparison table **was not executed**, because rerun567's parent
checkpoints are **not on local disk** (verified twice):

- Stage4 parent (`stage4_parent_sha256_prefix: c4e06e2c`) -- absent.
- Stage5 ("fresh GPU checkpoint") -- absent.

S7-F1 sec.26 forbids re-running Stage5/Stage6 to regenerate them, and the fix-only
rerun by definition must start Stage7 from **rerun567's own** parent so that packing
is the only changed variable. With the parent absent, the rerun cannot be started,
so there is **no produced checkpoint to hash** (sec.28). This file records the exact
recipe, the code SHAs, and the honest comparison table so a reviewer who has the
parents can execute a certifying rerun without re-deriving anything.

---

## 1. Code SHAs (the only intended variable)

| role | git commit | `goofspiel/training/stages.py` sha256 (git blob, LF) |
|---|---|---|
| **buggy** (rerun567 lineage) | `7366c85bc812d7c83c6b787dafd1896227d1b566` | `5a89664d9bcc70e0b82ac8155997e20bc8823e793bffc64d6e329320805b8501` |
| **fixed, value-slot** (S7-F1 Commit 1) | `7de8fd20e72e007c32f50baf99683db7512f971c` | `1f65a1f0dfed6b0f2a2db992faa15ebb71f9bc87725f7eff2632e4b80c2f71b7` |
| **fixed + fail-closed guard** (S7-F1 HEAD) | `39b1fee0e6621886223bd90cb66b81dd3a061690` | `48c25c210e3fbafb48644a7a53c6c0043df7d10fdb7cfa0e13d458a0beadd28d` |

The sha256 column is the **LF-normalized git-blob** hash, reproducible on any
platform via `git show <commit>:goofspiel/training/stages.py | sha256sum`; the git
commit SHA is the authoritative identifier. (Earlier drafts recorded the CRLF
working-file hashes `659ce37e...` / `b5e37a78...`; those are Windows-checkout
artifacts and are superseded here for platform-independence.)

The **only behavioral** difference between the buggy and fixed trees inside the
Stage7 correction is the teacher-policy packing loop (`run_stage7_redteam`):
positional -> value-slotted. The HEAD tree (`39b1fee`) additionally carries a
fail-closed length-contract guard around that loop; the guard is a **no-op on valid
input** -- the contract `len(teacher_policy)==len(self_actions)` is code-proven on
both teacher branches (EXACT + REFERENCE_NASH_Q), so it can only raise if a future
upstream change breaks the contract. The single behavioral variable on real data
therefore remains positional->value-slot. Everything else on the fixed tree (LR,
correction_steps, AdamW + param groups, frozen modules = none, replay = none,
teacher anchor = none, loss coefficients, the `_immediate_target` Q term, attack
generation, train/heldout split, attack-success metric, Stage7 promotion semantics,
Stage6, Stage5, GPU backend, FULL budgets) is byte-for-byte the rerun567
configuration -- see the S7-F1 sec.10 scope guard.

## 2. Recipe to produce a CERTIFYING fixed-column (when parents are on disk)

1. Check out `39b1fee0e6621886223bd90cb66b81dd3a061690` (the S7-F1 fix HEAD:
   value-slot packing + fail-closed guard).
2. Locate rerun567's Stage4 parent (sha256 prefix `c4e06e2c`) and its fresh GPU
   Stage5 checkpoint. Verify the Stage4 sha256 matches before use.
3. Start **Stage7 only** from that parent (do NOT re-run Stage4/5/6). Use rerun567's
   Stage7 configuration verbatim: `focused_correction`, FULL budgets, same seed,
   same attack generation, same train/heldout split, same GPU eval backend
   (S7-F1 sec.29-30 -- packing is the only changed variable; eval backend must match
   rerun567, not the CPU reduced-budget tripwires in sec.8 below).
4. Emit a new artifact `rerun567_stage7_indexfix_<timestamp>` recording
   `buggy_code_sha=7366c85...`, `fixed_code_sha=39b1fee...`, the parent Stage4/Stage5
   identities, and the **sha256 of the produced corrected checkpoint** (sec.28).
5. Evaluate the five layers of sec.3 and fill the sec.4 table's Fixed column.

## 3. Evaluation layers (S7-F1 sec.31)

1. train-attack set (the states corrected on)
2. heldout-same-family attacks
3. heldout-other-family attacks
4. teacher-agreement (teacher-NLL -- the robust regression signal; match-rate can
   rise while NLL blows up)
5. arena vs {random, heuristic, strong_nash, league, redteam}

## 4. Three-column comparison (S7-F1 sec.32) -- Fixed column BLOCKED

Robust = rerun567 pre-correction head; Buggy = rerun567 corrected head (real
measurements of a mis-indexed correction, from `provenance.json`); Fixed = the
value-slot fix corrected head. **Fixed is BLOCKED (parents absent) -- not estimated,
not fabricated.**

| layer | Robust (pre-corr) | Buggy corrected (rerun567) | Fixed corrected |
|---|---|---|---|
| train-attack match-rate | 0.467 | 0.417 | **BLOCKED** |
| heldout-same match-rate | 0.500 | 0.325 | **BLOCKED** |
| heldout-other match-rate | 0.500 | 0.360 | **BLOCKED** |
| teacher-NLL (lower better) | 2.32 | 12.09 | **BLOCKED** |
| arena delta vs random | +0.92 | -0.80 | **BLOCKED** |
| arena delta vs heuristic | -1.83 | -4.69 | **BLOCKED** |
| arena delta vs strong_nash | +0.42 | -4.84 | **BLOCKED** |
| arena delta vs league | +4.30 | -5.08 | **BLOCKED** |
| arena delta vs redteam | +5.46 | -5.00 | **BLOCKED** |

**Recovery formulae (S7-F1 sec.33), to apply once Fixed exists:**
- higher-better layers: `recovery = fixed_after - buggy_after` (and fraction of the
  regression recovered = `(fixed_after - buggy_after) / (robust - buggy_after)`).
- teacher-NLL: `recovery = buggy_NLL_after - fixed_NLL_after` (positive = recovered).

## 5. What CAN be shown now: production CODE-PATH fix evidence (NOT the sec.4 lineage)

These runs exercise the **exact fixed `run_stage7_redteam` code path** on the fixed
tree (`39b1fee`, value-slot packing + fail-closed guard; the guard is a no-op on
valid input so these numbers are identical to `7de8fd2`), but at **reduced CPU
budgets on a different, smaller attack set** (`attack_cases=6`,
`correction_steps=20`, `arena_games=0`). They are the S7-F1
sec.19/sec.20 tripwires -- **not** the rerun567 five-layer GPU arena eval -- so they
are reported here **separately** and are deliberately kept out of the sec.4 table.

| run | train slice | teacher-NLL before -> after | match before -> after | buggy-code NLL after (S7-D1) |
|---|---|---|---|---|
| seed=7, train=6 | includes gapped {1,3},{2,3} | **0.8449 -> 0.0000** (d -0.8449) | 0.333 -> 1.000 | **6.908** (blew up) |
| seed=43, train=2 | contiguous-only | **1.1073 -> 0.0000** (d -1.1073) | 0.000 -> 1.000 | 0.000 (packing-invariant) |

Reading (S7-F1 sec.20/sec.21): on the fixed tree the gapped-train teacher-NLL no
longer blows up (buggy 0.845 -> 6.908 becomes 0.845 -> 0.000), and the contiguous
control is unchanged vs buggy (sec.19 non-regression: packing is identity on
contiguous masks). Primary signal = teacher-card prob / NLL / slot round-trip, all
green; these are consistent with an indexing-dominant cause, but they do **not**
substitute for the sec.4 arena/heldout-other Fixed numbers.

## 6. Interpretation gate (S7-F1 sec.34-37) -- CANNOT be evaluated yet

sec.34-37 branch on the Fixed arena/heldout numbers, which are BLOCKED. The
production CODE-PATH evidence (sec.5) is consistent with sec.34-A (indexing-dominant)
but the arena layer is required to distinguish sec.34-A from sec.34-B
(teacher/train recover but arena still regresses -> approve S7-D2 amplifier work).
No interpretation verdict is asserted here. Per S7-F1 sec.49, no amplifier mitigation
(replay / teacher anchor / encoder freeze / lower LR / redesign) is implemented; this
work unit stops at the correctness fix and awaits review.

## 7. Second correctness defect flagged (S7-F1 sec.48) -- NOT fixed here

`goofspiel/training/stages.py:588` (`run_stage3_sft`) packs a teacher policy with the
**same positional indexing pattern** (`teacher_policy[b, i] = float(v)`), i.e. the
same defect class in Stage3. It is **out of scope** for S7-F1 and, critically, must
remain **unchanged** so that the sec.4 fixed-vs-buggy comparison isolates the Stage7
packing as the single variable. Flagged as `SECOND CORRECTNESS DEFECT SUSPECTED`
for a separate work unit; do not fold it into this branch.
