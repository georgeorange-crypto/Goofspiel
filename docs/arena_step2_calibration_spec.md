# Arena Step-2 Calibration — Specification (implement AFTER integration SHA)

**Status:** SPEC ONLY. No code lands until Arena Step-1 (`815a620`) is rebased onto
the gated `phase2/stage5-gpu` SHA and a combined-regression **integration SHA** exists.
This document fixes the architecture so implementation is mechanical at that point.

**Base of record for Step-1:** `commit=815a620ad4b35c0837bb96362ffbe282478b75ed`,
branch `arena/step1-budget-statistics`, base `6eca4c3`.

---

## Part A — Parent-run / lineage contract for calibration

### A.0 The problem being solved (verified in code at `815a620`)

- Stage6 discovers its role checkpoints ONLY from the artifact-dir convention paths:
  `<artifact-dir>/checkpoints/stage4_robust_rl.pt` (ROBUST),
  `<artifact-dir>/stage5_adaptive.pt` (EXPLOITER),
  `<artifact-dir>/checkpoints/stage3_sft.pt` (AGGRESSIVE) — via
  `_resolve_checkpoint`/`_THETA_CHECKPOINT_RELPATH` (coordinator.py:56-57,129-148).
- Stage7 discovers its P4 robust parent the same way (`stage4_robust_rl.pt`).
- `--init-from-checkpoint` threads into θ-stages ONLY (coordinator.py:244-291); it does
  **not** reach stage6/7.
- Single-stage CLI (`--stage stage6_league`) runs through `coordinator.run()` with
  `strict=False`. Under `strict=False`, a downstream stage that cannot resolve its parent
  does **not** raise — it "degrades honestly — mints its own seed and says so"
  (coordinator.py:212-215).

**Consequence:** a fresh calibration `--artifact-dir` with no parent checkpoints does NOT
get rejected. It SILENTLY runs Stage6/7 against random-init policies and reports a
throwaway-seed result. Wall-clock would be measurable but meaningless for calibrating
against formal1k. This is the failure to prevent.

### A.1 Hard constraints (do NOT violate)

1. **Do NOT copy or symlink** formal1k checkpoints into a fresh run dir under the
   convention relpaths to make discovery "just work" — that fabricates lineage
   (`git_commit`/parent-hash provenance would lie).
2. **Do NOT disable or weaken strict-lineage.** Keep every `_require_checkpoint(...,
   consumer=...)` call, `_DOWNSTREAM_ROLE` string, and the `strict` branch semantics
   byte-identical.
3. **Do NOT touch** Stage4/Stage5 algorithm/device, rank0-guard, or the correction body.
4. Additive only: new CLI flags default None; new coordinator kwargs default None and
   preserve today's auto-discovery when unset.

### A.2 Chosen mechanism — explicit parent pointing (option 2)

Add an **explicit parent map** the operator supplies for a calibration run, resolved
BEFORE dispatch, with a strict existence check that raises (never silently degrades).

**CLI (scripts/train_goofspiel_full.py), new flags (all default None):**

```
--stage6-parent-robust      PATH   # -> ROBUST role   (formal1k stage4_robust_rl.pt)
--stage6-parent-aggressive  PATH   # -> AGGRESSIVE    (formal1k stage3_sft.pt)
--stage6-parent-exploiter   PATH   # -> EXPLOITER     (formal1k stage5_adaptive.pt)
--stage7-parent-robust      PATH   # -> P4 robust     (formal1k stage4_robust_rl.pt)
--parent-manifest           PATH   # JSON alternative to the four flags (A.3)
```

Alternatively (equivalent, preferred for reproducibility) a single `--parent-manifest`.
Exactly one of {flags, manifest} may be given; giving both is an error.

**Config surface:** carry the resolved parent map in
`TrainingRunConfig.extra["parent_checkpoints"]` (mirrors how budgets ride in
`extra["budgets"]`; serializes into `resolved_config.json`), shape:

```json
{
  "stage6": {"robust": "<abs path>", "aggressive": "<abs path>", "exploiter": "<abs path>"},
  "stage7": {"robust": "<abs path>"}
}
```

**Coordinator resolution (new method `_parent_overrides()`):**
- Read `extra["parent_checkpoints"]` (None when unset → today's behavior, unchanged).
- In the stage6/stage7 dispatch, when an override is present it REPLACES the
  `_resolve_checkpoint(...)` result for that role.
- **Existence is mandatory when an override is supplied:** a supplied-but-missing path
  raises a `RuntimeError` naming the role + path (reuse the `_require_checkpoint`
  message shape; do NOT fall through to auto-discovery or to a minted seed). This is the
  behavior that turns the silent-degrade into an honest refusal.
- When NO override is supplied for a role, behavior is exactly as today (auto-discover
  from artifact-dir; `strict=False` single-stage still degrades-with-notice — that path
  is untouched and remains correct for genuine standalone runs).

**Provenance stamping (mandatory for calibration):** for every override actually used,
record in the stage result / report:
```
parent_checkpoints_used = {role: {"path": ..., "sha256": <hash of the .pt bytes>}}
```
Compute the sha256 by reading the checkpoint bytes at run time (RE-EXECUTE, do not trust
a stored field). This is what lets a calibration artifact point back at the exact
formal1k checkpoint.

### A.3 `--parent-manifest` JSON (preferred)

```json
{
  "source_run": "artifacts/runs/formal1k_20260902_191804",
  "source_commit": "<40-char SHA the formal1k run was produced at, if known>",
  "stage6": {
    "robust":     "artifacts/runs/formal1k_20260902_191804/checkpoints/stage4_robust_rl.pt",
    "aggressive": "artifacts/runs/formal1k_20260902_191804/checkpoints/stage3_sft.pt",
    "exploiter":  "artifacts/runs/formal1k_20260902_191804/stage5_adaptive.pt"
  },
  "stage7": {
    "robust":     "artifacts/runs/formal1k_20260902_191804/checkpoints/stage4_robust_rl.pt"
  }
}
```

The manifest is copied verbatim into the calibration artifact-dir and its resolved
absolute paths + computed sha256 are echoed into `resolved_config.json`.

### A.4 Calibration run identity (independent of `--profile`)

A calibration run MUST be un-mistakable for a certification run even when it passes
`--profile FULL` (needed to exercise the FULL code paths / budgets):

- New CLI: `--run-purpose {training,budget_calibration}` (default `training`), OR the
  narrower `--evaluation-purpose calibration`. Stored in
  `extra["run_purpose"]`.
- When `run_purpose == "budget_calibration"`:
  - Stamp `run_purpose="budget_calibration"` and `binding_promotion=false` in every
    report/artifact regardless of profile name.
  - Force the promotion path to `NOT_EVALUATED` even if `profile.name.upper()=="FULL"`
    (i.e. `emits_binding_promotion` is additionally gated by run_purpose). A calibration
    run never emits PROMOTE/REJECT.
- This is the enforcement layer on top of the existing tri-state
  (`decide_promotion`/`emits_binding_promotion`).

### A.5 Tests (must RE-EXECUTE the fact — see [[testing-principle]])

1. `test_calibration_parent_override_runs_against_real_parent`: fresh empty artifact-dir
   + explicit `--stage6-parent-*` pointing at a fixture checkpoint → Stage6 actually
   loads THOSE bytes (assert `parent_checkpoints_used[role].sha256` == sha256 recomputed
   from the fixture file), NOT a minted seed.
2. `test_calibration_missing_parent_refuses`: fresh dir + a supplied parent path that
   does not exist → RuntimeError naming the role+path. Must NOT degrade to a seed.
3. `test_no_override_preserves_autodiscovery`: no parent flags → byte-identical to
   today's standalone behavior (degrade-with-notice on empty dir; discover on populated
   dir). Guards additivity.
4. `test_calibration_run_purpose_suppresses_promotion`: `--profile FULL
   --run-purpose budget_calibration` → report carries `binding_promotion=false`,
   `run_purpose="budget_calibration"`, promotion == NOT_EVALUATED. Re-derive via
   `decide_promotion`.

---

## Part B — Orthogonal calibration sweep design

### B.0 Goal

Fit an identifiable cost model per dominant stage:

```
T  ≈  T0  +  a·games  +  b·blocks  +  c·bootstrap_iters
```

where `blocks = seeds × prize_sequences`. Varying multiple axes together (e.g.
20×5, 50×10, 100×20) only yields total-workload scaling and cannot separate
per-game cost from per-block fixed cost from bootstrap cost. So sweep ONE axis at a
time around a fixed operating point.

### B.1 Stage6 sweep (dominant workload)

Fixed point: `games=50, seeds=5, prize_sequences=3` unless measurement says otherwise.

- **games axis** (seeds=5, prize_seq=3 fixed): games ∈ {20, 50, 100}
- **seeds axis** (games=50, prize_seq=3 fixed): seeds ∈ {3, 10, 20}
- **prize_sequences axis** (games=50, seeds=5 fixed): prize_seq ∈ {1, 3, 5}
- **≥3 repeats** at each of the two dominant points (the fixed operating point and the
  largest games point) for variance/wall-clock stability.

### B.2 Stage7 sweep (dominant workload)

Fixed point: `attack_cases=200, heldout=200, arena_games=50, correction_steps=40`.

- **attack_cases axis** (others fixed): {100, 300, 500}
- **arena_games axis** (others fixed): {20, 50, 100}
- (correction_steps stays 40 — it is the certified optimizer-step count, not a sweep var.)
- **≥3 repeats** at the fixed operating point and the largest attack point.

### B.3 evaluate

`games_per_matchup ∈ {50, 100}` at `seeds=3`; single axis, low priority (cheap).

### B.4 What each run must emit (measurement surface — already present at `815a620`)

Per dominant stage, from the workload block:
- wall-clock seconds, raw games played, games/sec, actions/sec
- seeds, prize_sequences, matchups, total_blocks
- Stage7: attack candidates generated, failures, corrections, heldout tests,
  regression games, arena games, correction cost
- evaluate throughput

Plus the Part-A provenance (`parent_checkpoints_used` with sha256) and run identity
(`run_purpose=budget_calibration`, `binding_promotion=false`).

### B.5 Fitting + output

- One artifact-dir per (stage, axis-point, repeat): e.g.
  `artifacts/runs/_cal_stage6_games100_s5_q3_rep2/`.
- After collection: least-squares fit `T0, a, b, c` per stage from the orthogonal points;
  report per-game / per-block / per-bootstrap-iter marginal cost with R².
- ONLY THEN propose FULL preset numbers with an explicit safety margin. The provisional
  values in `budgets.py` (stage6 100/20/5, stage7 500/200/100/20, eval 100/20) stay TBD
  placeholders until this fit exists.

### B.6 Provenance requirements for every calibration run (governance rule 12)

Each artifact records: `git_commit` (the integration SHA), `git_branch`, `dirty=false`,
config, seed, `parent_checkpoints_used` (path + sha256), world_size, device,
`run_purpose=budget_calibration`, `binding_promotion=false`. A dirty tree run is marked
`DEV / NON-BINDING / NOT FOR PROMOTION` and never fed into the preset fit.

---

## Implementation order (when integration SHA lands)

1. Rebase `arena/step1-budget-statistics` (`815a620`) `--onto <phase2-gated-SHA>`;
   resolve the small stage5 device-hunk overlap on the integration branch; rerun the
   76-test regression → **integration SHA**.
2. On the integration branch, implement Part A (parent-pointing + run-purpose) + its 4
   tests. Commit `feat(arena): explicit calibration parent-pointing + run-purpose gate`.
3. Push; HPDPS-4 checks out that exact SHA (`dirty=false`).
4. Run the Part B orthogonal sweep against formal1k via `--parent-manifest`.
5. Fit the cost model; propose FULL presets with margin; STOP for review before writing
   final preset numbers into `budgets.py`.
