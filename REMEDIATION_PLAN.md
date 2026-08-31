# Goofspiel-13 Remediation Plan

> **Status of this document.** Every `file:line` below was read directly against the
> current `main` checkout (commit `ea09939`) — not inferred from docstrings, metric
> names, or the existing acceptance reports. Where a claim came from a delegated
> search, it was re-confirmed by reading the cited lines.
>
> **One-line verdict.** This repo is an **A-grade engineering skeleton wrapped around a
> research loop that is not yet connected.** The game engine, the joint-action
> `Q(s,a,b)` architecture, the RM+ matrix solver, P4 self-play mechanics, replay,
> target-EMA and the artifact/checkpoint system are real. But the P1→P2→P3→P4
> *parameter* curriculum, the NeuRD robust objective, the opponent/adaptive branches,
> "Mamba", P5/P6/P7, the benchmark, and the reasoning↔neural integration are **not
> closed loops.** Scaling steps on an H200 now would mostly *train the wrong things
> more expensively.*
>
> **Ordering philosophy.** Install the thermometer and stop the self-deception first
> (Phase 0), then fix each stage's *objective and data* (Phases 1–2), and only then
> *connect the curriculum* (Phase 3). Chaining broken stages together just propagates
> broken representations more expensively — but chaining is nonetheless **required before
> the first real robust run** (see the two-tier [H200 red line](#h200-red-line)), so it is
> the last thing before that gate, not an afterthought.

### Execution priority (authoritative ordering)

| Priority | Work | Blocks robust-only run? |
|---|---|---|
| **P0** | Honest evaluator + de-fake metrics (Phase 0) | ✅ |
| **P1** | NeuRD math fix (1.1) | ✅ |
| **P1** | carry/stake feature contract (1.2) | ✅ |
| **P2** | reachable-state coverage (2.1) | ✅ |
| **P2** | P2→P3 teacher dataset actually consumed (2.2b) | ✅ |
| **P3** | **P1→P3→P4 checkpoint chaining (3.1)** | ✅ **(first red line)** |
| P2b | real Mamba (2.3) | ❌ robust-only / ✅ full |
| P3b | opponent LSTM/Mamba/adaptive actually trained (3.2) | ❌ robust-only / ✅ full |
| P4 | neural ↔ reasoning integration (4.1–4.2) | full run only |
| P4 | real league (4.3) | full run only |
| P4 | real red-team correction (4.4) | full run only |
| P5 | statistical benchmark + promotion gates | release only |

> **Revision note (v2).** This revision resolves four structural problems from review:
> (1) the H200 gate is now **two-tier** — a first red line at P1→P3→P4 chaining (3.1) for a
> robust-only run, a second before the full opponent-aware run — so "chain the curriculum"
> and "don't scale broken stages" no longer contradict each other;
> (2) checkpoint reuse is split into **`init_from_checkpoint`** (new optimizer/step-0, for
> curriculum hand-off) vs **`resume_checkpoint`** (restores optimizer + RNG + step, for
> interrupted runs) — see [3.1](#phase-3);
> (3) the **P3↔P5 circular dependency is removed**: P3 (Robust Strategic SFT) sources targets
> only from solver/search/CFR/pseudo/symmetry and carries **no opponent-behavior**, keeping
> `Q_R ⊥ Q_A`; anything opponent-conditioned lives in P5;
> (4) **robust/adaptive view isolation** is explicit: `state.robust_view()` stays
> opponent-agnostic and a separate `state.adaptive_view()` carries `opponent_history`, so the
> robust head structurally cannot see opponent identity (a firewall, not a convention).
> Refinements also applied: NeuRD **equilibrium-convergence** test (not agreement-with-min),
> deterministic 2-checkpoint discrimination test (not a CI-flaky win-rate threshold),
> exploitability **naming discipline** (`full_game_exploitability` vs
> `one_step_matrix_nash_gap`), **two** carry features (`carry_norm` + `stake_norm`), P3
> **coverage buckets**, a checkpoint **lineage contract**, P5 **gradient-ownership** hard test
> (`‖Δθ_R‖==0`, `‖∇θ_A L_A‖>0`), and real Mamba (2.3) gates only the **full** run, not
> robust-only.

---

## Contents

- [Phase 0 — Install a thermometer + stop self-deception](#phase-0)
- [Phase 1 — Cheap fixes that directly target observed misbehavior](#phase-1)
- [Phase 2 — Make each stage learn the right target on the right data](#phase-2)
- [Phase 3 — Connect the curriculum (only meaningful now)](#phase-3)
- [Phase 4 — Integrate + close the outer loops](#phase-4)
- [Phase 5 — Final gates](#phase-5)
- [H200 red line](#h200-red-line)
- [Appendix: verified findings ledger](#appendix-verified-findings-ledger)

---

<a name="phase-0"></a>
## Phase 0 — Install a thermometer + stop self-deception

**Why first.** You are about to change ~10 subsystems. Right now *nothing honestly
measures whether a change helped*: the benchmark has no model argument, hard gates are
literal `True`, and the P7 "regression passed" test reads the boolean the code itself
wrote. **Before touching the furnace, fix the thermometer — and stop printing fake
readings on it.**

### 0.1 Minimal honest evaluation harness

**Problem.** `run_unified_benchmark(profile)` takes no model/checkpoint
(`goofspiel/training/benchmark.py:115`); `E2_N13_ROBUST` resolves to `Heuristic vs
Random` (`goofspiel/training/evaluation.py:22-24`, defaults `bot_a=BOT_HEURISTIC,
bot_b=BOT_RANDOM`). No trained checkpoint is ever evaluated anywhere.

**Change.** Add a small, honest evaluator that:
- accepts a `.pt` checkpoint path, loads `GoofspielModel`, runs greedy/`robust_policy_logits`;
- plays it (N=5 and N=7) vs `Random`, `Heuristic`, and `Nash` (`goofspiel/bots.py:48-50`);
- reports **win-rate**, **mean score-diff**, and an exploitability figure — *named
  according to what is actually computed* (see 0.3).

**0.3 — exploitability naming (research integrity).** The repo has a **real recursive
extensive-form solver**: `GoofspielExactSolver` with `F(A,B,R)` recursion + memoization
(`goofspiel/solver.py:496`, `_solve_chance` at `:719`, which folds continuation values
into each node's LP), and a carry-aware `GoofspielCarrySolver` (`:1219`, `_solve_chance`
at `:1387`). So a *true* full-game best-response value is computable for small N. The
evaluator must therefore expose **two clearly distinct metrics** and never conflate them:

| Metric name | What it computes | Valid range |
|---|---|---|
| `full_game_exploitability` | best-response gap vs the recursive solver's game value | small N only (solver budget) |
| `one_step_matrix_nash_gap` | duality gap of the *current-state* matrix only | any N (proxy) |

The proxy must never be labeled "exploitability" unqualified. This naming discipline is
locked in from the first commit.

**Files.** New `goofspiel/training/model_eval.py`; wire an optional `--eval-checkpoint`
into `scripts/train_goofspiel_full.py`.

**Acceptance.** Running it on the current `stage4_robust_rl.pt` produces real numbers (no
hardcoded rows). For small N, `full_game_exploitability` of the exact Nash bot is ≈0 and
of `Random` is clearly >0 — this validates the metric itself.

**Unit test — deterministic, never flaky (revised).** *Do not* assert vague policy
strength (e.g. "always-highest beats Random > 0.5") — in Goofspiel with random prize
order, one-shot resource spend, ties, and carry, always-highest is **not** provably
> 50%, and uniform-vs-random hovers at 0.5 with sampling noise, so such a test would
flake in CI. Instead test the one thing that must hold: **the evaluator actually reads
and uses the checkpoint.**
- `tests/unit/training/test_model_eval.py`: build two tiny deterministic models — ckpt A
  forced to `P(card 1)=0.999`, ckpt B forced to `P(card 3)=0.999` — on a fixed N=2/N=3
  scripted game whose outcome is exactly computable. Assert the evaluator observes
  `E[Δscore]_A ≠ E[Δscore]_B` and that each matches its hand-computed value.
- Real win-rate/exploitability benchmarks belong in an **integration test**, not unit CI.

### 0.2 The "de-fake" commit — replace every hardcoded success signal

**The scoreboard is fake; make it honest before trying to score higher.** None of these
are algorithm changes — they are truth-in-labeling changes so Phases 1–5 are
measurable.

| What | Where | Fix |
|---|---|---|
| Overall smoke PASS hinges only on stage 0 | `stages.py:907` (`summary["ok"] = bool(stage0.ok)`) | PASS must also require the 0.1 harness to beat Random by a margin, or at minimum surface a separate `algorithmic_ok` field instead of implying end-to-end success. |
| `opponent_model_usable=True`, `ece=0.0` are literals | `stages.py:624`, `:630`, `:645`, `:649` | Remove/compute. Until P5 trains a model (Phase 3.2), report `opponent_model_usable=False` honestly. |
| P5 "NLL" is a constant `log(n)` | `stages.py:590` (`prob = 1.0/len(opp_cards)`) → `:593` → `:621` | Either compute a real predictor NLL or rename the field `uniform_reference_nll` so it cannot be read as a model metric. |
| P7 regression booleans are literals | `stages.py:815-817`, `:839-840` | Do not write `True` until 4.4 actually re-runs the regression. Until then omit the field or write `null`. |
| Benchmark reference rows | `benchmark.py:132`, `:137`, `:166` (`"gate": "PASS_REFERENCE"`) and gate literals `:171-177` (`G1/G3/G4/G7 = True`, `G5/G6 = bool(rows)`) | Mark clearly as `REFERENCE_ROW_NOT_A_RESULT`; do not let them satisfy promotion. |
| P3 four metrics all equal `exact_anchor_count`; `pretraining_anchors_retained=1.0` | `stages.py:203-207` | Collapse to one honest `strategic_sft_samples`; delete `pretraining_anchors_retained` until P1→P3 chaining exists (Phase 3.1). |
| `exact_tool` stamps `NUMERICAL_EXACT` on the immediate matrix for non-terminal states | `goofspiel/reasoning/exact_tool.py:65` (real solver imported then `del solver` at `:39`) | For non-terminal states, stamp `Exactness.APPROXIMATE` (or run the real `GoofspielCarrySolver`). Do not claim exactness you did not compute. |
| "Mamba" is a GRU | `goofspiel/models/goofspiel_model.py:62-87` (`SimpleMambaMemory` = `Conv1d` + `nn.GRU`) | Rename to `PlaceholderSequenceMemory` now (real Mamba is Phase 2.3). Do not report "Mamba implemented" under a passed-spec banner. |
| 5 registry aliases → same P4 checkpoint | `stages.py:542-543` (`for kind in ("latest","best_robust","best_raw","best_search","best_generalization")`) | Stop aliasing until each dimension is actually evaluated (Phase 5). Register only `latest` for now. |

**Acceptance.** `grep` for the literals above returns zero matches in a "success"
position. A fresh smoke run's summary can no longer report PASS on stage-0 alone.

**Status — DONE (2026-08-31).** All 9 items applied and verified:

- Smoke `summary["ok"]` now `stage0.ok AND algorithmic_ok`, where `algorithmic_ok`
  is *computed* by re-executing the Phase 0.1 evaluator (`_smoke_algorithmic_check`)
  on the P4 checkpoint the run actually produced — beat-Random on a real mean
  score-diff. Verified the gate can FAIL (no checkpoint → `algorithmic_ok=False`),
  so it is a genuine discriminator, not a tautology. Fresh smoke run at N=3:
  `mean_score_diff=0.25` over 32 games, `ok=True` honestly.
- P5: `opponent_model_usable=0.0`/`False`, NLL renamed `uniform_reference_nll`,
  fabricated `ece=0.0` removed, gate → `CURRICULUM_BUILT_NO_MODEL`.
- P7: regression pass/fail written as `null` in the report and the metric keys
  omitted from `StageMetrics` (no `*_regression_passed` fabrication).
- `benchmark.py`: reference rows relabelled `REFERENCE_ROW_NOT_A_RESULT`; unrun
  hard gates (G1/G3/G4/G5/G6/G7) are `None` (falsy in `all(...)`), so promotion
  cannot ride on an unrun gate.
- P3: four aliased metrics collapsed to one honest `strategic_sft_samples`;
  `pretraining_anchors_retained` deleted.
- `exact_tool.py`: `NUMERICAL_EXACT` only on the last round (`prize_mask==0`);
  non-terminal immediate matrix → `APPROXIMATE` with an explanatory diagnostic.
- `SimpleMambaMemory` → `PlaceholderSequenceMemory` (attribute `inter_game_mamba`
  kept per §2.3 for a drop-in SSM swap).
- Registry: register only `latest` (dropped the 4 `best_*` aliases to one file).

Tests updated in lockstep (`test_training_pipeline.py` stage5/stage7 assertions
now pin the honest values). Full suite: **159 passed, 1 skipped**.

**Out-of-scope honesty gap found (not fixed — flagging only):**
`REQUIREMENTS_TRACE.md` row **ORDER-003** is marked `DONE` but its deliverable
`goofspiel/_core.cp310-win_amd64.pyd` does not exist in this Python 3.12
environment (it is a 3.10 compiled extension). `test_requirements_trace.py` fails
on this and did so before Phase 0.2 began — it is unrelated to any de-fake edit.
It is the same *class* of dishonesty (DONE with a missing artifact) but not one of
the 9 enumerated items; fixing it requires either building the C++ extension or
correcting the ledger, neither in Phase 0 scope.

---

<a name="phase-1"></a>
## Phase 1 — Cheap fixes that directly target observed misbehavior

These are near-one-line changes that plausibly explain the specific weird games you saw
(e.g. `carry=13, prize=3` and the model not protecting a 16-point stake).

### 1.1 Fix the NeuRD robust objective (it currently pulls best-case, not worst-case)

**Problem.** P4 trains with `neurd_loss` (`stages.py:481`), whose action value is
`q_matrix ... .max(dim=-1)` over opponent actions (`goofspiel/learning/game_theory/neurd.py:31`).
`q_robust` is *self* payoff (P4 regresses it toward `returns_t` = normalized
self−opp score, `stages.py:474`). So `max_b` = **best-case opponent** — the opposite of
robust. Meanwhile the RM+ solver, on the same sign convention, uses `-q` for the column
(`goofspiel/learning/game_theory/regret_matching_plus.py:52`) and `row_guarantee =
... .min(dim=-1)` over opponent actions (`:60`) — correct minimax. **The actor loss and
the Nash anchor are pulling in opposite directions.** More steps cannot fix this.

**Corroborating detail.** `neurd_loss` computes `col_policy` and immediately `del`s it
(`neurd.py:29-30`), and a *correct* function `row_action_regret` already exists
(`neurd.py:10-18`) — it properly contracts `Q` against the opponent column policy via
`torch.bmm` — **but is never called in training.**

**Change.** Replace the `neurd_loss` call at `stages.py:481-487` with `row_action_regret`
fed the opponent policy (`sol.col_policy` from the RM+ solve already computed at
`stages.py:472`). Keep `neurd_loss` only if you first fix its reduction to a policy-weighted
contraction (not `max`).

**Unit test (revised — test convergence, not a sign flip).** Asserting only that the
gradient "agrees with the `min` side of `row_guarantee`" is too weak *and* subtly wrong:
NeuRD action-regret uses the *expected* value against the opponent policy,
`Q(a,σ) = Σ_b σ(b)·Q(a,b)`, then `R(a) = Q(a,σ) − V`. When `σ` is the RM+ equilibrium
column policy this drives toward Nash, but "expected value vs the equilibrium opponent" is
**not** the same operator as "per-action `min_b Q(a,b)`". So test *equilibrium
convergence*, not the reduction:
- `tests/unit/learning/test_neurd_convergence.py`: run the actor update for K steps on
  known games — **Matching Pennies** (Nash `(0.5, 0.5)`), **Rock–Paper–Scissors** (Nash
  `(1/3,1/3,1/3)`), and **a matrix with a strictly dominated action**. Assert: (a) the
  dominated action's probability decreases monotonically; (b) the policy approaches the
  reference Nash within tolerance; (c) the matrix duality gap / exploitability decreases
  over the updates.
This tests *"does it converge to the minimax equilibrium"*, not merely *"was `max`
changed to `min`"*.

### 1.2 Put carry/stake into the model's features

**Problem.** The model's hand-built immediate feature uses `current_prize` only:
`goofspiel/models/goofspiel_model.py:272`
(`immediate = current_prize * sign / total`), and `_global_features` (`:205-214`) has
eight features, **none** of them carry. Meanwhile the *teacher target* correctly uses
`stake = current_prize + carry_pool` (`stages.py:74-75`), and the game logic uses stake
(`goofspiel/game/state.py:127`). So the model is handed a feature that *contradicts* the
label it's trained toward.

> Nuance (correcting the stronger early claim "the model can't see carry"): carry is in
> principle *recoverable* — non-terminal `carry = S_N − remaining_mass − current_prize −
> self_score − opp_score`, and `_global_features` has all those terms. So this is **not a
> strict information loss; it is a bad inductive bias plus a teacher/feature
> inconsistency.** The model must learn subtraction just to discover the stake the teacher
> already knows.

**Change.**
- `_global_features` (`goofspiel_model.py:198-214`): add **two** normalized features —
  `carry_norm = carry_pool / total` and `stake_norm = (current_prize + carry_pool) / total`
  — taking the input vector from 8 → **10** (bump `nn.Linear(8, 128)` at `:134` to
  `nn.Linear(10, 128)`). Giving the network the stake directly means it never has to learn
  the algebra `carry = S_N − remaining − current_prize − scores` just to recover what the
  environment already knows.
- `_pair_features` (`:259-274`): change `immediate` at `:272` to
  `immediate = (current_prize + carry_pool) * sign / total`, matching `_immediate_target`
  (`stages.py:74-75`) and the game logic (`game/state.py:127`).

**The contract this enforces:**

$$\text{Environment stake} \;=\; \text{Teacher label stake} \;=\; \text{Model feature stake} \;=\; \frac{\text{current\_prize}+\text{carry}}{S_N}$$

All three now agree; today only the middle two do.

**Acceptance.** In a scripted `carry=13, prize=3` state, the model's implied immediate
matrix magnitude reflects stake=16, not 3. Add a direct assertion for this, plus a test
that `stake_norm` and `carry_norm` appear in the global feature vector.

---

<a name="phase-2"></a>
## Phase 2 — Make each stage learn the right target on the right data

> **Architectural correction — break the P3↔P5 dependency cycle.** The first draft
> listed "opponent-behavior from P5 sessions" as one of P3's four teacher sources. But
> P5 is only implemented later (3.2), while the pipeline runs `P3 → P4 → P5`. That is a
> circular dependency (`P3 requires P5`, `P5 runs after P3`). It is also a *design*
> violation: mixing opponent-conditioned signal into the robust actor breaks the
> `Q_R ⊥ Q_A` separation. **Resolution: P3 is purely Robust Strategic SFT; anything
> opponent-conditioned moves to P5.**
>
> $$\boxed{\text{Robust training (P1–P4)} \quad\perp\quad \text{Adaptive/opponent training (P5+)}}$$

### 2.1 Broaden P1/P3 state coverage (reachable states, with a coverage artifact)

**Problem.** `_sample_states` (`stages.py:56-63`) only produces full-hand, score-0:0,
round-1, carry-0 opening states. P1/P3 never see endgames, score crises, sustained carry,
or asymmetric remaining cards — exactly the situations where you observed failures. No
number of steps teaches states never sampled.

**Change.** Sample reachable mid/endgame states (non-zero scores, partial masks, carry>0,
asymmetric `self_mask`/`opp_mask`). Make **coverage a first-class artifact**, not a single
opaque `teacher_samples=N`. Emit histograms + labeled buckets so a training report can be
audited:
- histograms: `round_index`, `carry`, `score_diff`, `remaining_cards`, `stake`, `hand_asymmetry`
- buckets: `OPENING`, `MIDGAME`, `ENDGAME`, `HIGH_CARRY`, `MUST_WIN`, `MUST_NOT_LOSE`,
  `SCORE_AHEAD`, `SCORE_BEHIND`, `ASYMMETRIC_HAND`

**Acceptance.** The training report states how many samples landed in each bucket; a
coverage test asserts every bucket is non-empty for a full-N run.

**Status — DONE (2026-08-31).** New `goofspiel/training/state_coverage.py`:
`sample_reachable_states(batch, n, step, seed)` builds states only via
`transition` (never hand-built), interleaving scripted crisis rollouts
(HIGH_CARRY / MUST_WIN / MUST_NOT_LOSE / SCORE_AHEAD / SCORE_BEHIND / MIDGAME /
ENDGAME / ASYMMETRIC_HAND) with random reachable games so every bucket is
covered even at batch 8 for N≥3. P1 (`run_stage1_pretrain`) and P3
(`run_stage3_sft`) now sample from it, emit `coverage_bucket_*` metrics, and
write an auditable `*_state_coverage.json` (bucket counts + 6 histograms:
round_index/carry/score_diff/remaining_cards/stake/hand_asymmetry). Test
`tests/unit/training/test_state_coverage.py` (7 tests) **re-executes**
classification (reclassifies sampled states via `classify_state`) and asserts a
reachability invariant on every sampled state, rather than reading the emitted
counts. N=2 legitimately has no MIDGAME (a 2-card game is opening+endgame only),
so the full-coverage guarantee is stated for N≥3.

### 2.2 P3 = **Robust Strategic SFT** (real, distinct teacher sources — no opponent signal)

**Problem.** P2 writes `teacher_dataset.jsonl` (`stages.py:241`) that **no one consumes**;
P3 (`run_stage3_sft`, `:160-230`) ignores it and uses `_sample_states` + `_immediate_target`
(`:182-187`). P3's four SFT metrics are **all the same `exact_anchor_count`** (`:203-206`).

**Change (2.2b — the part that gates the robust run).**
- Make P3 actually load and train on `teacher_dataset.jsonl`.
- Give P3's teacher anchors **distinct real, robust-only sources** so its four metrics
  measure four different things:
  - **Exact** — recursive solver for small N (`goofspiel/solver.py:496` / carry solver `:1219`)
  - **Search** — `goofspiel/reasoning/search.py` (SM-MCTS / GT-CFR)
  - **CFR** — regret-matching solve over reachable matrices
  - **Pseudo** — high-confidence *robust* self-labels
  - (plus symmetry / strategic-corpus anchors)
  - **NOT** opponent-behavior — that is P5's job (see the cycle-break note above).

**Acceptance.** The four P3 sample counts differ; P3 loss responds to
`teacher_dataset.jsonl` content; no P3 metric derives from opponent sessions.

**Status — DONE (2026-08-31).** New `goofspiel/training/teacher_dataset.py` with
four genuinely distinct robust-only sources differing by *algorithm and search
depth*: `CFR` (RM+ equilibrium of the immediate matrix), `SEARCH` (depth-1
lookahead folding one ply of continuation, budget-gated), `EXACT` (full-game
recursive Nash via the carry solver's `policy_map`, small-N only), `PSEUDO`
(confidence/entropy-gated robust self-labels). P2 (`run_stage2_semi_supervised`)
now writes all four to `teacher_dataset.jsonl` with per-source counts that
differ (e.g. N=5: CFR 48 / SEARCH 37 / EXACT 33 / PSEUDO 44); P3
(`run_stage3_sft`) loads that file and trains the robust policy toward the
*stored* source-specific teacher policy (falls back to on-the-fly solves only
when the file is absent). None of the sources uses opponent behaviour, so
`Q_R ⊥ Q_A` holds. Tests `tests/unit/training/test_teacher_dataset.py` (4 tests):
the discriminating one builds two datasets differing ONLY in the stored teacher
policy and asserts P3's first-step loss differs — proving P3 consumes the file
rather than re-deriving targets; another re-executes CFR vs EXACT on a
continuation-sensitive state and asserts the policies differ.

### 2.3 Real Mamba (keep LSTM) — *does not gate a robust-only run*

**Problem.** `SimpleMambaMemory` (`goofspiel_model.py:62-87`) is `Conv1d(depthwise)` +
`nn.GRU` — a GRU, not a state-space model. The spec requires both LSTM **and** Mamba.

**Change.** Introduce a genuine selective-SSM (e.g. `mamba-ssm` if the H200 toolchain
allows, else a faithful minimal SSM scan) as `inter_game_mamba` (`:180`); keep
`intra_game_lstm` (`:178`). Preserve the `OpponentMemoryBatch` caller interface so the
rest of the model is untouched.

> **Scheduling note.** Mamba lives entirely in the inter-game memory feeding the
> *adaptive* branch. The P4 robust actor does not consume it, so 2.3 can be developed in
> parallel with the first robust-only run and only gates the **full** run (see the
> [two-tier red line](#h200-red-line)).

**Acceptance.** The module is a scan-based SSM (state recurrence in the time dimension),
not an `nn.GRU`; `parameter_count_by_module()` (`:384-400`) reports it under a truthful
name.

**Status — DONE (2026-08-31).** `PlaceholderSequenceMemory` (Conv1d + `nn.GRU`) is
replaced by `SelectiveStateSpaceMemory` in `goofspiel/models/goofspiel_model.py` — a
genuine selective SSM (Mamba/S6): input-dependent Δ, B, C (`x_proj`/`dt_proj`), a diagonal
stable state matrix `A = -exp(A_log)`, a skip `D`, a short causal depthwise conv, and a
SiLU gate. The recurrence `hₜ = exp(Δₜ⊙A)⊙hₜ₋₁ + (Δₜ⊙Bₜ)xₜ; yₜ = Σ Cₜ·hₜ + D⊙xₜ` is an
explicit **scan over the games (time) dimension**. `inter_game_mamba` is now this module;
`intra_game_lstm` is unchanged; the `OpponentMemoryBatch(game_summary_sequence, valid_mask)`
caller interface is preserved; `parameter_count_by_module()["mamba_memory"]` reports it
truthfully. Tests (`tests/unit/models/test_selective_ssm_memory.py`, 4) re-execute the
defining facts rather than read a name: (1) no GRU/LSTM/RNN survives and the SSM params
exist; (2) an independent re-implementation of the scan from the module's own weights
reproduces `forward()` exactly (a GRU could not); (3) Δ⊙A is input-**selective** (S6, not
LTI); (4) perturbing only game 0 changes the last-step output across a sequence longer than
the conv width — so the scan, not the local conv, carries the long-range dependency. All 4
new + 5 existing model tests pass.

---

<a name="phase-3"></a>
## Phase 3 — Connect the curriculum (only meaningful now)

> Deliberately **not** step 1. Chaining stages whose objectives/data were wrong (Phases
> 1–2) would just carry bad representations forward. Now that each stage learns the right
> thing, wiring them yields a real curriculum.

### 3.1 P1 → P3 → P4 checkpoint chaining

**Problem.** Each stage freshly random-inits: `run_stage1_pretrain` (`stages.py:94`),
`run_stage3_sft` (`:175`), `run_stage4_robust_rl` (`:414`). The **only** `load_state_dict`
in the file is `:416` — P4's target net copying its *own* fresh init. The orchestration
layer has no seam for chaining either: `coordinator.py:71-117` and `run_smoke_pipeline`
(`:888-895`) call each stage with only `steps/batch_size/out_dir/device/n_cards`;
`stage1.checkpoint`/`stage3.checkpoint` are captured only to write into the summary JSON
(`:889`, `:893`), never fed forward.

**Change — two *distinct* interfaces, never conflated.** "Inherit weights across a stage
boundary" and "resume a crashed run" are different operations; a single `resume_from`
invites the research accident *"I thought it resumed, it only loaded weights."* Split them
from the start:

```python
init_from_checkpoint: str | None   # stage transition P1→P3→P4: load θ ONLY;
                                   # fresh optimizer / scheduler / global_step; replay may reset
resume_checkpoint:    str | None   # crash recovery: restore model + target_model + optimizer
                                   # + scheduler + scaler + replay metadata + RNG + global_step
```

- Stage chaining uses `init_from_checkpoint`: `stage1 → stage3(init_from=...)` and
  `stage3 → stage4(init_from=...)`, threaded through both `coordinator.py:71-117` and
  `run_smoke_pipeline` (`stages.py:888-895`).
- Mid-run restart uses `resume_checkpoint` and restores full training state.

**3.1b — lineage contract (kills this bug class permanently).** Extend the existing
`CheckpointMetadata` (`goofspiel/training/checkpoint.py:17-27`, which already has
`checkpoint_id`, `training_stage`, `global_step`, `git_commit`) with:
`parent_checkpoint_id`, `init_checkpoint_id`, `model_config_hash`, `dataset_manifest_ids`,
`teacher_dataset_ids`, `optimizer_reset`. Use the existing `sha256_file` (`:30`) to log at
startup, e.g. P4: `INIT_FROM stage3_sft.pt sha256=… ; TARGET_INIT_FROM online_loaded_weights`.
Lineage should read like `stage4_robust_001 ⟵ stage3_sft_004 ⟵ stage1_pretrain_006`, not
merely "`stage4.pt` exists".

**Acceptance.** Assert **`θ_{P3,t=0} == θ_{P1,final}`** on the shared encoder (byte-equal
parameters before the first P3 step) and likewise `θ_{P4,t=0} == θ_{P3,final}`. This single
assertion would have caught today's disconnect immediately.

**Status — DONE (2026-08-31).** Two distinct, never-conflated interfaces added to
`goofspiel/training/checkpoint.py`: `init_from_checkpoint(model, path)` (stage transition —
loads θ ONLY, guarded by a `model_config_hash` architecture check, leaves optimizer/step at
fresh) and `resume_checkpoint(model, path, optimizers, target_model)` (crash recovery —
restores model + every optimizer + target network + `global_step` + rng). `stages.py` gained
`_apply_init_or_resume` (raises if both are passed — the split's whole point) and threads
`init_from_checkpoint` / `resume_checkpoint` params through `run_stage1_pretrain`,
`run_stage3_sft`, `run_stage4_robust_rl`; `run_smoke_pipeline` chains
`P1.checkpoint → P3(init_from=…) → P4(init_from=…)`, and P4 seeds its **target** net from the
inherited θ (not a fresh init). `coordinator.py` exposes both fields on `TrainingRunConfig`.
**3.1b lineage:** `CheckpointMetadata` extended with `parent_checkpoint_id`,
`init_checkpoint_id`, `model_config_hash`, `dataset_manifest_ids`, `teacher_dataset_ids`,
`optimizer_reset`; `validate_checkpoint_resume` surfaces them. A real run now records
`stage4_robust ⟵ stage3_sft ⟵ stage1_pretrain` with `optimizer_reset=True`. Tests
(`tests/unit/training/test_checkpoint_chaining.py`, 6) re-execute the facts:
(1) init copies θ only while the optimizer stays empty, resume restores optimizer state +
step; (2) init and resume are mutually exclusive; (3) config-hash guards an architecture
mismatch; (4) **byte-equal `θ_{P3,t=0}==θ_{P1,final}` and `θ_{P4,t=0}==θ_{P3,final}`** on the
shared encoder (plus the P4 target net), and with steps>0 the encoder provably *moves* (P3
trains from the inherited θ, not a frozen copy); (5) lineage metadata is recorded. Full smoke
pipeline stays `ok=True / stage0_ok=True / algorithmic_ok=True`. All 6 new + 19 existing
checkpoint/pipeline tests pass.

### 3.2 Actually train the opponent/adaptive branches → then P5 earns its name

**Problem.** In the final `stage4_robust_rl.pt`, the opponent-memory (LSTM/Mamba),
`opp_short/long/fused` heads, and the adaptive branch are effectively still at
initialization: P1 supervises only `opponent_fused_logits` (`stages.py:116`) and is not
chained in; P3 doesn't train them; P4 trains only `q_robust` + `robust_policy_logits`
(`:472-488`); P5 trains nothing at all (see ledger).

**Change.** Add supervised losses for `opponent_short_logits` and
`opponent_long_logits` (not just fused) in P1; once chaining (3.1) lands, let P5 *train*
an opponent model on the P5 sessions (real backprop, save a checkpoint) instead of
emitting a constant-NLL diagnostic. Only then compute and report a real NLL/ECE.

**3.2b — explicit gradient ownership (do not rely on `.detach()` alone).** The core
Idea-Fidelity rule is *Adaptive must not pollute Robust*. Today the firewall is only the
`.detach()` calls at `goofspiel_model.py:352-353` — one refactor away from silently
leaking gradient into the robust backbone. P5 must therefore **freeze** the robust
parameters explicitly rather than optimizing `model.parameters()`:

| Frozen (robust) | Trained (adaptive/opponent) |
|---|---|
| public encoder / backbone, robust Q heads, robust actor | LSTM, Mamba, opponent heads (short/long/fused), adaptive FiLM, adaptive matrix-CNN, adaptive Q, adaptive actor |

**Hard unit test.** After each P5 step assert both: `‖Δθ_R‖ == 0` (robust params
unchanged) **and** `‖∇θ_A L_A‖ > 0` (adaptive params actually receive gradient). This is
stronger than trusting `.detach()` and pins the `Q_R ⊥ Q_A` firewall in CI.

**Acceptance.** P5 saves a non-`None` checkpoint; robust params are provably unchanged by
P5; its opponent-model NLL is measured and beats the uniform reference on scripted regimes.

**Status — DONE (2026-08-31).** P5 (`run_stage5_adaptive`) now TRAINS the opponent/adaptive
branch instead of emitting a constant-NLL diagnostic. It builds real multi-game sessions
(`games_per_session=3`, so the inter-game Mamba has genuine cross-game context), featurizes
every decision point into `(public, intra-game history → LSTM, prior-game summary sequence →
Mamba, opponent-action target)`, and supervises `opponent_fused_logits` (+ auxiliary
short/long) against the scripted regime's true next-action. It chains via
`init_from_checkpoint` (P4→P5) and saves `stage5_adaptive.pt` with full lineage. Real
NLL/accuracy/ECE are computed and reported; convergence is monotone (n=5: gain-over-uniform
0.013→0.032→0.117 and accuracy 0.51→0.62→ as steps go 10→30→60).
**3.2(a):** P1 now also supervises `opponent_short_logits` and `opponent_long_logits` (not
just fused) — previously those two heads had no P1 gradient at all.
**3.2b firewall (explicit, not `.detach()`-only):** added `robust_parameters()` /
`adaptive_parameters()` (an *exact* partition — `assert_partition_is_complete()` guards
disjoint ∧ exhaustive over all 32 modules) and `set_robust_requires_grad(False)`. P5
optimizes ONLY `adaptive_parameters()` with robust frozen, and asserts the firewall *in the
run itself* every step: no robust param may carry gradient (`grad is None`), and
`‖Δθ_R‖₁ == 0` end-to-end while `‖∇θ_A‖ > 0`. The dead `game_summary_projector` module
(declared, never used) is now wired as the SSM input projection, so every adaptive param is
live. Tests (`tests/unit/training/test_adaptive_firewall.py`, 3) re-execute the facts:
(1) the partition is exact; (2) freezing robust yields `grad is None` on every robust param
while adaptive still flows; (3) end-to-end P5 leaves robust byte-equal the inherited P4
weights in the *saved* checkpoint, moves adaptive, saves a non-None checkpoint, and beats the
uniform NLL. `test_training_pipeline.py::test_stage5_trains_opponent_model_behind_firewall`
updated to the new contract (was `opponent_model_usable==0.0`, checkpoint None). Full smoke
pipeline stays `ok=True / stage0_ok=True / algorithmic_ok=True`; P5 lineage records
`stage5_adaptive ⟵ stage4_robust ⟵ stage3_sft ⟵ stage1_pretrain`.

---

<a name="phase-4"></a>
## Phase 4 — Integrate + close the outer loops

### 4.1 Wire the trained checkpoint into the reasoning router

**Problem.** The reasoning layer and the neural net are **adjacent, not integrated.**
`GameAgent.__init__` takes `model_version: str` (`goofspiel/reasoning/agent.py:14`) — a
provenance label, never a model or `.pt` path. Every Q the router uses comes from the
handcrafted `immediate_q_matrix` (`goofspiel/training/teachers.py:23-32`), via
`router.py:37-40`, `search.py:38/84`, `exact_br.py:25`. No `GoofspielModel`,
`load_state_dict`, or forward pass exists anywhere under `goofspiel/reasoning/`. The
neural outputs `q_robust`/`robust_policy_logits` are consumed **only** in
`goofspiel/training/*`.

**Change.** Give the router an optional trained-model provider; when present, source the
robust Q/policy from the model (falling back to `immediate_q_matrix` only when absent or
out of budget).

### 4.2 Revive the dead adaptive → safe-mixture cascade

**Problem.** `router.py:107` calls `final_decision(tools, ...)` without adaptive args;
`decision.py:73` gates the adaptive branch on `adaptive_result is not None and
opponent_belief is not None`, so it never runs — meaning `safe_exploit_mixture` is never
called and the **robust floor inside it** (`goofspiel/reasoning/safe_mixture.py:42`) is
never exercised. Worse, `state.robust_view()` (`state.py:52`) drops `opponent_history`,
so the adaptive tier has no input even if invoked.

**Change.** Pass a real `opponent_belief` (from 3.2's trained opponent model) and
`adaptive_result` into `final_decision`. **Do not modify `robust_view` to keep
`opponent_history`** — that would be an architecture regression. The design deliberately
separates an opponent-*agnostic* robust value from an opponent-*conditioned* adaptive
value:

$$Q_R(s,a,b) \;\perp\; \text{opponent history} \qquad\text{vs.}\qquad Q_A(s,h,a,b)\ \text{conditioned on it}$$

So instead:
- `state.robust_view()` (`state.py:52`) **stays** opponent-agnostic — explicitly
  documented as *containing no opponent private/history/adaptation information*. The
  router keeps using it for the robust tiers.
- Add `state.adaptive_view()` carrying current-game history, opponent action history,
  session summary, LSTM memory, Mamba memory, and opponent belief — used only by the
  adaptive tier.

This guarantees `robust branch = opponent-agnostic` and `adaptive branch =
opponent-conditioned` as a structural invariant, not a convention.

**Acceptance.** A test where exploiting a known-biased opponent (via `adaptive_view`)
raises value **without** dropping the guaranteed robust value below `robust_value −
epsilon`; plus an assertion that `robust_view()` exposes no opponent-history fields.

> **Status — DONE (2026-08-31).** 4.1: new `goofspiel/reasoning/model_provider.py`
> `TrainedModelProvider` wraps a loaded `GoofspielModel`; `ToolRouter(..., model_provider=)`
> and `GameAgent(checkpoint=)` source robust Q/masks from `q_robust` (public state only —
> no history/memory, so `Q_R ⊥ opponent` holds for the model too), falling back to
> `immediate_q_matrix` on absence or any provider error. The router trace records
> `q_source` (`model_q_robust` vs `immediate_q_matrix`). 4.2: `state.py` keeps
> `robust_view()` opponent-agnostic (documented; new `exposes_opponent_information()`
> re-checks every field in `OPPONENT_INFORMATION_FIELDS`) and adds real opponent-conditioned
> fields (`opponent_belief`, `opponent_memory`, `current_game_history`) surfaced by
> `adaptive_view()`. The router now derives a belief from the adaptive view (explicit
> injection wins; else the trained `opponent_fused` head), builds an `EXACT_BEST_RESPONSE`
> adaptive candidate, and threads both into `final_decision`, reviving the
> `safe_exploit_mixture` cascade. `safe_mixture.py` was corrected: the robust floor now
> (a) excludes illegal padded columns via a new `opponent_mask` arg — a `0 >= V_R` from a
> pad column would otherwise kill every exploit — and (b) defaults `robust_value` to the
> policy's **minimax worst case over legal columns** (a real guarantee), not the
> belief-weighted average. Tests (`tests/unit/reasoning/test_model_router_integration.py`,
> 5) **re-execute** every fact: belief-value and worst-case are recomputed from
> (policy, Q); a counterfactual proves the floor refuses an unsafe exploit (α→0, value
> unchanged); `robust_view()` leaks nothing; and the provider genuinely sources Q + belief
> from a real chained P1 checkpoint. 23/23 reasoning+learning tests pass, 0 regressions.

### 4.3 P6 league plays real checkpoint snapshots

**Problem.** League agents carry `checkpoint_path=None` (`stages.py:694`); cross-play
substitutes fixed baselines keyed by role (`:704-716`), and those "baselines" are
one-liners: `create_baseline("PPO")` returns the **uniform** distribution
(`goofspiel/training/baseline_algorithms.py:42-43`), while "CFR+/Minimax-Q" are
row-min-then-softmax over the immediate matrix (`:50-51`). So "league cross-play" is a
3×3 round-robin of uniform + two heuristics, unrelated to any trained agent.

**Change.** Populate `checkpoint_path` from real snapshots and have `_play_policy_match`
play the loaded models. Keep the handcrafted baselines only as labeled *reference*
opponents.

### 4.4 P7 does a real focused fine-tune and re-runs the regression

**Problem.** P7 writes a dict literally named `"training_plan"` (`stages.py:806-813`)
that nothing consumes; it never loads/trains/saves a checkpoint (returns `None`, `:843`);
and it writes `original_attack_regression_passed=True` (`:815`) directly. The tests are
tautologies: `tests/unit/training/test_training_pipeline.py:325-326` assert the boolean
the code wrote; `:139` asserts the hardcoded `== 1.0`.

**Change.** Actually fine-tune the current checkpoint on the correction set, save an
improved checkpoint, **re-play** the attack states through the 0.1 harness before/after,
and write the *measured* pass/fail. Rewrite the tests to re-run the regression, not read
a field.

---

<a name="phase-5"></a>
## Phase 5 — Final gates

**Problem.** `run_unified_benchmark` has no model argument (`benchmark.py:115`); E2 is
Heuristic-vs-Random (`evaluation.py:22-24`); E3/E4/E7 are hardcoded reference rows
(`benchmark.py:132/137/166`); hard gates are literal `True` (`:171-177`); and the five
registry aliases all point at the same P4 checkpoint (`stages.py:542-543`).

**Change.** Rewrite the hard gates so `best_robust` / `best_search` /
`best_generalization` are produced by **their own** evaluations (built on the 0.1
harness) and can select *different* checkpoints. No gate may be a literal; no alias may
be registered without a distinguishing evaluation.

**Acceptance.** On a run where two checkpoints genuinely differ, the registry aliases can
resolve to different files; every gate value traces to a computed metric.

---

<a name="h200-red-line"></a>
## H200 red line (two tiers)

The earlier single red line ("finish Phases 0–2") was **self-contradictory**: it would
have permitted a full H200 run while `P1→P3→P4` was still unchained — precisely one of
the "scale cannot help, only makes it more expensive" defects. Chaining must gate the
*first* real run. Conversely, Mamba (2.3) and the adaptive branch (3.2/Phase 4) do **not**
gate a robust-only run, because the P4 robust actor does not depend on inter-game memory.
So there are two distinct red lines:

| Long-run type | Must be complete first |
|---|---|
| **Robust-only run** (`P1→P3→P4`, validating Robust Q + NeuRD + RM+ + curriculum) | Phase 0 + Phase 1 (1.1, 1.2) + 2.1 (coverage) + 2.2b (P2→P3 consume) + **3.1 checkpoint chaining** |
| **Full Robust+Adaptive run** (`P0→P7`) | All of the above **+** 2.3 real Mamba + 3.2 adaptive/opponent training + Phase 4 (integration, league, red-team) |

**The single most important change vs. the first draft: `3.1 chaining` is now inside the
first red line.** A robust-only run with unchained stages is forbidden.

- **Scale *can* help:** narrow P1/P3 state distribution and low step counts — once the
  data is broadened (2.1) and chained (3.1), more steps pay off.
- **Scale *cannot* help — only makes errors more expensive:** broken stage chaining
  (3.1), the reversed NeuRD objective (1.1), the carry/feature inconsistency (1.2), and
  the empty benchmark (Phase 5). These are structural — *training longer just trains the
  mistakes in more expensively.*
- **Scale is orthogonal (does not gate robust-only, does gate full):** real Mamba (2.3)
  and untrained adaptive/opponent branches (3.2) live entirely in the adaptive path and
  can be developed in parallel with the first robust run.

---

<a name="appendix-verified-findings-ledger"></a>
## Appendix: verified findings ledger

Each row was confirmed by reading the cited line(s) in commit `ea09939`.

| # | Finding | Evidence (file:line) | Verdict |
|---|---|---|---|
| 1 | P1/P3/P4 each random-init; no stage loads a prior checkpoint | `stages.py:94`, `:175`, `:414`; only `load_state_dict` is P4 target self-copy `:416` | Confirmed |
| 1b | Orchestration has no chaining seam; checkpoints only written to summary | `coordinator.py:71-117`; `stages.py:888-895` (`:889`,`:893`) | Confirmed |
| 1c | Overall smoke PASS depends only on stage 0 | `stages.py:907` | Confirmed |
| 2 | NeuRD actor uses `max_b` (best-case) on self-payoff Q | `neurd.py:31`; Q sign from `stages.py:474`; `neurd_loss` wired at `:481` | Confirmed |
| 2b | RM+ solver is correct minimax (`-q` column, `min` guarantee) | `regret_matching_plus.py:52`, `:60` | Confirmed |
| 2c | `col_policy` computed then `del`'d; correct `row_action_regret` never called | `neurd.py:29-30`, `:10-18` | Confirmed |
| 3 | Model immediate feature ignores carry; global features have no carry | `goofspiel_model.py:272`, `:205-214` | Confirmed |
| 3b | Teacher target *does* use stake=current+carry (inconsistency) | `stages.py:74-75`; game logic `game/state.py:127` | Confirmed |
| 4 | "Mamba" is Conv1d + `nn.GRU` | `goofspiel_model.py:62-87` (`:73-74`) | Confirmed |
| 5 | P3 four SFT metrics all equal one count; `anchors_retained=1.0` hardcoded | `stages.py:203-207` | Confirmed |
| 5b | P2 teacher dataset written but never consumed | `stages.py:241` (write); no reader in P3 `:160-230` | Confirmed |
| 6 | P1/P3 training states are opening-only | `stages.py:56-63` | Confirmed |
| 7 | GUI "Trained Smoke N=7" bot not in `main`; app loads no `.pt` | `bots.py:48-50`; no `.pt` load in `app.py` | Confirmed |
| 8 | P5 does not train/load/save; success flags are literals | `stages.py:556-655` (`:582-590`,`:624`,`:630`,`:645`,`:649`,`:654`) | Confirmed |
| 8b | P5 "NLL" is constant `log(n)` (uniform `prob`) | `stages.py:590`,`:593`,`:621` | Confirmed |
| 9 | P6 agents `checkpoint_path=None`; cross-play uses baselines | `stages.py:694`, `:704-716` | Confirmed |
| 9b | "PPO" baseline is uniform; "CFR+/Minimax-Q" are 1-step row-min heuristics | `baseline_algorithms.py:42-43`, `:50-51` | Confirmed |
| 10 | P7 writes a `training_plan` nothing consumes; regression booleans literal; returns `None` | `stages.py:806-817`, `:839-843` | Confirmed |
| 10b | P7 tests read the written boolean (tautology) | `tests/unit/training/test_training_pipeline.py:325-326`, `:139` | Confirmed |
| 11 | Benchmark takes no model; E2 = Heuristic vs Random; gates literal | `benchmark.py:115`, `evaluation.py:22-24`, `benchmark.py:171-177` | Confirmed |
| 11b | E3/E4/E7 are hardcoded `PASS_REFERENCE` rows | `benchmark.py:132`, `:137`, `:166` | Confirmed |
| 12 | Five registry aliases → same P4 checkpoint, no per-dimension eval | `stages.py:542-543` | Confirmed |
| 13 | Reasoning layer never loads a model; `model_version` is a string | `reasoning/agent.py:14`; no `GoofspielModel`/`load_state_dict` under `reasoning/` | Confirmed |
| 13b | Reasoning Q source is handcrafted `immediate_q_matrix` | `teachers.py:23-32`; `router.py:37-40`; `search.py:38/84`; `exact_br.py:25` | Confirmed |
| 13c | Adaptive→safe-mixture cascade + robust floor is dead code from the router | `router.py:107`; `decision.py:73`; `safe_mixture.py:42`; `state.py:52` drops history | Confirmed |
| 13d | `exact_tool` stamps `NUMERICAL_EXACT` on non-terminal immediate matrix; real solver `del`'d | `reasoning/exact_tool.py:65`, `:39` | Confirmed |

### What is genuinely solid (do not rewrite)

| Area | Evidence |
|---|---|
| Game engine: simultaneous actions, carry, tie-discard, bitmask, pure transition | `goofspiel/game/state.py` (`:127` stake, `:41-74`) |
| True joint-action `Q(s,a,b) ∈ R^{N×N}` (not a degenerate DQN) | `goofspiel_model.py:320-321` |
| Robust/adaptive detach firewall | `goofspiel_model.py:352-353` |
| RM+ matrix solver (correct minimax), target EMA, curriculum, replay, checkpoint registry | `regret_matching_plus.py`; `stages.py:494-497` (EMA), `:424` (curriculum) |
| Variable-N rank encoding, Transformer, relational path, matrix CNN, ensemble heads | `goofspiel_model.py:32-46`, `:139-168`, `:170-196` |
