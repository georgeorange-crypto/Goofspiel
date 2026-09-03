# S7-D1 — ROOT CAUSE CANDIDATE **CONFIRMED**

**Work unit:** S7-D1 — Focused Correction Failure Diagnosis (diagnosis only; no
production fix applied). **Date:** 2026-09-04. **Audited code:** HEAD `7366c85`
(== rerun567 integration commit), `stages.py` sha256 `659ce37e…`.

---

## Verdict (one line)

The rerun567 Stage7 `focused_correction` regression is a **target/loss indexing
bug (class D1-B)**: the KL policy target is packed by **positional** index while
the policy head is **value-slotted**, so on any non-contiguous ("gapped") legal
mask the teacher's mass lands on the wrong logit slot and the objective pushes
probability *away* from the teacher's true card. A **co-occurring amplifier**
(full-network AdamW on 6.14M params, 0 frozen, **no** general replay / anchor —
despite the report advertising `retain_general_replay_fraction: 0.25`) spreads
the per-state policy damage across the shared trunk into every bucket.

> **Answering the central question** ("是实现错了，还是实现正确但优化/目标设计不稳定"):
> **实现错了 (implemented wrong).** It is a concrete index-misalignment defect,
> not an unstable-but-correct optimizer. Proven by holding LR/steps/optimizer/
> init/loss constant and flipping *only* the packing — the regression appears
> iff the packing is positional and iff the mask is gapped.

---

## 1. The defect — exact code path (CODE-PROVEN)

`goofspiel/training/stages.py:3138-3142`, inside `run_stage7_redteam`:

```python
teacher_policy = torch.zeros(len(train_states), 13)
for b, sample in enumerate(train_samples):
    for i, v in enumerate((sample.teacher_policy or [])[:13]):
        teacher_policy[b, i] = float(v)          # <<< slot = POSITIONAL index i
teacher_policy = teacher_policy / teacher_policy.sum(...).clamp_min(1e-12)
```

`sample.teacher_policy` (`teachers.py:37,44`) is `row.tolist()` over
`a_cards = legal_cards(...)` — a vector indexed **positionally** over the legal
cards. But everything it is compared against is **value-slotted** (slot `k` ⟷
card value `k+1`):

| value-slotted (slot = card−1) | source |
|---|---|
| `robust_policy_logits` + its `masked_fill(~self_action_mask,-1e9)` | `goofspiel_model.py:413` |
| `robust_policy_fn` decode `logits[card-1]` | `model_eval.py` |
| the correction's own Q target `q[b,a-1,o-1]` | `stages.py:187-198` (`_immediate_target`) |

So **within the same loss** the Q term is correctly slotted and the policy term
is not. Positional == value-slot **iff** `self_actions == [1,2,…,m]`
(contiguous-from-1). See `audit/loss_contract.md` for the full contract.

## 2. Evidence

Reproducer: `stage7_diagnosis/reproduce/index_misalignment.py`
(→ `evidence/reproducer_summary.json`). Teacher = EXACT solver, wants **card 3**.

### 2a. Static packing (EXPERIMENTALLY CONFIRMED)
| state | positional mass → slot | = card | status | static pi_loss |
|---|---|---|---|---|
| `{1,2,3}` | 2 | 3 | correct | 1.13 |
| `{1,3}` | 1 | **2 = ILLEGAL** (logit −1e9) | round-trip FAIL | **1.0e9** |
| `{2,3}` | 1 | **2 = legal but WRONG** | round-trip FAIL | 0.68 |

### 2b. Dynamic — run the real loop, flip only the packing (EXPERIMENTALLY CONFIRMED)
20 steps, lr=1e-3, seed 0, same init/loss; measured `p(teacher_card)`:
| state | POSITIONAL (production) | BY-VALUE (proposed fix) |
|---|---|---|
| `{1,2,3}` | 3→3, p 0.34→**1.00** ✔ | identical ✔ |
| `{1,3}` | 1→1, p 0.49→**0.00**, loss 1e9 ✘ | 1→**3**, p→**1.00** ✔ |
| `{2,3}` | 3→**2**, p 0.51→**0.00**, loss 0.18 ✘ | 3→**3**, p→**1.00** ✔ |

Both gapped modes drive `p(teacher_card)→0`; since the eval metric is
`NLL = −log p(teacher_card)`, this reproduces the OBSERVED teacher-NLL blowup and
match-rate collapse. Contiguous is packing-invariant → why SMOKE (3 legacy
contiguous states) never caught it.

### 2c. End-to-end through PRODUCTION `run_stage7_redteam` (EXPERIMENTALLY CONFIRMED)
| run | train slice | match_rate | teacher-NLL | verdict |
|---|---|---|---|---|
| seed=43, train=2 | contiguous only | 0.00→1.00 | 1.107→**0.000** | clean |
| seed=7, train=6 | includes `{1,3}`,`{2,3}` | 0.333→0.667 | 0.845→**6.908** | NLL blows up |

The seed=7 NLL 0.845→6.908 mirrors OBSERVED rerun567 NLL 2.32→12.09 (same
direction, same magnitude class). Note match-rate can *rise* while NLL blows up
— **teacher-NLL is the robust regression signature**, not match-rate.

### 2d. Amplifier — optimizer/parameter flow (CODE-PROVEN)
`audit/parameter_inventory.json`: `AdamW(model.parameters())` →
**6,140,611** trainable params, **0** frozen, 1 group, covers 100% of the net.
`grep -n replay stages.py`: replay exists **only** in Stage4 (1019-1621); the
Stage7 correction region has **no** replay/anchor — full-network fine-tune on ≤6
states with no anchor is catastrophic-forgetting by construction, which is what
turns per-state policy damage into the heldout/arena-wide collapse.

## 3. Classification against the open root-cause classes

| class | status | basis |
|---|---|---|
| **D1-B target/loss bug** | **CONFIRMED (primary)** | §1 CODE-PROVEN + §2a/b/c EXPERIMENTALLY CONFIRMED |
| D1-D objective/anchor conflict | **CONFIRMED (co-occurring amplifier)** | §2d CODE-PROVEN (no replay/anchor; full-net) |
| D1-C optimizer instability | **EXCLUDED** | §2b: LR/steps/optimizer/init held constant; regression flips with packing alone |
| D1-A pipeline wiring | **EXCLUDED** | Gate D1 0-step identity is exact (before==after; corrected==init) |
| eval/report bug (Class D) | **EXCLUDED** | Gate D1 identity + every metric RE-EXECUTED, not read |

## 4. Committed tests (pin the confirmed bug — S7-D1 §44/§48)

`tests/unit/training/test_stage7_correction_diagnosis.py` — 9 passed, 1 xfailed:
- `test_contiguous_mask_round_trips` — control.
- `test_gapped_2_3_packs_onto_wrong_legal_card` **[PINS-BUG]** — silent wrong-card.
- `test_gapped_1_3_packs_onto_illegal_slot` **[PINS-BUG]** — illegal masked slot, pi_loss>1e8.
- `test_correction_contrast_positional_vs_value[gapped_1_3|gapped_2_3]` — isolation + fix demo.
- `test_correction_contrast_contiguous_is_packing_invariant` — control.
- `test_teacher_target_is_a_frozen_constant` — rules out teacher drift.
- `test_zero_step_correction_is_identity` — Gate D1 (Class D excluded).
- `test_endtoend_contiguous_only_train_corrects_cleanly` — stable production guard.
- `test_endtoend_gapped_train_blows_up_teacher_nll` — **xfail(strict)** production
  tripwire; XPASSES (→ must remove marker) once the fix lands.

Run: `PYTHONPATH=. python -m pytest tests/unit/training/test_stage7_correction_diagnosis.py -v`

## 5. Affected artifacts (do NOT overwrite — S7-D1)

- `rerun567_integration_20260904_014006` (the negative baseline) — its
  `focused_correction_report.json` numbers are **real measurements of a
  mis-indexed correction**, not fabricated. The corrected checkpoint is a
  genuinely degraded model; it must **never** be promoted over the robust head
  (fail-closed already holds: `original_attack_regression_passed=False`).
- Any run reaching this correction on a train/heldout set containing a gapped
  mask is affected. SMOKE (contiguous-only) runs are not.

## 6. Proposed minimal fix — **NOT APPLIED** (awaiting review, S7-D1 §48)

`stages.py:3139-3141`, pack by card **value**:

```python
for b, (sample, state) in enumerate(zip(train_samples, train_states)):
    row = sample.teacher_policy or []
    for i, v in enumerate(row[: len(state.self_actions)]):
        teacher_policy[b, state.self_actions[i] - 1] = float(v)   # slot = card value - 1
```

This makes the policy target agree with `robust_policy_logits`, the legality
mask, `robust_policy_fn`, and the Q target. Nothing else changes (no LR/steps/
optimizer/algorithm change). Demonstrated correct by the `by_value` branch of
`test_correction_contrast_positional_vs_value` (both gapped states → p→1.0).

**Fix-time test contract:** applying the fix must (a) flip the two **[PINS-BUG]**
tests to assert the corrected contract, and (b) remove the `xfail(strict)` marker
on `test_endtoend_gapped_train_blows_up_teacher_nll` (it will XPASS). The
amplifier (§2d) is a *separate* design question (add general replay / anchor /
encoder freeze) and is explicitly **out of scope** for this diagnosis.

## 7. Deferred

- **Stage6 antisymmetry audit (S7-D1 §42)** — BLOCKED: needs rerun567's raw
  `league_report.json`, not on local disk. Recipe when available: build the
  pairwise score matrix `M`; for every unordered pair compute
  `antisymmetry_error(i,j) = |M[i,j] + M[j,i]|` (zero-sum ⇒ should be ≈0);
  report `max` and `mean` over pairs and flag any `> 1e-6`.

## 8. What was NOT done (S7-D1 constraints honored)

No hyperparameter/LR/steps/replay/coefficient/optimizer/algorithm/attack-generator
change; no Stage5/Stage6 edit; no production Stage7 semantics change; no
overwrite/delete of rerun567 artifacts or the corrected checkpoint; no `git add
.` of the rerun567 worktree. Per §48/§49, once the bug was CONFIRMED I stopped —
no 5/40-step trajectory or reduced-arena sweeps were run (those belong to the
no-bug branch).
