# Stage7 `focused_correction` — Loss & Target Contract Audit

Scope: the correction optimization inside `run_stage7_redteam`
(`goofspiel/training/stages.py`), HEAD `7366c85`, file sha256
`659ce37e…`. Line numbers are exact for that revision.

Evidence grade legend (S7-D1 §43): **CODE-PROVEN** = read directly from source;
**EXPERIMENTALLY CONFIRMED** = reproduced by running real code; **OBSERVED** =
from the rerun567 artifact; **INFERRED** = reasoned, not yet executed.

---

## 1. The correction loop (verbatim, stages.py:3136-3157)

```python
3136  batch = public_state_from_game(train_states, max_cards=13)
3137  target_q, q_mask = _immediate_target(train_states, 13)
3138  teacher_policy = torch.zeros(len(train_states), 13)
3139  for b, sample in enumerate(train_samples):
3140      for i, v in enumerate((sample.teacher_policy or [])[:13]):
3141          teacher_policy[b, i] = float(v)                 # (A) PACKING
3142  teacher_policy = teacher_policy / teacher_policy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
3143  opt = torch.optim.AdamW(model.parameters(), lr=lr)       # (B) OPTIMIZER
3144  model.train()
3145  last_loss = 0.0
3146  for _step in range(int(correction_steps)):
3147      out_m = model(batch)
3148      logp = F.log_softmax(out_m.robust_policy_logits, dim=-1)
3149      pi_loss = F.kl_div(logp, teacher_policy, reduction="batchmean")   # (C) POLICY LOSS
3150      q_loss = F.smooth_l1_loss(out_m.q_robust[q_mask], target_q[q_mask], beta=0.1)  # (D) Q LOSS
3151      loss = pi_loss + q_loss                              # (E) TOTAL — 2 terms only
3152      opt.zero_grad(set_to_none=True)
3153      loss.backward()
3154      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
3155      opt.step()
3156      last_loss = float(loss.detach().cpu())
3157  model.eval()
```

## 2. Target semantics — the two slotting conventions (CODE-PROVEN)

There are **two coordinate systems** for a 13-wide action vector in this codebase:

| Convention | Definition | Used by |
|---|---|---|
| **value-slot** | slot `k` ⟷ card **value** `k+1` | `robust_policy_logits` mask (`goofspiel_model.py:413`, `masked_fill(~self_action_mask,-1e9)`), `public_state_from_game` (`self_cards[row,card-1]=1`), `robust_policy_fn` decode (`model_eval.py`, `logits[card-1]`), **and the Q target** `_immediate_target` (`stages.py:187-198`, `q[b,a-1,o-1]`) |
| **positional** | slot `i` ⟷ the `i`-th legal card = `state.self_actions[i]` | `sample.teacher_policy` = `row.tolist()` over `a_cards = legal_cards(...)` (`teachers.py:37,44`) |

- **(C) policy target** at line **3141** packs the positional row directly into
  slot `i` — i.e. it treats a **positional** vector as if it were **value-slotted**.
- **(D) Q target** at line 3137/`_immediate_target` uses **value-slot** (`a-1`),
  which correctly matches `q_robust`.

So within the *same* loss the Q term is correctly slotted and the policy term is
not. The policy target is only correct when `self_actions == [1,2,…,m]`
(contiguous-from-1), because only then does positional index `i` equal
`self_actions[i]-1`. **CODE-PROVEN.**

### Consequence on non-contiguous (gapped) masks (EXPERIMENTALLY CONFIRMED)

`stage7_diagnosis/reproduce/index_misalignment.py`, teacher = EXACT solver,
teacher wants **card 3** in every case:

| state | mass lands on slot | = card | status | static pi_loss |
|---|---|---|---|---|
| `{1,2,3}` contiguous | 2 | 3 | correct | 1.13 |
| `{1,3}` gapped | 1 | 2 | **ILLEGAL** (2∉{1,3}); logit is `-1e9` | **1.0e9** |
| `{2,3}` gapped | 1 | 2 | legal but **WRONG card** | 0.68 |

## 3. Why this regresses everything — the actual mechanism (EXPERIMENTALLY CONFIRMED)

The gradient of `pi_loss = KL(teacher‖student)` w.r.t. logits is
`softmax(logits) − teacher` (bounded), and the illegal slot is re-masked to
`-1e9` every forward pass, so its gradient is blocked. **Therefore the damage is
NOT gradient explosion** (an earlier inference, now refined). The damage is that
the target sits on the wrong slot, so the KL objective pushes probability
**away** from the teacher's true card:

Running the real loop (20 steps, lr=1e-3, seed 0) and measuring `p(teacher_card)`:

| state | POSITIONAL (production) | BY-VALUE (proposed fix) |
|---|---|---|
| `{1,2,3}` | argmax 3→3, p 0.34→**1.00**, learned ✔ | 3→3, p→1.00 ✔ (**identical**) |
| `{1,3}` | argmax 1→1, p 0.49→**0.00**, loss 1e9, learned ✘ | 1→**3**, p→**1.00** ✔ |
| `{2,3}` | argmax 3→**2**, p 0.51→**0.00**, loss 0.18, learned ✘ | 3→**3**, p→**1.00** ✔ |

- Both gapped modes drive `p(teacher_card) → 0`. Since the eval metric computes
  `NLL = −log p(teacher_card)` (`_attack_state_regression`), `p→0` ⇒ NLL→∞ —
  this reproduces the **OBSERVED** teacher-NLL 2.32→12.09 and the match-rate
  collapse, and (via the shared trunk, §4) the arena/heldout collapse.
- The `{2,3}` "silent" mode has an ordinary-looking loss (0.18) yet trains the
  **wrong legal card** — invisible to any loss-threshold guard.
- Contiguous masks are unaffected (both packings identical) — this is exactly
  why the SMOKE suite (3 legacy states, all contiguous-from-1) never caught it.
- LR, steps, optimizer, model init, and both loss terms were held **constant**
  across the two columns; only the packing changed ⇒ optimizer instability
  (D1-C) and eval/report bug (D1-D) are **excluded** as the cause.

## 4. Optimizer / parameter-flow contract (CODE-PROVEN)

`opt = torch.optim.AdamW(model.parameters(), lr=lr)` (line 3143) — see
`audit/parameter_inventory.json`:

- **6,140,611** trainable parameters, **0** frozen, **1** param group, optimizer
  covers **100%** of the network (`optimizer_covers_all_params: true`).
- Every top module (card_transformer 1.78M, matrix_blocks, adaptive_cnn,
  intra_game_lstm, inter_game_mamba, all heads) is updated on **≤ a handful** of
  attack states.
- **No general replay, no anchor/EWC, no encoder freeze.** `grep -n replay
  stages.py` shows replay machinery exists **only** in Stage4 (lines 1019-1621);
  the Stage7 correction region references `replay` at exactly two points: line
  3258 (a report string) and line 3317 (a comment).

This is a catastrophic-forgetting configuration by construction: full-network
fine-tuning on tiny data with no anchor corrupts the shared trunk that every
bucket (heldout, arena, teacher-NLL) reads. It is the **amplifier** that turns
per-state policy damage (§3) into across-the-board regression.

## 5. Report vs code discrepancy (CODE-PROVEN)

The focused-correction report block (stages.py:3250-3259) declares:

```python
"freeze_public_encoder": False,          # line 3257
"retain_general_replay_fraction": 0.25,  # line 3258
```

- `retain_general_replay_fraction: 0.25` is **dead metadata** — no code path
  consumes it; there is no replay in the correction loop (§4).
- `freeze_public_encoder: False` is literally true but misleading: nothing is
  frozen anywhere; it reads as a deliberate choice when it is just the default of
  an unfrozen 6.1M-param network.

## 6. Teacher target is frozen (CODE-PROVEN)

`teacher_policy` is built from `torch.zeros` + Python floats (lines 3138-3142),
so it is a constant leaf with `requires_grad=False`; the only `.detach()` (line
3156) is for loss logging. The teacher itself is the EXACT/REFERENCE_NASH matrix
solver (`teachers.py`), which has no torch parameters. There is no "teacher
drifts during correction" failure mode. This rules out that sub-hypothesis.
