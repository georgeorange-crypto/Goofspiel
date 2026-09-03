"""S7-D1 — Stage7 `focused_correction` failure DIAGNOSIS (committed evidence).

These tests pin the CONFIRMED root cause of the rerun567 correction regression:
the focused-correction POLICY target is packed **positionally** over the legal
cards (`stages.py:3138-3142`) while every other action vector in the pipeline —
`robust_policy_logits`, its legality mask, the `robust_policy_fn` decoder, and
the correction's own Q target `_immediate_target` — is **value-slotted**
(slot k ⟷ card value k+1). On any legal mask that is not contiguous-from-1
("gapped"), the teacher's probability mass lands on the wrong logit slot, so the
KL objective pushes probability AWAY from the teacher's true card.

Evidence grade (S7-D1 §43): every assertion below RE-EXECUTES the fact against
real code (`GoofspielModel`, `TeacherRouter`, `_immediate_target`,
`robust_policy_fn`, `run_stage7_redteam`) — none reads a stored field.

Test map (S7-D1 §44):
  * Gate D1 / A / C / G  — `test_zero_step_correction_is_identity`
  * Test D (teacher frozen) — `test_teacher_target_is_a_frozen_constant`
  * Test E (target legality) — `test_gapped_1_3_packs_onto_illegal_slot` [PINS-BUG]
  * Test F (index round-trip) — `test_gapped_2_3_packs_onto_wrong_legal_card`,
                                `test_contiguous_mask_round_trips` [PINS-BUG]
  * Isolation + fix demo — `test_correction_contrast_positional_vs_value`
  * Production tripwire — `test_endtoend_contiguous_only_train_corrects_cleanly` (stable),
                          `test_endtoend_gapped_train_blows_up_teacher_nll` (xfail strict)

MAINTENANCE CONTRACT: the tests tagged **[PINS-BUG]** assert the *current buggy*
packing behavior; when the value-slot fix is applied (S7-D1 sec.'Proposed fix'),
they MUST be flipped to assert the corrected contract, and the xfail(strict)
tripwire will start XPASSING — remove its marker at that time. The corrected
contract is already spelled out by the `by_value` branch of
`test_correction_contrast_positional_vs_value`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="Stage7 correction diagnosis needs torch")

import torch
import torch.nn.functional as F

from goofspiel.game import GameState
from goofspiel.models import GoofspielModel, public_state_from_game
from goofspiel.training.model_eval import robust_policy_fn
from goofspiel.training.stages import _immediate_target
from goofspiel.training.teachers import TeacherRouter

MAX_CARDS = 13
_ROUTER = TeacherRouter()

# --- deterministic gapped / contiguous fixtures (teacher = EXACT solver) -------
CONTIGUOUS_123 = GameState.initial(3, current_prize=1)
GAPPED_13 = GameState(n=3, self_mask=0b101, opp_mask=0b101, prize_mask=0b010,
                      current_prize=2, carry_pool=1, round_index=2)
GAPPED_23 = GameState(n=3, self_mask=0b110, opp_mask=0b110, prize_mask=0b001,
                      current_prize=1, carry_pool=1, round_index=2)


def _teacher_row_and_card(state: GameState):
    sample = _ROUTER.label_state(state)
    row = list(sample.teacher_policy or [1.0] * len(state.self_actions))
    best = max(range(len(state.self_actions)), key=lambda i: row[i])
    return row, state.self_actions[best]


def _pack_positional(states, rows):
    """VERBATIM replica of stages.py:3138-3142 (production, under audit)."""
    tp = torch.zeros(len(states), MAX_CARDS)
    for b, row in enumerate(rows):
        for i, v in enumerate(row[:MAX_CARDS]):
            tp[b, i] = float(v)
    return tp / tp.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _pack_by_value(states, rows):
    """Proposed minimal fix: slot = card value - 1 (matches robust_policy_logits
    and the _immediate_target Q slotting)."""
    tp = torch.zeros(len(states), MAX_CARDS)
    for b, (state, row) in enumerate(zip(states, rows)):
        for i, v in enumerate(row[: len(state.self_actions)]):
            tp[b, state.self_actions[i] - 1] = float(v)
    return tp / tp.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _mass_slot_and_card(state: GameState) -> tuple[int, int]:
    row, _ = _teacher_row_and_card(state)
    tp = _pack_positional([state], [row])[0]
    slot = int(torch.argmax(tp).item())
    return slot, slot + 1  # robust_policy_fn decodes logits[card-1] ⇒ slot k ⟷ card k+1


# =============================================================================
# Test F — indexing round-trip
# =============================================================================
def test_contiguous_mask_round_trips():
    """Contiguous-from-1 mask: positional packing == value slotting, so the
    teacher card round-trips and SMOKE never sees the bug."""
    row, tcard = _teacher_row_and_card(CONTIGUOUS_123)
    assert CONTIGUOUS_123.self_actions == [1, 2, 3]
    slot, decoded = _mass_slot_and_card(CONTIGUOUS_123)
    assert decoded == tcard, "contiguous mask must round-trip (control)"
    assert decoded in CONTIGUOUS_123.self_actions


def test_gapped_2_3_packs_onto_wrong_legal_card():  # [PINS-BUG]
    """GAPPED {2,3}: the round-trip FAILS silently — mass lands on a *legal* but
    WRONG card, so the loss looks normal while the wrong action is trained."""
    row, tcard = _teacher_row_and_card(GAPPED_23)
    assert tcard == 3  # teacher wants card 3
    slot, decoded = _mass_slot_and_card(GAPPED_23)
    assert decoded == 2, "positional packing routes card-3 mass to slot1=card2"
    assert decoded in GAPPED_23.self_actions, "card 2 IS legal here (silent mode)"
    assert decoded != tcard, "…but it is the WRONG card — round-trip is broken"


# =============================================================================
# Test E — target legality (teacher mass on an illegal, hard-masked slot)
# =============================================================================
def test_gapped_1_3_packs_onto_illegal_slot():  # [PINS-BUG]
    """GAPPED {1,3}: the teacher's card-3 mass is packed onto slot1 = card VALUE
    2, which is ILLEGAL. The model hard-masks that logit to -1e9, so the static
    KL target is unsatisfiable and pi_loss is ~1e9."""
    row, tcard = _teacher_row_and_card(GAPPED_13)
    assert tcard == 3
    slot, decoded = _mass_slot_and_card(GAPPED_13)
    assert decoded == 2
    assert decoded not in GAPPED_13.self_actions, "card 2 is ILLEGAL in {1,3}"

    model = GoofspielModel(max_cards=MAX_CARDS).eval()
    with torch.no_grad():
        logits = model(public_state_from_game([GAPPED_13], max_cards=MAX_CARDS)).robust_policy_logits[0]
    assert logits[slot].item() <= -1e8, "illegal slot must be hard-masked (~-1e9)"
    tp = _pack_positional([GAPPED_13], [row])
    pi_loss = F.kl_div(F.log_softmax(logits.unsqueeze(0), dim=-1), tp, reduction="batchmean")
    assert float(pi_loss) > 1e8, "target on a masked slot yields a ~1e9 KL"


# =============================================================================
# Isolation + fix demonstration — run the REAL loss both ways
# =============================================================================
def _run_correction(state: GameState, packer, *, steps: int = 20, lr: float = 1e-3, seed: int = 0):
    """VERBATIM replica of the correction loop stages.py:3143-3157 on one state,
    swapping only the teacher-policy packer. Returns (argmax_card_after,
    p_teacher_card_after)."""
    torch.manual_seed(seed)
    model = GoofspielModel(max_cards=MAX_CARDS)
    row, tcard = _teacher_row_and_card(state)
    batch = public_state_from_game([state], max_cards=MAX_CARDS)
    target_q, q_mask = _immediate_target([state], MAX_CARDS)
    teacher_policy = packer([state], [row])
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for _ in range(steps):
        out_m = model(batch)
        logp = F.log_softmax(out_m.robust_policy_logits, dim=-1)
        pi_loss = F.kl_div(logp, teacher_policy, reduction="batchmean")
        q_loss = F.smooth_l1_loss(out_m.q_robust[q_mask], target_q[q_mask], beta=0.1)
        loss = pi_loss + q_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    pol = robust_policy_fn(model, greedy=False, temperature=1.0)(state)
    return max(pol, key=pol.get), float(pol.get(tcard, 0.0))


@pytest.mark.parametrize("state", [GAPPED_13, GAPPED_23], ids=["gapped_1_3", "gapped_2_3"])
def test_correction_contrast_positional_vs_value(state):
    """Identical hyperparameters, identical init, identical loss — only the
    packing differs. Positional (production) FAILS to learn the teacher card on
    gapped states; by-value (proposed fix) learns it. This isolates the cause to
    the packing and EXCLUDES LR/steps/optimizer (D1-C) and eval (D1-D)."""
    _, tcard = _teacher_row_and_card(state)

    arg_pos, p_pos = _run_correction(state, _pack_positional)
    assert arg_pos != tcard, "positional packing must NOT learn the teacher card (bug)"
    assert p_pos < 0.05, "positional packing drives p(teacher_card) toward 0"

    arg_val, p_val = _run_correction(state, _pack_by_value)
    assert arg_val == tcard, "value-slot packing (fix) learns the teacher card"
    assert p_val > 0.9, "value-slot packing (fix) drives p(teacher_card) toward 1"


def test_correction_contrast_contiguous_is_packing_invariant():
    """Control: on a contiguous mask both packings are identical, so both learn
    the teacher card. Confirms the defect is specific to gapped masks."""
    _, tcard = _teacher_row_and_card(CONTIGUOUS_123)
    for packer in (_pack_positional, _pack_by_value):
        arg, p = _run_correction(CONTIGUOUS_123, packer)
        assert arg == tcard and p > 0.9


# =============================================================================
# Test D — the teacher target is a frozen constant (rules out teacher drift)
# =============================================================================
def test_teacher_target_is_a_frozen_constant():
    row, _ = _teacher_row_and_card(GAPPED_13)
    tp = _pack_positional([GAPPED_13], [row])
    assert tp.requires_grad is False, "teacher target must be a constant leaf"
    # The router/solver has no torch parameters — nothing to train into it.
    assert not hasattr(_ROUTER, "parameters"), "TeacherRouter is a solver, not a module"


# =============================================================================
# Gate D1 — 0-step identity control (rules out Class D: eval/report/pipeline bug)
# =============================================================================
def test_zero_step_correction_is_identity(tmp_path: Path):
    """With correction_steps=0 the pipeline (generate → teacher → pack → save →
    reload → eval) must be a no-op: before==after on every bucket AND the saved
    'corrected' checkpoint is byte-identical to the init checkpoint. Proves eval
    is non-mutating and any regression is caused by the correction STEPS."""
    from goofspiel.training.budgets import Stage7Budget
    from goofspiel.training.checkpoint import load_checkpoint
    from goofspiel.training.stages import run_stage7_redteam

    budget = Stage7Budget(attack_cases=3, correction_steps=0, correction_train_cases=3,
                          heldout_attack_cases=0, arena_games=0, arena_seeds=0)
    run_stage7_redteam(out_dir=tmp_path / "s7", correction_steps=0, n_cards=13, seed=1, budget=budget)
    report = json.loads((tmp_path / "s7" / "redteam" / "focused_correction_report.json").read_text(encoding="utf-8"))

    reg = report["regression"]
    assert reg["match_rate_before"] == reg["match_rate_after"], "0-step must not change match-rate"
    assert reg["mean_teacher_nll_before"] == reg["mean_teacher_nll_after"], "0-step must not change NLL"

    tp = report["training_plan"]
    init_sd = load_checkpoint(tp["init_checkpoint"])["model_state"]
    corr_sd = load_checkpoint(tp["corrected_checkpoint"])["model_state"]
    assert init_sd.keys() == corr_sd.keys()
    for k in init_sd:
        assert torch.equal(init_sd[k], corr_sd[k]), f"0-step corrected weights differ at {k}"


# =============================================================================
# Production tripwires (drive the REAL run_stage7_redteam onto real attack sets)
# =============================================================================
def test_endtoend_contiguous_only_train_corrects_cleanly(tmp_path: Path):
    """seed=43, train_cases=2 → the train slice is contiguous legacy states only.
    The real correction must reduce teacher-NLL and not regress match-rate.
    Stable (invariant to the packing fix)."""
    from goofspiel.training.budgets import Stage7Budget
    from goofspiel.training.stages import run_stage7_redteam

    budget = Stage7Budget(attack_cases=6, correction_steps=20, correction_train_cases=2,
                          heldout_attack_cases=0, arena_games=0, arena_seeds=0)
    run_stage7_redteam(out_dir=tmp_path / "s7", correction_steps=20, n_cards=13, seed=43, budget=budget)
    reg = json.loads((tmp_path / "s7" / "redteam" / "focused_correction_report.json").read_text(encoding="utf-8"))["regression"]
    assert reg["mean_teacher_nll_after"] <= reg["mean_teacher_nll_before"] + 1e-6, "contiguous correction must not worsen NLL"
    assert reg["match_rate_after"] >= reg["match_rate_before"], "contiguous correction must not regress match-rate"


@pytest.mark.xfail(strict=True, reason="S7-D1 CONFIRMED BUG: stages.py:3138-3142 packs the teacher "
                   "policy positionally; on gapped train states this blows up teacher-NLL. Remove "
                   "this marker when the value-slot fix lands (it will XPASS).")
def test_endtoend_gapped_train_blows_up_teacher_nll(tmp_path: Path):
    """seed=7, train_cases=6 → the train slice INCLUDES gapped states ({1,3},{2,3}).
    CORRECT contract: a focused correction should not worsen teacher-NLL on the
    very states it trained on. It currently does (≈0.85 → ≈6.9), so this xfails
    until the packing is fixed."""
    from goofspiel.training.budgets import Stage7Budget
    from goofspiel.training.stages import run_stage7_redteam

    budget = Stage7Budget(attack_cases=6, correction_steps=20, correction_train_cases=6,
                          heldout_attack_cases=0, arena_games=0, arena_seeds=0)
    run_stage7_redteam(out_dir=tmp_path / "s7", correction_steps=20, n_cards=13, seed=7, budget=budget)
    reg = json.loads((tmp_path / "s7" / "redteam" / "focused_correction_report.json").read_text(encoding="utf-8"))["regression"]
    assert reg["mean_teacher_nll_after"] <= reg["mean_teacher_nll_before"] + 0.5
