"""S7-F1 -- permanent slot-contract regression tests for the Stage7 teacher-policy
value-slot fix.

The Stage7 focused correction packs each stored teacher policy (a distribution
ordered POSITIONALLY over `state.self_actions`) into a global length-13 target
tensor that is scored against `robust_policy_logits`. The head, its legality mask,
`robust_policy_fn`, and the `_immediate_target` Q target are all VALUE-slotted
(slot k <-> card value k+1). The S7-F1 fix packs the teacher policy by card value:

    for card, prob in zip(state.self_actions, sample.teacher_policy or []):
        teacher_policy[b, card - 1] = float(prob)

`pack_by_value` / `unpack_by_value` below MIRROR that shipped production packing
(goofspiel/training/stages.py, run_stage7_redteam). These tests re-execute the slot
contract exhaustively; the end-to-end binding to real production lives in
test_stage7_correction_diagnosis.py (the seed=7 / seed=43 run_stage7_redteam
tripwires). The 0-step identity control (S7-F1 sec.17) also lives there.
"""
from __future__ import annotations

import math
import random
from itertools import combinations

import pytest

pytest.importorskip("torch", reason="value-slot contract tests need torch")

import torch
import torch.nn.functional as F

from goofspiel.game import GameState
from goofspiel.models import GoofspielModel, public_state_from_game
from goofspiel.training.stages import _generate_attacks
from goofspiel.training.teachers import TeacherRouter

MAX_CARDS = 13
_ROUTER = TeacherRouter()

CONTIGUOUS_123 = GameState.initial(3, current_prize=1)
GAPPED_13 = GameState(n=3, self_mask=0b101, opp_mask=0b101, prize_mask=0b010,
                      current_prize=2, carry_pool=1, round_index=2)
GAPPED_23 = GameState(n=3, self_mask=0b110, opp_mask=0b110, prize_mask=0b001,
                      current_prize=1, carry_pool=1, round_index=2)


def pack_by_value(self_actions_list, rows, *, max_cards: int = MAX_CARDS, normalize: bool = False):
    """MIRROR of the shipped stages.py packing: place each positional teacher prob
    onto the slot of its card VALUE (slot = card - 1)."""
    tp = torch.zeros(len(rows), max_cards)
    for b, (actions, row) in enumerate(zip(self_actions_list, rows)):
        for card, prob in zip(actions, row):
            tp[b, card - 1] = float(prob)
    if normalize:
        tp = tp / tp.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return tp


def unpack_by_value(tp_row, actions):
    """Read a value-slotted target back into positional order over `actions`."""
    return [float(tp_row[card - 1]) for card in actions]


def _all_nonempty_subsets(cards):
    for r in range(1, len(cards) + 1):
        for combo in combinations(cards, r):
            yield list(combo)


# =============================================================================
# S7-F1 sec.12 -- exhaustive N=5 legal subsets (all 31 non-empty)
# =============================================================================
def test_exhaustive_n5_subsets_value_slot_contract():
    """For every non-empty legal subset of {1..5}, a distinct-valued teacher row
    packs so that (a) each legal card's prob sits on slot card-1, (b) every illegal
    slot is exactly zero, (c) total mass is conserved. Distinct values make any
    mis-slotting detectable."""
    n = 5
    subsets = list(_all_nonempty_subsets(range(1, n + 1)))
    assert len(subsets) == 31, "there are exactly 2**5 - 1 non-empty subsets"
    for actions in subsets:
        row = [float(i + 1) for i in range(len(actions))]  # distinct positive
        tp = pack_by_value([actions], [row])[0]
        for i, card in enumerate(actions):
            assert tp[card - 1].item() == row[i], (actions, card)
        legal_slots = {c - 1 for c in actions}
        for slot in range(MAX_CARDS):
            if slot not in legal_slots:
                assert tp[slot].item() == 0.0, (actions, slot)
        assert abs(tp.sum().item() - sum(row)) < 1e-6, actions


# =============================================================================
# S7-F1 sec.13 -- round-trip: unpack(pack(p, A), A) == p
# =============================================================================
def test_round_trip_unpack_pack_is_identity():
    """The packing is invertible on the legal support: normalizing a distribution,
    packing it by value, then unpacking positionally returns the original."""
    rng = random.Random(0)
    for _ in range(300):
        n = rng.randint(1, MAX_CARDS)
        m = rng.randint(1, n)
        actions = sorted(rng.sample(range(1, n + 1), m))
        raw = [rng.random() + 1e-3 for _ in range(m)]
        s = sum(raw)
        p = [x / s for x in raw]
        tp = pack_by_value([actions], [p], normalize=True)[0]
        back = unpack_by_value(tp, actions)
        assert len(back) == len(p)
        for a, b in zip(p, back):
            assert math.isclose(a, b, rel_tol=0, abs_tol=1e-6), (actions, p, back)


# =============================================================================
# S7-F1 sec.14 -- named reproducers {1,2,3}, {1,3}, {2,3}
# =============================================================================
@pytest.mark.parametrize("actions,teacher_card", [([1, 2, 3], 3), ([1, 3], 3), ([2, 3], 3)])
def test_named_masks_teacher_card_lands_on_its_value_slot(actions, teacher_card):
    """A one-hot teacher on `teacher_card` (expressed positionally over `actions`)
    must decode to slot teacher_card-1 with zero mass on any illegal slot -- for the
    contiguous case and both historically-failing gapped cases."""
    row = [1.0 if c == teacher_card else 0.0 for c in actions]
    tp = pack_by_value([actions], [row])[0]
    assert int(torch.argmax(tp).item()) == teacher_card - 1
    legal = {c - 1 for c in actions}
    assert all(tp[s].item() == 0.0 for s in range(MAX_CARDS) if s not in legal)


# =============================================================================
# S7-F1 sec.15 -- N=13 gapped masks keep the slot contract
# =============================================================================
@pytest.mark.parametrize("actions", [[1, 7, 13], [2, 5, 9, 12], [4, 13], [3, 6, 9, 12, 13]])
def test_n13_gapped_masks_slot_contract(actions):
    row = [float(i + 1) for i in range(len(actions))]
    tp = pack_by_value([actions], [row])[0]
    for i, card in enumerate(actions):
        assert tp[card - 1].item() == row[i], (actions, card)
    legal = {c - 1 for c in actions}
    assert all(tp[s].item() == 0.0 for s in range(MAX_CARDS) if s not in legal)


# =============================================================================
# S7-F1 sec.16 -- target legality gate, bound to the REAL teacher and REAL model
# =============================================================================
@pytest.mark.parametrize("state", [CONTIGUOUS_123, GAPPED_13, GAPPED_23],
                         ids=["contiguous_123", "gapped_1_3", "gapped_2_3"])
def test_target_mass_only_on_legal_slots_real_teacher(state):
    """Using the REAL TeacherRouter output and the shipped packing, every slot that
    carries target mass must be a LEGAL (non -1e9-masked) slot of the REAL model's
    robust_policy_logits; illegal slots carry exactly zero."""
    sample = _ROUTER.label_state(state)
    row = list(sample.teacher_policy or [])
    tp = pack_by_value([state.self_actions], [row], normalize=True)[0]

    legal_slots = {c - 1 for c in state.self_actions}
    for slot in range(MAX_CARDS):
        if slot not in legal_slots:
            assert tp[slot].item() == 0.0, (state.self_actions, slot)

    model = GoofspielModel(max_cards=MAX_CARDS).eval()
    with torch.no_grad():
        logits = model(public_state_from_game([state], max_cards=MAX_CARDS)).robust_policy_logits[0]
    for slot in range(MAX_CARDS):
        if tp[slot].item() > 0.0:
            assert logits[slot].item() > -1e8, ("mass on a hard-masked slot", state.self_actions, slot)


# =============================================================================
# S7-F1 sec.9 -- the length contract that justifies packing by value
# =============================================================================
def test_teacher_policy_length_equals_self_actions_across_attacks():
    """RE-EXECUTE the CODE-PROVEN contract the fix relies on: for every state the
    Stage7 attack generator produces, len(teacher_policy) == len(self_actions), so
    zip(self_actions, teacher_policy) pairs every legal card with its own prob (no
    truncation). Exercises both teacher paths and confirms gapped masks occur."""
    cases = _generate_attacks(n_cards=13, seed=7, count=12)
    assert cases, "attack generator must yield states"
    seen_gapped = False
    for case in cases:
        st = case.state
        sample = _ROUTER.label_state(st)
        assert sample.teacher_policy is not None, st.self_actions
        assert len(sample.teacher_policy) == len(st.self_actions), (
            st.self_actions, len(sample.teacher_policy))
        if st.self_actions != list(range(1, len(st.self_actions) + 1)):
            seen_gapped = True
    assert seen_gapped, "seed=7 attack set must contain at least one gapped mask"
