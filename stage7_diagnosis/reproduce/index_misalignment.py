"""S7-D1 reproducer — focused-correction teacher-policy INDEX MISALIGNMENT.

Run:  python stage7_diagnosis/reproduce/index_misalignment.py

This is a self-contained, deterministic reproducer for the confirmed Stage7
`focused_correction` defect.  It has two parts:

  PART 1 (static)  — replicates the EXACT teacher-policy packing from
                     stages.py:3138-3142 and the EXACT decode from
                     model_eval.robust_policy_fn, and shows that on GAPPED legal
                     masks the teacher's probability mass lands on the wrong
                     logit slot (either an illegal, hard-masked slot, or the
                     wrong legal card).

  PART 2 (dynamic) — runs the EXACT correction loop from stages.py:3143-3157
                     TWICE on the same states, at identical hyperparameters,
                     changing ONLY how the teacher policy is packed:
                       (a) POSITIONAL  = production (stages.py, buggy)
                       (b) BY-VALUE    = proposed minimal fix
                     and MEASURES the teacher-card probability the corrected
                     model assigns, before vs after.  This isolates mask
                     geometry as the sole cause and demonstrates the fix.

Nothing here mutates any artifact or production code; it only imports the real
loss pieces (`_immediate_target`, model forward, `F.kl_div`) and the real
decoder (`robust_policy_fn`).  The correction loop is copied verbatim (and
cited) because it is inline inside `run_stage7_redteam`, not a callable unit.
"""
from __future__ import annotations

import copy
import json
import warnings

import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

from goofspiel.game import GameState
from goofspiel.models import GoofspielModel, public_state_from_game
from goofspiel.training.model_eval import robust_policy_fn
from goofspiel.training.stages import _immediate_target
from goofspiel.training.teachers import TeacherRouter

MAX_CARDS = 13
LR = 1e-3
STEPS = 20
SEED = 0

router = TeacherRouter()


def teacher_row_and_card(state: GameState):
    """The teacher row (positional over state.self_actions) and the argmax CARD."""
    sample = router.label_state(state)
    row = list(sample.teacher_policy or [1.0] * len(state.self_actions))
    best_index = max(range(len(state.self_actions)), key=lambda i: row[i])
    return row, state.self_actions[best_index], sample.teacher_source


def pack_positional(states, rows):
    """EXACT replica of stages.py:3138-3142 (production, buggy)."""
    tp = torch.zeros(len(states), MAX_CARDS)
    for b, row in enumerate(rows):
        for i, v in enumerate(row[:MAX_CARDS]):
            tp[b, i] = float(v)  # <<< slot = positional index i
    return tp / tp.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def pack_by_value(states, rows):
    """Proposed minimal fix: slot = card VALUE - 1 (aligns with the value-slotted
    robust_policy_logits and the _immediate_target Q slotting)."""
    tp = torch.zeros(len(states), MAX_CARDS)
    for b, (state, row) in enumerate(zip(states, rows)):
        for i, v in enumerate(row[: len(state.self_actions)]):
            card = state.self_actions[i]
            tp[b, card - 1] = float(v)  # <<< slot = card value - 1
    return tp / tp.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def part1_static():
    print("=" * 78)
    print("PART 1 (static) — packing vs decode round-trip")
    print("=" * 78)
    cases = {
        "CONTIGUOUS {1,2,3}": GameState.initial(3, current_prize=1),
        "GAPPED {1,3}": GameState(n=3, self_mask=0b101, opp_mask=0b101, prize_mask=0b010,
                                  current_prize=2, carry_pool=1, round_index=2),
        "GAPPED {2,3}": GameState(n=3, self_mask=0b110, opp_mask=0b110, prize_mask=0b001,
                                  current_prize=1, carry_pool=1, round_index=2),
    }
    model = GoofspielModel(max_cards=MAX_CARDS).eval()
    out = {}
    for tag, state in cases.items():
        row, tcard, source = teacher_row_and_card(state)
        tp = pack_positional([state], [row])[0]
        mass_slot = int(torch.argmax(tp).item())
        decoded_card = mass_slot + 1  # robust_policy_fn decodes logits[card-1]
        legal = list(state.self_actions)
        with torch.no_grad():
            logits = model(public_state_from_game([state], max_cards=MAX_CARDS)).robust_policy_logits[0]
        kl = float(F.kl_div(F.log_softmax(logits.unsqueeze(0), dim=-1), tp.unsqueeze(0), reduction="batchmean"))
        rec = {
            "self_actions": legal, "teacher_source": source, "teacher_card": tcard,
            "positional_mass_slot": mass_slot, "decoded_card_at_slot": decoded_card,
            "roundtrip_ok": decoded_card == tcard, "target_on_legal_slot": decoded_card in legal,
            "logit_at_mass_slot": round(float(logits[mass_slot]), 4), "pi_loss": kl,
        }
        out[tag] = rec
        print(f"\n{tag}: self_actions={legal} teacher_card={tcard} ({source})")
        print(f"  positional mass -> slot {mass_slot} = card {decoded_card}  "
              f"roundtrip={'OK' if rec['roundtrip_ok'] else 'FAIL'}  "
              f"legal_slot={'OK' if rec['target_on_legal_slot'] else 'ILLEGAL'}")
        print(f"  model logit at that slot = {rec['logit_at_mass_slot']:.4g}   pi_loss = {kl:.4e}")
    return out


def run_correction(state, row, packer, steps=STEPS, lr=LR):
    """VERBATIM replica of the correction loop stages.py:3143-3157 for a single
    state, swapping only the teacher-policy packer.  Returns teacher-card prob
    before/after and the final loss."""
    torch.manual_seed(SEED)
    model = GoofspielModel(max_cards=MAX_CARDS)
    _, tcard, _ = teacher_row_and_card(state)

    def teacher_card_prob(m):
        pol = robust_policy_fn(m, greedy=False, temperature=1.0)(state)  # {card: prob} over legal
        return float(pol.get(tcard, 0.0)), max(pol, key=pol.get)

    model.eval()
    p_before, arg_before = teacher_card_prob(model)

    batch = public_state_from_game([state], max_cards=MAX_CARDS)
    target_q, q_mask = _immediate_target([state], MAX_CARDS)
    teacher_policy = packer([state], [row])
    opt = torch.optim.AdamW(model.parameters(), lr=lr)      # stages.py:3143 (ENTIRE net)
    model.train()
    last = 0.0
    for _ in range(steps):                                   # stages.py:3146-3156
        out_m = model(batch)
        logp = F.log_softmax(out_m.robust_policy_logits, dim=-1)
        pi_loss = F.kl_div(logp, teacher_policy, reduction="batchmean")
        q_loss = F.smooth_l1_loss(out_m.q_robust[q_mask], target_q[q_mask], beta=0.1)
        loss = pi_loss + q_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last = float(loss.detach().cpu())
    model.eval()
    p_after, arg_after = teacher_card_prob(model)
    return {"teacher_card": tcard, "p_before": round(p_before, 4), "arg_before": arg_before,
            "p_after": round(p_after, 4), "arg_after": arg_after, "final_loss": last,
            "learned": arg_after == tcard, "prob_gain": round(p_after - p_before, 4)}


def part2_dynamic():
    print("\n" + "=" * 78)
    print(f"PART 2 (dynamic) — run the REAL correction loop, {STEPS} steps, lr={LR}")
    print("   positional (production/buggy)  vs  by-value (proposed fix)")
    print("=" * 78)
    cases = {
        "CONTIGUOUS {1,2,3}": GameState.initial(3, current_prize=1),
        "GAPPED {1,3}": GameState(n=3, self_mask=0b101, opp_mask=0b101, prize_mask=0b010,
                                  current_prize=2, carry_pool=1, round_index=2),
        "GAPPED {2,3}": GameState(n=3, self_mask=0b110, opp_mask=0b110, prize_mask=0b001,
                                  current_prize=1, carry_pool=1, round_index=2),
    }
    out = {}
    for tag, state in cases.items():
        row, _, _ = teacher_row_and_card(state)
        pos = run_correction(state, row, pack_positional)
        val = run_correction(state, row, pack_by_value)
        out[tag] = {"positional": pos, "by_value": val}
        print(f"\n{tag}: teacher wants card {pos['teacher_card']}")
        print(f"  POSITIONAL (prod): argmax {pos['arg_before']}->{pos['arg_after']}  "
              f"p(teacher_card) {pos['p_before']}->{pos['p_after']}  "
              f"final_loss={pos['final_loss']:.3e}  learned={pos['learned']}")
        print(f"  BY-VALUE  (fix) : argmax {val['arg_before']}->{val['arg_after']}  "
              f"p(teacher_card) {val['p_before']}->{val['p_after']}  "
              f"final_loss={val['final_loss']:.3e}  learned={val['learned']}")
    return out


if __name__ == "__main__":
    from pathlib import Path

    static = part1_static()
    dynamic = part2_dynamic()
    summary = {"config": {"lr": LR, "steps": STEPS, "seed": SEED, "max_cards": MAX_CARDS},
               "static": static, "dynamic": dynamic}
    print("\n" + "=" * 78)
    print("MACHINE-READABLE SUMMARY")
    print("=" * 78)
    print(json.dumps(summary, indent=2))
    ev = Path(__file__).resolve().parent.parent / "evidence" / "reproducer_summary.json"
    ev.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[written] {ev}")
