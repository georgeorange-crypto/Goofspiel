#!/usr/bin/env python
"""Experiment D -- diagnostic probe of Stage-4's NeuRD robust actor.

Goal (narrow, by explicit instruction): observe the *unmodified* Stage-4 update
dynamics and decide whether the ugly actor/entropy curves come from NeuRD's
raw-logit replicator dynamics (logits diverging to +/- infinity, as the NeuRD paper's own
Remark warns) rather than from anything DDP-related.

Variants:
  * before: pre-repair Stage4 actor update with raw ``regret * logits``,
    action-dependent ``baseline = chosen_q.detach()`` PG, and entropy bonus.
  * after: repaired Stage4 update with NeuRD raw-logit thresholding and no PG
    baseline/entropy bonus.
  * The average-policy accumulator is a SIDE-CHANNEL observer.  It never feeds
    back into the loss or the optimizer; it only accumulates a running mean of
    the instantaneous softmax policy so we can evaluate the *time-averaged*
    trajectory (the object NeuRD's convergence theory actually talks about),
    entirely off to the side.

What we record each logged step (default every 5):
    step, curriculum_n,
    max|logit|, max logit-gap (legal actions only),
    entropy, actor_loss, pg_loss, q_loss,
    regret mean/std/max, grad-norm (pre-clip), any NaN/Inf flag.

At the end we OPTIONALLY compute full-game exploitability for
    (a) the final instantaneous greedy policy, and
    (b) the time-averaged policy accumulated over the run,
reusing goofspiel.training.model_eval.full_game_exploitability UNCHANGED; the
average policy is wrapped as a PolicyFn exactly the way carry_nash_policy_fn
wraps its solved policy_map.  If coverage is too thin for that to be meaningful
we say so honestly rather than dressing a proxy up as a convergence proof.

Usage (single-card, CPU, ~120 steps, starting from a Stage-3 checkpoint):
    PYTHONPATH=$HOME/pylibs:. python scripts/diag_stage4_neurd.py \
        --init-from-checkpoint <stage3_sft.pt> \
        --steps 120 --batch-size 128 --n-cards 5 --device cpu \
        --out artifacts/diag/stage4_neurd_$(date +%Y%m%d_%H%M%S)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the EXACT production building blocks Stage-4 uses.  Importing them (not
# reimplementing) guarantees the diagnostic loop trains on the same data, the
# same regret, the same solve, as the real runner.
from goofspiel.game import GameState
from goofspiel.learning.game_theory.neurd import (
    NEURD_LOGIT_THRESHOLD,
    legal_logits,
    neurd_actor_loss_from_regret,
    row_action_regret,
)
from goofspiel.learning.game_theory.regret_matching_plus import solve_batch
from goofspiel.models import GoofspielModel, public_state_from_game
from goofspiel.training.curriculum import ProgressiveCurriculum
from goofspiel.training.replay import TrajectoryReplayBuffer
from goofspiel.training.stages import (
    _apply_init_or_resume,
    _flatten_trajectory_batch,
    _rollout_selfplay_game,
    _trajectory_sample_id,
)


def _finite(x: float) -> bool:
    return not (math.isnan(x) or math.isinf(x))


class AveragePolicyAccumulator:
    """Side-channel running mean of the instantaneous softmax policy.

    Keyed by the same canonical state tuple ``carry_nash_policy_fn`` uses, so the
    resulting map can be wrapped as a PolicyFn and fed to the exact-BR machine.
    NEVER touches the optimizer; pure observation of the time-averaged policy.
    """

    def __init__(self) -> None:
        # key -> (legal_cards tuple, running-mean prob vector aligned to legal, count)
        self._acc: dict[tuple, tuple[tuple[int, ...], list[float], int]] = {}

    @staticmethod
    def _key(state: GameState) -> tuple:
        from goofspiel.game.state import card_bit

        r_mask = state.prize_mask | (card_bit(state.current_prize) if state.current_prize else 0)
        return (state.self_mask, state.opp_mask, r_mask, state.carry_pool, state.current_prize)

    def observe(self, state: GameState, legal: list[int], probs: list[float]) -> None:
        key = self._key(state)
        entry = self._acc.get(key)
        if entry is None:
            self._acc[key] = (tuple(legal), list(probs), 1)
            return
        cards, mean, n = entry
        if cards != tuple(legal):
            # Same canonical key must expose the same legal set; if not, skip
            # (defensive; should not happen for a fixed n game).
            return
        n1 = n + 1
        for i, p in enumerate(probs):
            mean[i] += (p - mean[i]) / n1
        self._acc[key] = (cards, mean, n1)

    def as_policy_fn(self):
        """Wrap the accumulated mean policy as a model_eval.PolicyFn.

        Off-support states (never visited during the short run) fall back to
        uniform over legal actions; EXACTLY the honest fallback
        carry_nash_policy_fn uses.  The BR enumerator walks all reachable
        states, so this fallback is why a short run's exploitability is a coarse
        lower-information proxy, not a convergence certificate.
        """
        acc = self._acc

        def policy(state: GameState) -> Mapping[int, float]:
            legal = state.self_actions
            if not legal:
                return {}
            entry = acc.get(AveragePolicyAccumulator._key(state))
            if entry is None:
                p = 1.0 / len(legal)
                return {c: p for c in legal}
            cards, mean, _n = entry
            total = float(sum(max(0.0, v) for v in mean)) or 1.0
            return {c: max(0.0, mean[i]) / total for i, c in enumerate(cards)}

        return policy

    def coverage(self) -> int:
        return len(self._acc)


def run_diag(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    device = args.device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Build model + optimizer EXACTLY as Stage-4 does (stages.py:769-785) ----
    model = GoofspielModel(max_cards=13).to(device)
    target_model = GoofspielModel(max_cards=13).to(device)
    opt_q = torch.optim.AdamW(model.parameters(), lr=args.lr)
    lineage = _apply_init_or_resume(
        model,
        init_from_checkpoint_path=args.init_from_checkpoint,
        resume_checkpoint_path=None,
        optimizers={"robust_rl": opt_q},
        target_model=target_model,
    )
    target_model.load_state_dict(getattr(model, "module", model).state_dict())
    target_model.eval()
    # NOTE: single process, NO DistributedDataParallel wrap.  This isolates the
    # NeuRD update dynamics from anything DDP-related, which is the whole point.

    replay_buffer = TrajectoryReplayBuffer(out / "replay" / "diag_selfplay.jsonl")
    warmup_n = args.n_cards if int(args.steps) <= 1 else min(3, args.n_cards)
    curriculum = ProgressiveCurriculum(
        target_n=args.n_cards,
        warmup_n=warmup_n,
        ramp_every=max(1, int(args.steps) // max(1, args.n_cards - warmup_n + 1)),
    )
    rng = random.Random(17 + args.n_cards + args.batch_size)
    avg_policy = AveragePolicyAccumulator()

    rows: list[dict[str, Any]] = []
    saw_nonfinite = False

    for step in range(int(args.steps)):
        cstep = curriculum.at(step)
        model.eval()
        trajectories = [
            _rollout_selfplay_game(
                model,
                n_cards=cstep.n_cards,
                rng=rng,
                device=device,
                model_version=f"diag_step_{step}",
                game_index=step * args.batch_size + i,
                sample_id=_trajectory_sample_id(
                    stage="diag_stage4",
                    rank=0,
                    step=step,
                    game_index=step * args.batch_size + i,
                    seed=args.seed,
                    n_cards=cstep.n_cards,
                ),
            )
            for i in range(int(args.batch_size))
        ]
        replay_buffer.append_many(trajectories)
        sampled = replay_buffer.sample(args.batch_size, rng)
        states, action_self, action_opp, mc_returns = _flatten_trajectory_batch(sampled or trajectories)
        batch = public_state_from_game(states, max_cards=13, device=device)
        idx = torch.arange(len(states), device=device)
        action_self_t = torch.tensor(action_self, dtype=torch.long, device=device)
        action_opp_t = torch.tensor(action_opp, dtype=torch.long, device=device)
        returns_t = torch.tensor(mc_returns, dtype=torch.float32, device=device)

        model.train()
        out_model = model(batch)
        with torch.no_grad():
            target_out = target_model(batch)
        sol = solve_batch(
            target_out.q_robust.detach(), batch.self_action_mask, batch.opponent_action_mask, iterations=64
        )
        chosen_q = out_model.q_robust[idx, action_self_t, action_opp_t]
        bootstrapped_returns = 0.8 * returns_t + 0.2 * target_out.q_robust[idx, action_self_t, action_opp_t].detach()
        q_loss = F.smooth_l1_loss(chosen_q, bootstrapped_returns, beta=0.1)
        centered_logits = legal_logits(out_model.robust_policy_logits, batch.self_action_mask)
        if args.variant == "before":
            logp = F.log_softmax(out_model.robust_policy_logits, dim=-1)[idx, action_self_t]
            baseline = chosen_q.detach()
            pg_loss = -(logp * (returns_t - baseline)).mean()
            policy = F.softmax(out_model.robust_policy_logits, dim=-1)
            log_policy = F.log_softmax(out_model.robust_policy_logits, dim=-1)
        else:
            pg_loss = torch.zeros((), device=device)
            policy = F.softmax(centered_logits, dim=-1)
            log_policy = F.log_softmax(centered_logits, dim=-1)
        entropy = -(policy * log_policy).sum(dim=-1).mean()
        action_regret = row_action_regret(
            out_model.q_robust.detach(),
            policy.detach(),
            sol.column_policy.detach(),
            batch.self_action_mask,
        )
        self_mask_f = batch.self_action_mask.float()
        denom = self_mask_f.sum(dim=-1).clamp_min(1.0)
        if args.variant == "before":
            thresholded_force = action_regret.detach()
            actor_loss = -(
                (thresholded_force * out_model.robust_policy_logits * self_mask_f).sum(dim=-1) / denom
            ).mean()
        else:
            actor_loss, _centered_actor_logits, thresholded_force = neurd_actor_loss_from_regret(
                out_model.robust_policy_logits,
                action_regret,
                batch.self_action_mask,
                threshold=NEURD_LOGIT_THRESHOLD,
                threshold_step_size=args.lr,
            )
        anchor = F.kl_div(
            log_policy, sol.row_policy.detach(), reduction="batchmean"
        )
        loss = q_loss + actor_loss + 0.1 * anchor
        if args.variant == "before":
            loss = loss + pg_loss - 0.01 * entropy
        opt_q.zero_grad(set_to_none=True)
        loss.backward()

        # ---- INSTRUMENTATION (read-only; happens between backward and clip) ----
        grad_norm = float(
            torch.sqrt(
                sum(
                    (p.grad.detach().float().pow(2).sum() for p in model.parameters() if p.grad is not None),
                    torch.tensor(0.0),
                )
            )
        )
        # Same clip as production; kept so the update itself is
        # identical; we simply recorded grad_norm PRE-clip above.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_q.step()
        with torch.no_grad():
            for tp, sp in zip(target_model.parameters(), model.parameters()):
                tp.mul_(0.995).add_(sp.detach(), alpha=0.005)

        # ---- Side-channel: accumulate the time-averaged policy (no grad) ----
        with torch.no_grad():
            observed_logits = out_model.robust_policy_logits.detach()
            for b, st in enumerate(states):
                legal = st.self_actions
                if not legal:
                    continue
                row = observed_logits[b]
                sub = torch.tensor([float(row[c - 1]) for c in legal])
                sm = F.softmax(sub, dim=-1).tolist()
                avg_policy.observe(st, legal, sm)

        # ---- Metrics we care about ----
        with torch.no_grad():
            logits = out_model.robust_policy_logits.detach()
            mask_b = batch.self_action_mask.bool()
            # max |logit| over legal entries
            legal_vals = logits.masked_fill(~mask_b, 0.0)
            max_abs_logit = float(legal_vals.abs().max())
            # max legit-gap per row = max_legal - min_legal, then batch-max
            very_neg = torch.finfo(logits.dtype).min
            hi = logits.masked_fill(~mask_b, very_neg).max(dim=-1).values
            lo = logits.masked_fill(~mask_b, -very_neg).min(dim=-1).values
            max_gap = float((hi - lo).max())
            reg_legal = thresholded_force.detach()[mask_b]
            reg_mean = float(reg_legal.mean()) if reg_legal.numel() else 0.0
            reg_std = float(reg_legal.std()) if reg_legal.numel() > 1 else 0.0
            reg_max = float(reg_legal.abs().max()) if reg_legal.numel() else 0.0

        ent_v = float(entropy.detach())
        actor_v = float(actor_loss.detach())
        pg_v = float(pg_loss.detach())
        q_v = float(q_loss.detach())
        vals = [max_abs_logit, max_gap, ent_v, actor_v, pg_v, q_v,
                reg_mean, reg_std, reg_max, grad_norm]
        row_nonfinite = any(not _finite(v) for v in vals)
        saw_nonfinite = saw_nonfinite or row_nonfinite

        if step % args.log_every == 0 or step == int(args.steps) - 1:
            row = {
                "step": step,
                "curriculum_n": cstep.n_cards,
                "max_abs_logit": max_abs_logit,
                "max_logit_gap": max_gap,
                "entropy": ent_v,
                "actor_loss": actor_v,
                "pg_loss": pg_v,
                "q_loss": q_v,
                "regret_mean": reg_mean,
                "regret_std": reg_std,
                "regret_absmax": reg_max,
                "grad_norm_preclip": grad_norm,
                "nonfinite": row_nonfinite,
            }
            rows.append(row)
            print(
                f"step {step:4d}/{args.steps}  n={cstep.n_cards}  "
                f"|logit|max={max_abs_logit:9.2f}  gap={max_gap:9.2f}  "
                f"H={ent_v:.4f}  actor={actor_v:12.2f}  "
                f"q={q_v:.4f}  reg(mean/std/max)={reg_mean:.3f}/{reg_std:.3f}/{reg_max:.3f}  "
                f"|g|={grad_norm:8.2f}{'  <<NaN/Inf' if row_nonfinite else ''}",
                flush=True,
            )

    (out / "curve.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )

    # ---- Optional exploitability: instantaneous greedy vs time-averaged ----
    exploit: dict[str, Any] = {"attempted": False}
    if args.exploit and args.n_cards <= 6:
        from goofspiel.training.model_eval import (
            full_game_exploitability,
            robust_policy_fn,
            uniform_policy_fn,
        )

        exploit = {"attempted": True, "n_cards": int(args.n_cards), "avg_policy_coverage": avg_policy.coverage()}
        try:
            model.eval()
            inst = full_game_exploitability(robust_policy_fn(model, device=device, greedy=True), n_cards=args.n_cards)
            avg = full_game_exploitability(avg_policy.as_policy_fn(), n_cards=args.n_cards)
            unif = full_game_exploitability(uniform_policy_fn(), n_cards=args.n_cards)
            exploit.update(
                {
                    "instantaneous_greedy": inst,
                    "time_averaged": avg,
                    "uniform_reference": unif,
                    "note": (
                        "Short-run coverage is partial; off-support states fall back to uniform "
                        "in the averaged policy, so these numbers are a coarse proxy, not a "
                        "convergence certificate. The MEANINGFUL comparison is avg vs instantaneous."
                    ),
                }
            )
        except Exception as exc:  # pragma: no cover - diagnostic best-effort
            exploit["error"] = f"{type(exc).__name__}: {exc}"

    summary = {
        "config": {
            "init_from_checkpoint": str(args.init_from_checkpoint) if args.init_from_checkpoint else None,
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "n_cards": int(args.n_cards),
            "device": device,
            "lr": args.lr,
            "seed": args.seed,
            "variant": args.variant,
            "lineage": {k: lineage.get(k) for k in ("parent_checkpoint_id", "init_checkpoint_id", "optimizer_reset")},
        },
        "saw_nonfinite": saw_nonfinite,
        "n_logged": len(rows),
        "first_row": rows[0] if rows else None,
        "last_row": rows[-1] if rows else None,
        "exploitability": exploit,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Experiment D -- diagnostic probe of Stage-4 NeuRD dynamics")
    ap.add_argument("--init-from-checkpoint", type=str, default=None, help="Stage-3 checkpoint to inherit theta from")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-cards", type=int, default=5)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--variant", choices=["before", "after"], default="after")
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--exploit", action="store_true", help="also compute full-game exploitability (n<=6)")
    args = ap.parse_args()
    run_diag(args)


if __name__ == "__main__":
    main()
