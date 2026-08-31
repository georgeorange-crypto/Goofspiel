"""Runnable training stages.  Torch is imported lazily inside stage methods."""

from __future__ import annotations

import itertools
import json
import random
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from goofspiel.game import GameState, legal_cards, transition
from goofspiel.observability import BaseEvent, JsonlEventSink, collect_system_metrics
from goofspiel.training.checkpoint import (
    CheckpointMetadata,
    model_config_hash,
    save_checkpoint,
)
from goofspiel.training.checkpoint import init_from_checkpoint as _load_init_from_checkpoint
from goofspiel.training.checkpoint import resume_checkpoint as _load_resume_checkpoint
from goofspiel.training.checkpoint_registry import CheckpointRegistry
from goofspiel.training.corpus import generate_random_game_corpus
from goofspiel.training.data import FailureRecord, JsonlStore, ReanalysisRecord, RobustTrajectorySample, RoundRecord, state_record_from_game_state
from goofspiel.training.distributed import barrier_if_distributed, setup_torch_distributed
from goofspiel.training.evaluation import evaluate_bot_matchup, exact_feasibility_sweep
from goofspiel.training.league import LeagueAgent, LeagueRegistry, ROLE_AGGRESSIVE, ROLE_EXPLOITER, ROLE_ROBUST
from goofspiel.training.curriculum import ProgressiveCurriculum
from goofspiel.training.promotion import evaluate_promotion_candidate, write_promotion_report
from goofspiel.training.replay import TrajectoryReplayBuffer
from goofspiel.training.redteam import CorrectionDataset, FailureBuffer
from goofspiel.training.stage0_verify import run_stage0_verify
from goofspiel.training.state_coverage import coverage_report, sample_reachable_states
from goofspiel.training.pretraining import build_pretraining_targets
from goofspiel.training.adaptive import default_opponent_curriculum, opponent_action_for_regime, oracle_opponent_diagnostic
from goofspiel.training.teachers import TeacherRouter


@dataclass
class StageMetrics:
    stage: str
    steps: int
    metrics: dict[str, float]
    checkpoint: str | None = None


def _torch_import():
    try:
        import torch
        import torch.nn.functional as F

        return torch, F
    except Exception as exc:  # pragma: no cover - depends on local torch install
        raise RuntimeError(
            "PyTorch is required for neural training stages. "
            "Run `python -c \"import torch; print(torch.__version__)\"` first. "
            f"Original import error: {exc!r}"
        ) from exc


def _sample_states(batch_size: int, *, n: int, step: int) -> list[GameState]:
    states = []
    full = (1 << n) - 1
    prizes = list(range(1, n + 1))
    for i in range(batch_size):
        p = prizes[(step + i) % n]
        states.append(GameState(n=n, self_mask=full, opp_mask=full, prize_mask=full & ~(1 << (p - 1)), current_prize=p))
    return states


def _write_coverage_artifact(out_dir: Path, stage: str, states: list[GameState]) -> Path:
    """Persist the auditable coverage report (buckets + histograms) for a stage.

    Phase 2.1 makes state coverage a first-class artifact: a training report can
    be inspected to confirm P1/P3 actually saw endgames, carry crises, score
    deficits, and asymmetric hands rather than only opening states.
    """
    report = coverage_report(states)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stage}_state_coverage.json"
    payload = {
        "stage": stage,
        "total_states": report.total,
        "bucket_counts": report.bucket_counts,
        "missing_buckets": report.missing_buckets(),
        # Histograms keyed by string for JSON portability.
        "histograms": {
            axis: {str(k): v for k, v in sorted(hist.items())} for axis, hist in report.histograms.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _immediate_target(states: list[GameState], max_cards: int):
    torch, _F = _torch_import()
    q = torch.zeros(len(states), max_cards, max_cards)
    mask = torch.zeros(len(states), max_cards, max_cards, dtype=torch.bool)
    for b, state in enumerate(states):
        total = state.n * (state.n + 1) // 2
        stake = state.current_prize + state.carry_pool
        for a in state.self_actions:
            for o in state.opponent_actions:
                q[b, a - 1, o - 1] = stake * (1 if a > o else (-1 if a < o else 0)) / total
                mask[b, a - 1, o - 1] = True
    return q, mask


def _apply_init_or_resume(
    model: Any,
    *,
    init_from_checkpoint_path: str | Path | None,
    resume_checkpoint_path: str | Path | None,
    optimizers: dict[str, Any] | None = None,
    target_model: Any | None = None,
) -> dict[str, Any]:
    """Apply the two DISTINCT checkpoint interfaces to a freshly-built model.

    ``init_from_checkpoint_path`` is a **stage transition** (P1→P3→P4): copy θ
    only, keep the fresh optimizer / step-0.  ``resume_checkpoint_path`` is a
    **crash resume**: restore the full training state.  They are mutually
    exclusive; conflating them is the bug 3.1 exists to prevent.  Returns a
    lineage dict to fold into the stage's ``CheckpointMetadata``.
    """
    if init_from_checkpoint_path and resume_checkpoint_path:
        raise ValueError(
            "init_from_checkpoint and resume_checkpoint are mutually exclusive: "
            "'inherit θ across a stage boundary' and 'resume a crashed run' are "
            "different operations and must not be conflated."
        )
    lineage: dict[str, Any] = {
        "parent_checkpoint_id": None,
        "init_checkpoint_id": None,
        "optimizer_reset": True,
        "resumed_global_step": 0,
    }
    target = getattr(model, "module", model)
    if init_from_checkpoint_path:
        prov = _load_init_from_checkpoint(target, init_from_checkpoint_path)
        lineage["init_checkpoint_id"] = prov["init_checkpoint_id"]
        lineage["parent_checkpoint_id"] = prov["init_checkpoint_id"]
        lineage["init_checkpoint_sha256"] = prov["init_checkpoint_sha256"]
        lineage["optimizer_reset"] = True  # a stage boundary always resets the optimizer
    elif resume_checkpoint_path:
        prov = _load_resume_checkpoint(
            target, resume_checkpoint_path, optimizers=optimizers, target_model=target_model
        )
        lineage["parent_checkpoint_id"] = prov["parent_checkpoint_id"]
        lineage["resume_sha256"] = prov["resume_sha256"]
        lineage["optimizer_reset"] = False  # a resume restores the optimizer
        lineage["resumed_global_step"] = prov["global_step"]
    return lineage


def run_stage1_pretrain(
    *,
    steps: int,
    batch_size: int,
    out_dir: str | Path,
    device: str = "cpu",
    n_cards: int = 13,
    lr: float = 3e-4,
    init_from_checkpoint: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
) -> StageMetrics:
    """P1 pretraining over swap, transition, joint-outcome, and opponent tasks."""
    torch, F = _torch_import()
    from goofspiel.models import GoofspielModel, public_state_from_game

    runtime, device = setup_torch_distributed(device)
    model = GoofspielModel(max_cards=13).to(device)
    if runtime.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[runtime.local_rank] if device.startswith("cuda") else None)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lineage = _apply_init_or_resume(
        model,
        init_from_checkpoint_path=init_from_checkpoint,
        resume_checkpoint_path=resume_checkpoint,
        optimizers={"public_pretrain": opt},
    )
    losses: list[float] = []
    metrics: dict[str, float] = {}
    all_states: list[GameState] = []
    for step in range(int(steps)):
        states = sample_reachable_states(batch_size, n=n_cards, step=step, seed=1)
        all_states.extend(states)
        batch = public_state_from_game(states, max_cards=13, device=device)
        p1_targets = [
            build_pretraining_targets(state, self_action=state.self_actions[0], opponent_action=state.opponent_actions[-1])
            for state in states
        ]
        target_q, mask = _immediate_target(states, 13)
        target_q = target_q.to(device)
        mask = mask.to(device)
        out = model(batch)
        loss_q = F.smooth_l1_loss(out.q_robust[mask], target_q[mask], beta=0.1)
        swapped_batch = public_state_from_game([target.player_swap_state for target in p1_targets], max_cards=13, device=device)
        swapped_out = model(swapped_batch)
        swap_loss = F.smooth_l1_loss(out.q_robust, -swapped_out.q_robust.transpose(1, 2), beta=0.1)
        opp_targets = torch.tensor([target.future_opponent_action - 1 for target in p1_targets], dtype=torch.long, device=device)
        opp_loss = F.cross_entropy(out.opponent_fused_logits, opp_targets)
        # Phase 3.2(a): give the SHORT- and LONG-horizon opponent heads their own
        # supervised signal too — previously only the fused head was trained in P1,
        # leaving opp_short/opp_long with no P1 gradient at all.  With no history /
        # memory fed here the LSTM/Mamba states are zero, so in P1 all three heads
        # learn the same public-state → opponent-action map; the short/long
        # DIVERGENCE is what P5 induces once real intra-/inter-game context flows.
        opp_short_loss = F.cross_entropy(out.opponent_short_logits, opp_targets)
        opp_long_loss = F.cross_entropy(out.opponent_long_logits, opp_targets)
        self_targets = torch.tensor([target.masked_history_action - 1 for target in p1_targets], dtype=torch.long, device=device)
        masked_action_loss = F.cross_entropy(out.robust_policy_logits, self_targets)
        style_loss = (1.0 + F.cosine_similarity(out.public_embedding, swapped_out.public_embedding.detach(), dim=-1)).clamp_min(0.0).mean()
        loss = (
            loss_q + 0.05 * swap_loss + 0.05 * opp_loss + 0.02 * masked_action_loss + 0.01 * style_loss
            + 0.05 * opp_short_loss + 0.05 * opp_long_loss
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
        metrics = {
            "loss_last": float(loss.detach().cpu()),
            "immediate_joint_outcome_loss": float(loss_q.detach().cpu()),
            "player_swap_loss": float(swap_loss.detach().cpu()),
            "future_opponent_behaviour_loss": float(opp_loss.detach().cpu()),
            "opponent_short_behaviour_loss": float(opp_short_loss.detach().cpu()),
            "opponent_long_behaviour_loss": float(opp_long_loss.detach().cpu()),
            "masked_history_action_loss": float(masked_action_loss.detach().cpu()),
            "style_contrastive_loss": float(style_loss.detach().cpu()),
            "known_transition_samples": float(len(p1_targets)),
        }

    checkpoint_path = None
    if metrics:
        metrics.update(coverage_report(all_states).as_metrics())
    if runtime.is_rank0:
        out_dir = Path(out_dir)
        ckpt = out_dir / "stage1_pretrain.pt"
        if all_states:
            _write_coverage_artifact(out_dir, "stage1_pretrain", all_states)
        manifest = save_checkpoint(
            ckpt,
            model=getattr(model, "module", model),
            optimizers={"public_pretrain": opt},
            metadata=CheckpointMetadata(
                checkpoint_id="stage1_pretrain",
                training_stage="P1_PRETRAIN",
                global_step=int(steps),
                policy_version=0,
                config={"steps": steps, "batch_size": batch_size, "n_cards": n_cards, "lr": lr, "world_size": runtime.world_size},
                metrics=metrics or {"loss_last": 0.0},
                parent_checkpoint_id=lineage["parent_checkpoint_id"],
                init_checkpoint_id=lineage["init_checkpoint_id"],
                model_config_hash=model_config_hash(getattr(model, "module", model)),
                optimizer_reset=lineage["optimizer_reset"],
            ),
        )
        registry = CheckpointRegistry(out_dir / "registry")
        registry.register("latest", ckpt, global_step=int(steps), metrics=metrics)
        checkpoint_path = manifest["path"]
    barrier_if_distributed()
    return StageMetrics("P1_PRETRAIN", int(steps), metrics or {"loss_last": 0.0}, checkpoint_path)


def _game_state_from_record(rec: dict[str, Any]) -> GameState:
    """Reconstruct a GameState from a persisted PublicStateRecord dict."""
    return GameState(
        n=int(rec["n"]),
        self_mask=int(rec["self_mask"]),
        opp_mask=int(rec["opponent_mask"]),
        prize_mask=int(rec["prize_mask"]),
        current_prize=int(rec["current_prize"]),
        self_score=int(rec.get("self_score", 0)),
        opp_score=int(rec.get("opponent_score", 0)),
        round_index=int(rec.get("round_index", 1)),
        done=bool(rec.get("done", False)),
        carry_pool=int(rec.get("carry_pool", 0)),
    )


def _load_teacher_dataset(path: Path) -> list[dict[str, Any]]:
    """Load teacher samples from a JSONL dataset, if present."""
    if not path.exists():
        return []
    store = JsonlStore(path)
    return list(store.iter_dicts())


def run_stage3_sft(
    *,
    steps: int,
    batch_size: int,
    out_dir: str | Path,
    device: str = "cpu",
    n_cards: int = 5,
    lr: float = 2e-4,
    teacher_dataset_path: str | Path | None = None,
    init_from_checkpoint: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
) -> StageMetrics:
    """P3 Robust Strategic SFT that CONSUMES the multi-source teacher dataset.

    Phase 2.2b: P3 now loads ``teacher_dataset.jsonl`` (written by P2) and trains
    the robust policy toward the *stored* teacher policies — so its loss depends
    on the file's content and its four source counts (CFR/SEARCH/EXACT/PSEUDO)
    measure four genuinely different computations, not one aliased number.  When
    the dataset is absent (e.g. a stage run in isolation) it falls back to
    solving reachable states on the fly, and the source counts are reported as
    ``fallback``.

    Phase 3.1: ``init_from_checkpoint`` inherits P1's θ (fresh optimizer);
    ``resume_checkpoint`` restores a crashed P3 run in full.  The two are never
    conflated.
    """
    torch, F = _torch_import()
    from goofspiel.learning.game_theory.regret_matching_plus import solve_batch
    from goofspiel.models import GoofspielModel, public_state_from_game
    from goofspiel.training.teacher_dataset import ROBUST_TEACHER_SOURCES

    runtime, device = setup_torch_distributed(device)
    model = GoofspielModel(max_cards=13).to(device)
    if runtime.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[runtime.local_rank] if device.startswith("cuda") else None)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lineage = _apply_init_or_resume(
        model,
        init_from_checkpoint_path=init_from_checkpoint,
        resume_checkpoint_path=resume_checkpoint,
        optimizers={"strategic_sft": opt},
    )
    last_loss = 0.0
    all_states: list[GameState] = []

    # Prefer the dataset written by P2; fall back to on-the-fly reachable states.
    if teacher_dataset_path is None:
        teacher_dataset_path = Path(out_dir).parent / "data" / "teacher_dataset.jsonl"
    dataset = _load_teacher_dataset(Path(teacher_dataset_path))
    source_counts = {src: 0 for src in ROBUST_TEACHER_SOURCES}

    if dataset:
        # Train directly on the stored teacher targets. Loss now RESPONDS to the
        # dataset: the KL target is the source-specific teacher policy.
        for sample in dataset:
            src = sample.get("teacher_source")
            if src in source_counts:
                source_counts[src] += 1
        step_indices = list(range(int(steps)))
        for step in step_indices:
            start = (step * batch_size) % len(dataset)
            window = [dataset[(start + i) % len(dataset)] for i in range(batch_size)]
            states = [_game_state_from_record(s["state"]) for s in window]
            all_states.extend(states)
            batch = public_state_from_game(states, max_cards=13, device=device)
            target_q, mask = _immediate_target(states, 13)
            target_q = target_q.to(device)
            mask = mask.to(device)
            # Source-specific stored teacher policy (length-13, padded), the signal
            # that makes P3 loss depend on which source produced each anchor.
            teacher_policy = torch.zeros(len(window), 13, device=device)
            for b, s in enumerate(window):
                pol = s.get("teacher_policy") or []
                for i, v in enumerate(pol[:13]):
                    teacher_policy[b, i] = float(v)
            teacher_policy = teacher_policy / teacher_policy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            out = model(batch)
            q_loss = F.smooth_l1_loss(out.q_robust[mask], target_q[mask], beta=0.1)
            logp = F.log_softmax(out.robust_policy_logits, dim=-1)
            pi_loss = F.kl_div(logp, teacher_policy, reduction="batchmean")
            loss = q_loss + 0.1 * pi_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_loss = float(loss.detach().cpu())
    else:
        for step in range(int(steps)):
            states = sample_reachable_states(batch_size, n=n_cards, step=step, seed=1)
            all_states.extend(states)
            batch = public_state_from_game(states, max_cards=13, device=device)
            target_q, mask = _immediate_target(states, 13)
            target_q = target_q.to(device)
            mask = mask.to(device)
            target_policy = solve_batch(target_q, batch.self_action_mask, batch.opponent_action_mask, iterations=128).row_policy.detach()
            out = model(batch)
            q_loss = F.smooth_l1_loss(out.q_robust[mask], target_q[mask], beta=0.1)
            logp = F.log_softmax(out.robust_policy_logits, dim=-1)
            pi_loss = F.kl_div(logp, target_policy, reduction="batchmean")
            loss = q_loss + 0.1 * pi_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            last_loss = float(loss.detach().cpu())

    checkpoint_path = None
    metrics = {
        "loss_last": last_loss,
        # Four DISTINCT robust-only teacher-source counts, consumed from the P2
        # dataset — no longer four aliases of one number (Phase 2.2b). Each source
        # is a different algorithm/search depth (CFR immediate, SEARCH depth-1,
        # EXACT full recursion, PSEUDO confident self-labels); none uses opponent
        # behaviour (that is P5's job, preserving Q_R ⊥ Q_A).
        "strategic_sft_samples": float(sum(source_counts.values())),
        "teacher_dataset_consumed": float(1.0 if dataset else 0.0),
    }
    for src in ROBUST_TEACHER_SOURCES:
        metrics[f"sft_source_{src.lower()}_samples"] = float(source_counts[src])
    metrics.update(coverage_report(all_states).as_metrics())
    if runtime.is_rank0:
        out_dir = Path(out_dir)
        ckpt = out_dir / "stage3_sft.pt"
        if all_states:
            _write_coverage_artifact(out_dir, "stage3_sft", all_states)
        manifest = save_checkpoint(
            ckpt,
            model=getattr(model, "module", model),
            optimizers={"strategic_sft": opt},
            metadata=CheckpointMetadata(
                checkpoint_id="stage3_sft",
                training_stage="P3_STRATEGIC_SFT",
                global_step=int(steps),
                policy_version=1,
                config={"steps": steps, "batch_size": batch_size, "n_cards": n_cards, "lr": lr, "world_size": runtime.world_size},
                metrics=metrics,
                parent_checkpoint_id=lineage["parent_checkpoint_id"],
                init_checkpoint_id=lineage["init_checkpoint_id"],
                model_config_hash=model_config_hash(getattr(model, "module", model)),
                teacher_dataset_ids=[str(teacher_dataset_path)] if dataset else [],
                optimizer_reset=lineage["optimizer_reset"],
            ),
        )
        registry = CheckpointRegistry(out_dir / "registry")
        registry.register("latest", ckpt, global_step=int(steps), metrics=metrics)
        registry.register("teacher_ema", ckpt, global_step=int(steps), metrics=metrics)
        checkpoint_path = manifest["path"]
    barrier_if_distributed()
    return StageMetrics("P3_STRATEGIC_SFT", int(steps), metrics, checkpoint_path)


def run_stage2_semi_supervised(
    *,
    steps: int,
    out_dir: str | Path,
    n_cards: int = 5,
) -> StageMetrics:
    """Generate the multi-source robust teacher dataset consumed by P3.

    Phase 2.2b: instead of a single confidence-filtered ensemble label, P2 now
    labels reachable states with FOUR genuinely distinct robust-only sources
    (CFR / SEARCH / EXACT / PSEUDO) that differ by algorithm and search depth,
    writing them all to ``teacher_dataset.jsonl`` — the file P3 actually trains
    on.  None of the sources uses opponent behaviour (that stays in P5).
    """
    from goofspiel.training.teacher_dataset import ROBUST_TEACHER_SOURCES, build_teacher_dataset

    store = JsonlStore(Path(out_dir) / "teacher_dataset.jsonl")
    states: list[GameState] = []
    for step in range(int(steps)):
        states.extend(sample_reachable_states(max(8, n_cards * 2), n=n_cards, step=step, seed=1))
    counts = build_teacher_dataset(states, store)
    total = sum(counts.values())
    metrics = {f"teacher_source_{src.lower()}_samples": float(counts[src]) for src in ROBUST_TEACHER_SOURCES}
    metrics["teacher_samples"] = float(total)
    metrics["distinct_teacher_sources"] = float(len({v for v in counts.values() if v > 0}))
    return StageMetrics("P2_SEMI_SUPERVISED", int(steps), metrics, None)


def _mirrored_state(state: GameState) -> GameState:
    return GameState(
        n=state.n,
        self_mask=state.opp_mask,
        opp_mask=state.self_mask,
        prize_mask=state.prize_mask,
        current_prize=state.current_prize,
        self_score=state.opp_score,
        opp_score=state.self_score,
        round_index=state.round_index,
        done=state.done,
        carry_pool=state.carry_pool,
    )


def _uniform_policy(cards: list[int], max_cards: int) -> list[float]:
    policy = [0.0] * max_cards
    if cards:
        p = 1.0 / len(cards)
        for card in cards:
            policy[card - 1] = p
    return policy


def _select_model_action(model: Any, state: GameState, *, device: str, temperature: float = 1.0) -> tuple[int, list[float], float]:
    torch, F = _torch_import()
    from goofspiel.models import public_state_from_game

    batch = public_state_from_game([state], max_cards=13, device=device)
    with torch.no_grad():
        logits = model(batch).robust_policy_logits[0]
        logits = logits / max(float(temperature), 1e-6)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action_idx = int(dist.sample().item())
        prob = float(probs[action_idx].detach().cpu())
    action = action_idx + 1
    if action not in state.self_actions:
        action = random.choice(state.self_actions)
        prob = 1.0 / len(state.self_actions)
    policy = [0.0] * 13
    for i, value in enumerate(probs.detach().cpu().tolist()[:13]):
        policy[i] = float(value)
    return action, policy, prob


def _rollout_selfplay_game(
    model: Any,
    *,
    n_cards: int,
    rng: random.Random,
    device: str,
    model_version: str,
    game_index: int,
) -> RobustTrajectorySample:
    first_prize = rng.choice(list(range(1, n_cards + 1)))
    state = GameState.initial(n_cards, current_prize=first_prize)
    states = []
    rounds = []
    self_policies = []
    opp_policies = []
    self_probs = []
    opp_probs = []
    while not state.done:
        states.append(state_record_from_game_state(state))
        self_action, self_policy, self_prob = _select_model_action(model, state, device=device)
        opp_view = _mirrored_state(state)
        opp_action, opp_policy, opp_prob = _select_model_action(model, opp_view, device=device)
        next_prize = rng.choice(legal_cards(state.prize_mask, state.n)) if state.prize_mask else None
        result = transition(state, self_action, opp_action, next_prize=next_prize)
        rounds.append(
            RoundRecord(
                round_index=state.round_index,
                prize=state.current_prize,
                self_action=self_action,
                opponent_action=opp_action,
                reward_self=result.reward_self,
                reward_opponent=result.reward_opp,
                carry_in=state.carry_pool,
                carry_out=result.state.carry_pool,
                done=result.state.done,
            )
        )
        self_policies.append(self_policy)
        opp_policies.append(opp_policy)
        self_probs.append(self_prob)
        opp_probs.append(opp_prob)
        state = result.state
    return RobustTrajectorySample(
        sample_id=f"selfplay_{game_index}_{uuid.uuid4().hex}",
        states=states,
        rounds=rounds,
        behavior_policy_self=self_policies,
        behavior_policy_opponent=opp_policies,
        action_prob_self=self_probs,
        action_prob_opponent=opp_probs,
        final_score_diff=state.self_score - state.opp_score,
        model_version=model_version,
        opponent_version=model_version,
        n=n_cards,
    )


def _flatten_trajectory_batch(trajectories: list[RobustTrajectorySample]) -> tuple[list[GameState], list[int], list[int], list[float]]:
    states: list[GameState] = []
    self_actions: list[int] = []
    opp_actions: list[int] = []
    returns: list[float] = []
    for traj in trajectories:
        total = traj.n * (traj.n + 1) / 2
        final_return = float(traj.final_score_diff) / float(total)
        for state_record, round_record in zip(traj.states, traj.rounds):
            states.append(
                GameState(
                    n=state_record.n,
                    self_mask=state_record.self_mask,
                    opp_mask=state_record.opponent_mask,
                    prize_mask=state_record.prize_mask,
                    current_prize=state_record.current_prize,
                    self_score=state_record.self_score,
                    opp_score=state_record.opponent_score,
                    round_index=state_record.round_index,
                    done=state_record.done,
                    carry_pool=state_record.carry_pool,
                )
            )
            self_actions.append(round_record.self_action - 1)
            opp_actions.append(round_record.opponent_action - 1)
            returns.append(final_return)
    return states, self_actions, opp_actions, returns


def run_stage4_robust_rl(
    *,
    steps: int,
    batch_size: int,
    out_dir: str | Path,
    device: str = "cpu",
    n_cards: int = 5,
    lr: float = 1e-4,
    init_from_checkpoint: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
) -> StageMetrics:
    """Self-play robust RL runner using trajectory replay + Nash/NeuRD anchors.

    Phase 3.1: ``init_from_checkpoint`` inherits P3's θ (θ-only, fresh optimizer;
    the target network is then seeded from the inherited weights).
    ``resume_checkpoint`` restores model + optimizer + target network in full.
    """
    torch, F = _torch_import()
    from goofspiel.learning.game_theory.neurd import row_action_regret
    from goofspiel.learning.game_theory.regret_matching_plus import solve_batch
    from goofspiel.models import GoofspielModel, public_state_from_game

    runtime, device = setup_torch_distributed(device)
    model = GoofspielModel(max_cards=13).to(device)
    target_model = GoofspielModel(max_cards=13).to(device)
    opt_q = torch.optim.AdamW(model.parameters(), lr=lr)
    # Load θ (init, θ-only) or restore full state (resume, incl. target_model)
    # BEFORE seeding the target network, so an inherited P3 θ propagates into it.
    lineage = _apply_init_or_resume(
        model,
        init_from_checkpoint_path=init_from_checkpoint,
        resume_checkpoint_path=resume_checkpoint,
        optimizers={"robust_rl": opt_q},
        target_model=target_model,
    )
    if not resume_checkpoint:
        # Fresh or θ-inherited: the target net tracks the (possibly inherited)
        # online weights.  On resume the target was already restored from `extra`.
        target_model.load_state_dict(getattr(model, "module", model).state_dict())
    target_model.eval()
    if runtime.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[runtime.local_rank] if device.startswith("cuda") else None)
    replay_buffer = TrajectoryReplayBuffer(Path(out_dir) / "replay" / "selfplay_robust.jsonl")
    event_sink = JsonlEventSink(Path(out_dir) / "events" / "stage4_robust_rl.jsonl")
    warmup_n = n_cards if int(steps) <= 1 else min(3, n_cards)
    curriculum = ProgressiveCurriculum(target_n=n_cards, warmup_n=warmup_n, ramp_every=max(1, int(steps) // max(1, n_cards - warmup_n + 1)))
    rng = random.Random(17 + n_cards + batch_size)
    last_q = last_actor = last_pg = last_entropy = 0.0
    trajectories_total = transitions_total = 0
    curriculum_path = Path(out_dir) / "curriculum" / "stage4_manifest.json"
    curriculum_path.parent.mkdir(parents=True, exist_ok=True)
    curriculum_path.write_text(json.dumps(curriculum.manifest(steps=int(steps)), indent=2, ensure_ascii=False), encoding="utf-8")
    for step in range(int(steps)):
        cstep = curriculum.at(step)
        collector = getattr(model, "module", model)
        collector.eval()
        trajectories = [
            _rollout_selfplay_game(
                collector,
                n_cards=cstep.n_cards,
                rng=rng,
                device=device,
                model_version=f"stage4_step_{step}",
                game_index=step * batch_size + i,
            )
            for i in range(int(batch_size))
        ]
        if runtime.is_rank0:
            replay_buffer.append_many(trajectories)
            event_sink.emit(
                BaseEvent(
                    event_type="STAGE4_SELFPLAY_COLLECTED",
                    run_id="stage4_robust_rl",
                    step=step,
                    payload={
                        "curriculum_n": cstep.n_cards,
                        "trajectories": len(trajectories),
                        "transitions": sum(len(t.rounds) for t in trajectories),
                        "mean_final_score_diff": sum(t.final_score_diff for t in trajectories) / max(len(trajectories), 1),
                    },
                )
            )
        trajectories_total += len(trajectories)
        sampled_trajectories = replay_buffer.sample(batch_size, rng) if runtime.is_rank0 else trajectories
        states, action_self, action_opp, mc_returns = _flatten_trajectory_batch(sampled_trajectories or trajectories)
        batch = public_state_from_game(states, max_cards=13, device=device)
        idx = torch.arange(len(states), device=device)
        action_self_t = torch.tensor(action_self, dtype=torch.long, device=device)
        action_opp_t = torch.tensor(action_opp, dtype=torch.long, device=device)
        returns_t = torch.tensor(mc_returns, dtype=torch.float32, device=device)
        out = model(batch)
        with torch.no_grad():
            target_out = target_model(batch)
        sol = solve_batch(target_out.q_robust.detach(), batch.self_action_mask, batch.opponent_action_mask, iterations=64)
        chosen_q = out.q_robust[idx, action_self_t, action_opp_t]
        bootstrapped_returns = 0.8 * returns_t + 0.2 * target_out.q_robust[idx, action_self_t, action_opp_t].detach()
        q_loss = F.smooth_l1_loss(chosen_q, bootstrapped_returns, beta=0.1)
        logp = F.log_softmax(out.robust_policy_logits, dim=-1)[idx, action_self_t]
        baseline = chosen_q.detach()
        pg_loss = -(logp * (returns_t - baseline)).mean()
        policy = F.softmax(out.robust_policy_logits, dim=-1)
        entropy = -(policy * F.log_softmax(out.robust_policy_logits, dim=-1)).sum(dim=-1).mean()
        # NeuRD robust actor (Phase 1.1). The former `neurd_loss` contracted Q with
        # `max_b` over opponent actions on the *self* payoff — i.e. it pulled the
        # actor toward the BEST-case opponent, the opposite of robust, fighting the
        # RM+ minimax anchor. We now use `row_action_regret`, which contracts Q
        # against the RM+ equilibrium column policy (`sol.column_policy`) — the
        # worst-case-consistent opponent the anchor itself computed — so the actor
        # gradient and the Nash anchor pull the same direction (toward minimax).
        action_regret = row_action_regret(
            out.q_robust.detach(),
            policy.detach(),
            sol.column_policy.detach(),
            batch.self_action_mask,
        )
        self_mask_f = batch.self_action_mask.float()
        denom = self_mask_f.sum(dim=-1).clamp_min(1.0)
        actor_loss = -(
            (action_regret.detach() * out.robust_policy_logits * self_mask_f).sum(dim=-1) / denom
        ).mean()
        anchor = F.kl_div(F.log_softmax(out.robust_policy_logits, dim=-1), sol.row_policy.detach(), reduction="batchmean")
        loss = q_loss + actor_loss + pg_loss + 0.1 * anchor - 0.01 * entropy
        opt_q.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt_q.step()
        with torch.no_grad():
            source_model = getattr(model, "module", model)
            for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
                target_param.mul_(0.995).add_(source_param.detach(), alpha=0.005)
        last_q = float(q_loss.detach().cpu())
        last_actor = float(actor_loss.detach().cpu())
        last_pg = float(pg_loss.detach().cpu())
        last_entropy = float(entropy.detach().cpu())
        transitions_total += len(states)

    checkpoint_path = None
    stage_metrics = {
        "q_loss_last": last_q,
        "actor_loss_last": last_actor,
        "policy_gradient_loss_last": last_pg,
        "entropy_last": last_entropy,
        "selfplay_trajectories": float(trajectories_total),
        "selfplay_transitions": float(transitions_total),
        "replay_samples": float(replay_buffer.count()) if runtime.is_rank0 else 0.0,
        "target_network_ema": 0.995,
        "curriculum_final_n": float(curriculum.at(max(0, int(steps) - 1)).n_cards),
    }
    promotion = evaluate_promotion_candidate(stage_metrics)
    promotion_artifact = None
    if runtime.is_rank0:
        ckpt = Path(out_dir) / "stage4_robust_rl.pt"
        promotion_artifact = write_promotion_report(promotion, Path(out_dir) / "promotion" / "stage4_promotion.json")
        manifest = save_checkpoint(
            ckpt,
            model=getattr(model, "module", model),
            optimizers={"robust_rl": opt_q},
            metadata=CheckpointMetadata(
                checkpoint_id="stage4_robust_rl",
                training_stage="P4_ROBUST_RL",
                global_step=int(steps),
                policy_version=2,
                config={"steps": steps, "batch_size": batch_size, "n_cards": n_cards, "lr": lr, "world_size": runtime.world_size},
                metrics=stage_metrics,
                parent_checkpoint_id=lineage["parent_checkpoint_id"],
                init_checkpoint_id=lineage["init_checkpoint_id"],
                model_config_hash=model_config_hash(getattr(model, "module", model)),
                optimizer_reset=lineage["optimizer_reset"],
            ),
            extra={
                "replay_path": str(replay_buffer.path),
                "event_log": str(event_sink.path),
                "curriculum": str(curriculum_path),
                "promotion": promotion_artifact,
                "target_model_state": target_model.state_dict(),
            },
        )
        registry = CheckpointRegistry(Path(out_dir) / "registry")
        # Register only `latest`. The former best_robust/best_raw/best_search/
        # best_generalization aliases all pointed at this same file, implying a
        # per-axis model selection that never ran. Re-introduce a `best_*` alias
        # only when an evaluation actually selects a distinct checkpoint for it.
        registry.register("latest", ckpt, global_step=int(steps), metrics=stage_metrics)
        checkpoint_path = manifest["path"]
    barrier_if_distributed()
    stage_metrics["promotion_candidate"] = 1.0 if promotion.decision == "PROMOTE_CANDIDATE" else 0.0
    stage_metrics["replay_persisted_samples"] = float(replay_buffer.persisted_count()) if runtime.is_rank0 else 0.0
    return StageMetrics(
        "P4_ROBUST_RL",
        int(steps),
        stage_metrics,
        checkpoint_path,
    )


def run_stage5_adaptive(
    *,
    steps: int,
    out_dir: str | Path,
    n_cards: int = 5,
) -> StageMetrics:
    """Build opponent sessions and run a curriculum calibration gate."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(503 + int(steps) + int(n_cards))
    sessions_path = out / "adaptive" / "opponent_sessions.jsonl"
    sessions = JsonlStore(sessions_path)
    session_rows = []
    total_predictions = 0
    correct_uniform_bucket = 0
    nll_total = 0.0
    rounds_total = 0
    regimes = default_opponent_curriculum()
    for session_idx in range(max(1, int(steps))):
        regime = regimes[session_idx % len(regimes)]
        first_prize = rng.choice(list(range(1, n_cards + 1)))
        state = GameState.initial(n_cards, current_prize=first_prize)
        rounds = []
        while not state.done:
            self_cards = state.self_actions
            opp_cards = state.opponent_actions
            self_action = rng.choice(self_cards)
            opp_action = opponent_action_for_regime(
                regime.regime_id,
                opp_cards,
                stake=state.stake,
                n_cards=n_cards,
                rng=rng,
            )
            # No opponent model exists yet (P5 does not train one — Phase 5).
            # This is the *uniform reference* likelihood: a constant log(len)
            # per prediction. It is named honestly and must never be presented
            # as a trained model's calibration.
            prob = 1.0 / len(opp_cards)
            total_predictions += 1
            correct_uniform_bucket += 1 if prob >= 1.0 / len(opp_cards) else 0
            nll_total += -__import__("math").log(max(prob, 1e-9))
            next_prize = rng.choice(legal_cards(state.prize_mask, state.n)) if state.prize_mask else None
            result = transition(state, self_action, opp_action, next_prize=next_prize)
            rounds.append(
                RoundRecord(
                    round_index=state.round_index,
                    prize=state.current_prize,
                    self_action=self_action,
                    opponent_action=opp_action,
                    reward_self=result.reward_self,
                    reward_opponent=result.reward_opp,
                    carry_in=state.carry_pool,
                    carry_out=result.state.carry_pool,
                    done=result.state.done,
                )
            )
            state = result.state
        rounds_total += len(rounds)
        from goofspiel.training.data import OpponentSession

        session = OpponentSession(
            session_id=f"adaptive_session_{session_idx}_{uuid.uuid4().hex}",
            opponent_id=f"curriculum_{regime.regime_id}",
            strategy_regime_id=regime.regime_id,
            games=[rounds],
        )
        sessions.append(session)
        session_rows.append(session)
    uniform_reference_nll = nll_total / max(total_predictions, 1)
    oracle = oracle_opponent_diagnostic(session_rows, n_cards=n_cards)
    report = {
        # No opponent model is trained in P5 yet (Phase 5). Report this honestly
        # rather than claiming a usable model. ECE is omitted (not fabricated as
        # 0.0) because there is no probabilistic predictor to calibrate.
        "opponent_model_usable": False,
        "gate": "CURRICULUM_BUILT_NO_MODEL",
        "calibration": {
            "sessions": max(1, int(steps)),
            "rounds": rounds_total,
            "uniform_reference_nll": uniform_reference_nll,
            "uniform_bucket_accuracy": correct_uniform_bucket / max(total_predictions, 1),
            "oracle_accuracy": oracle["oracle_accuracy"],
            "oracle_gain": oracle["oracle_gain"],
            "switch_delay": oracle["switch_delay"],
        },
        "opponent_curriculum": [regime.__dict__ for regime in regimes],
        "required": {"nll_better_than_uniform": False, "switch_benchmark": "curriculum_gate"},
        "session_path": str(sessions_path),
    }
    (out / "adaptive" / "adaptive_gate_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return StageMetrics(
        "P5_ADAPTIVE",
        int(steps),
        {
            # Honest: no trained opponent model exists (Phase 5). 0.0 = not usable.
            "opponent_model_usable": 0.0,
            "opponent_sessions": float(max(1, int(steps))),
            "opponent_rounds": float(rounds_total),
            "uniform_reference_nll": float(uniform_reference_nll),
            "opponent_regimes": float(len({s.strategy_regime_id for s in session_rows})),
            "oracle_gain": float(oracle["oracle_gain"]),
            "switch_delay": float(oracle["switch_delay"]),
        },
        None,
    )


def _sample_from_policy(policy: list[float], legal: list[int], rng: random.Random) -> int:
    weights = [max(0.0, float(policy[card - 1])) for card in legal]
    total = sum(weights)
    if total <= 0.0:
        return rng.choice(legal)
    threshold = rng.random() * total
    accum = 0.0
    for card, weight in zip(legal, weights):
        accum += weight
        if accum >= threshold:
            return card
    return legal[-1]


def _play_policy_match(row_policy: Any, col_policy: Any, *, n_cards: int, seed: int) -> float:
    rng = random.Random(seed)
    state = GameState.initial(n_cards, current_prize=1)
    while not state.done:
        row_dist = row_policy.policy_for_state(state)
        col_dist = col_policy.policy_for_state(_mirrored_state(state))
        row_action = _sample_from_policy(row_dist, state.self_actions, rng)
        col_action = _sample_from_policy(col_dist, state.opponent_actions, rng)
        next_prize = legal_cards(state.prize_mask, state.n)[0] if state.prize_mask else None
        state = transition(state, row_action, col_action, next_prize=next_prize).state
    return float(state.self_score - state.opp_score)


def run_stage6_league(*, out_dir: str | Path) -> StageMetrics:
    out = Path(out_dir)
    registry = LeagueRegistry(out / "league" / "registry.json")
    for role in (ROLE_ROBUST, ROLE_AGGRESSIVE, ROLE_EXPLOITER):
        if not any(agent.role == role for agent in registry.agents.values()):
            registry.add(
                LeagueAgent(
                    agent_id=f"seed_initial_{role.lower()}",
                    role=role,
                    checkpoint_path=None,
                    policy_version=0,
                    metrics={"priority": 1.0},
                )
            )
    counts = registry.counts_by_role()
    agents = sorted(registry.agents.values(), key=lambda a: a.agent_id)
    cross_play = []
    from goofspiel.training.baseline_algorithms import create_baseline

    baseline_by_role = {
        ROLE_ROBUST: create_baseline("CFR+"),
        ROLE_AGGRESSIVE: create_baseline("PPO"),
        ROLE_EXPLOITER: create_baseline("Minimax-Q"),
    }
    for row_agent in agents:
        for col_agent in agents:
            score = _play_policy_match(
                baseline_by_role[row_agent.role],
                baseline_by_role[col_agent.role],
                n_cards=3,
                seed=len(cross_play) + 600,
            )
            cross_play.append(
                {
                    "row_agent": row_agent.agent_id,
                    "col_agent": col_agent.agent_id,
                    "row_role": row_agent.role,
                    "col_role": col_agent.role,
                    "mean_score_diff": score,
                    "games": 1,
                    "source": "simulated_crossplay",
                }
            )
    pfsp_weights = {
        agent.agent_id: max(
            0.01,
            1.0
            - abs(
                sum(row["mean_score_diff"] for row in cross_play if row["row_agent"] == agent.agent_id)
                / max(1, sum(1 for row in cross_play if row["row_agent"] == agent.agent_id))
            )
            / 91.0,
        )
        for agent in agents
    }
    league_report = {
        "counts_by_role": counts,
        "pfsp_weights": pfsp_weights,
        "cross_play": cross_play,
        "historical_agents_frozen": all(agent.frozen for agent in agents),
    }
    report_path = out / "league" / "league_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(league_report, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics = {f"{k.lower()}_count": float(v) for k, v in counts.items()}
    metrics.update(
        {
            "league_agents": float(len(agents)),
            "crossplay_pairs": float(len(cross_play)),
            "pfsp_weights": float(len(pfsp_weights)),
        }
    )
    return StageMetrics("P6_LEAGUE", 1, metrics, None)


def run_stage7_redteam(*, out_dir: str | Path) -> StageMetrics:
    out = Path(out_dir)
    failures = FailureBuffer(out / "redteam" / "failures.jsonl")
    corrections = CorrectionDataset(out / "redteam" / "corrections.jsonl")
    router = TeacherRouter()
    attack_states = [
        GameState.initial(3, current_prize=1),
        GameState.initial(3, current_prize=2),
        GameState(n=3, self_mask=0b011, opp_mask=0b110, prize_mask=0b100, current_prize=1, carry_pool=2, round_index=2),
    ]
    attack_report = []
    for idx, state in enumerate(attack_states):
        sample = router.label_state(state)
        failure = FailureRecord(
            failure_id=f"redteam_attack_{idx}_{uuid.uuid4().hex}",
            failure_type="ADVERSARIAL_STATE_REANALYSIS",
            state=state_record_from_game_state(state),
            model_version="seed_initial",
            teacher_source=sample.teacher_source,
            details={
                "purpose": "minimal red-team correction loop",
                "attack_family": "carry_and_asymmetric_masks",
                "teacher_confidence": sample.teacher_confidence,
            },
        )
        failures.add(failure)
        corrections.add_reanalysis(
            ReanalysisRecord(
                sample_id=f"correction_{idx}_{uuid.uuid4().hex}",
                original_sample_id=failure.failure_id,
                state=failure.state,
                new_teacher_source=sample.teacher_source,
                teacher_q=sample.teacher_q,
                teacher_policy=sample.teacher_policy,
                teacher_value=sample.teacher_value,
            )
        )
        attack_report.append(
            {
                "failure_id": failure.failure_id,
                "state_hash": failure.state.state_hash,
                "teacher_source": sample.teacher_source,
                "teacher_confidence": sample.teacher_confidence,
            }
        )
    report_path = out / "redteam" / "redteam_report.json"
    focused_report = {
        "training_plan": {
            "method": "focused_correction_sft",
            "steps": len(attack_report),
            "source": str(corrections.store.path),
            "freeze_public_encoder": False,
            "retain_general_replay_fraction": 0.25,
        },
        "regression": {
            # No regression is actually executed in P7 yet (Phase 4.4). Do not
            # fabricate pass/fail booleans: report null so a reader cannot mistake
            # an unrun check for a passed one.
            "original_attack_regression_passed": None,
            "general_regression_passed": None,
            "recurrence": None,
        },
    }
    focused_path = out / "redteam" / "focused_correction_report.json"
    focused_path.write_text(json.dumps(focused_report, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(
        json.dumps(
            {"attacks": attack_report, "corrections": corrections.count(), "focused_correction": str(focused_path)},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return StageMetrics(
        "P7_REDTEAM",
        1,
        {
            "failures": float(failures.count()),
            "corrections": float(corrections.count()),
            "attack_families": 1.0,
            "teacher_relabels": float(len(attack_report)),
            "focused_correction_steps": float(len(attack_report)),
            # Regression pass/fail is intentionally NOT reported here until a real
            # regression suite runs (Phase 4.4). Emitting 1.0 would be a fabricated
            # success literal; omission is the honest state.
        },
        None,
    )


def run_evaluation_suite(*, out_dir: str | Path, num_games: int = 16) -> dict[str, Any]:
    from goofspiel.training.benchmark import EvaluationProfile, run_unified_benchmark, write_benchmark_report

    report_a = evaluate_bot_matchup(num_games=num_games)
    report_b = exact_feasibility_sweep(13)
    path = Path(out_dir) / "evaluation_report.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "reports": [
            {"name": report_a.name, "metrics": report_a.metrics, "details": report_a.details, "passed": report_a.passed},
            {"name": report_b.name, "metrics": report_b.metrics, "details": report_b.details, "passed": report_b.passed},
        ]
    }
    path.write_text(__import__("json").dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    benchmark = run_unified_benchmark(EvaluationProfile(name="QUICK", num_games=num_games, include_e7=False))
    payload["benchmark_report"] = write_benchmark_report(benchmark, Path(out_dir) / "reports" / "quick")
    return payload


def _smoke_algorithmic_check(
    checkpoint_path: str | None,
    *,
    n_cards: int,
    seed: int,
    num_games: int = 32,
) -> dict[str, Any]:
    """Re-execute the Phase 0.1 honest evaluator on the produced checkpoint.

    Returns a dict whose ``algorithmic_ok`` is computed — never a literal — from
    the trained policy's mean score-diff versus Random through the real env. If no
    checkpoint was produced (e.g. non-rank0), the gate cannot pass and says so.
    """
    if not checkpoint_path or not Path(checkpoint_path).exists():
        return {"algorithmic_ok": False, "reason": "no_checkpoint", "checkpoint": checkpoint_path}
    try:
        from goofspiel.training.model_eval import (
            load_model_from_checkpoint,
            play_policy_vs_bot,
            robust_policy_fn,
        )

        model, _meta = load_model_from_checkpoint(checkpoint_path)
        policy = robust_policy_fn(model, greedy=True)
        vs_random = play_policy_vs_bot(policy, "random", n_cards=n_cards, num_games=num_games, seed=seed)
    except Exception as exc:  # pragma: no cover - defensive; surfaced honestly
        return {"algorithmic_ok": False, "reason": f"eval_error:{type(exc).__name__}:{exc}", "checkpoint": checkpoint_path}
    # Beat Random on the computed mean score-diff. A smoke-sized model is weak, so
    # the bar is deliberately the honest zero line (strictly positive), not a
    # cosmetic threshold. This number comes entirely from real play.
    ok = vs_random["mean_score_diff"] > 0.0
    return {
        "algorithmic_ok": bool(ok),
        "checkpoint": checkpoint_path,
        "vs_random": vs_random,
    }


def run_smoke_pipeline(
    *,
    out_dir: str | Path,
    steps: int,
    batch_size: int,
    device: str = "cpu",
    n_cards: int = 3,
    num_corpus_games: int = 4,
    seed: int = 1,
) -> dict[str, Any]:
    """Run the smallest end-to-end training flow that still writes real artifacts."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sink = JsonlEventSink(out / "events" / "training_smoke.jsonl")

    def emit(stage: str, payload: dict[str, Any]) -> None:
        sink.emit(BaseEvent(event_type="TRAINING_SMOKE_STAGE", run_id="smoke_pipeline", payload={"stage": stage, **payload}))

    emit("system_metrics_start", collect_system_metrics())
    stage0 = run_stage0_verify(artifact_dir=out / "stage0_verify")
    emit("stage0_verify", {"ok": stage0.ok, "checks": stage0.checks})
    corpus = generate_random_game_corpus(out_path=out / "data" / "game_corpus.jsonl", num_games=num_corpus_games, seed=seed)
    emit("build_corpus", {"ok": True, "metrics": corpus})
    stage1 = run_stage1_pretrain(steps=steps, batch_size=batch_size, out_dir=out / "checkpoints", device=device, n_cards=n_cards)
    emit("stage1_pretrain", {"ok": True, "metrics": stage1.metrics, "checkpoint": stage1.checkpoint})
    stage2 = run_stage2_semi_supervised(steps=steps, out_dir=out / "data", n_cards=n_cards)
    emit("stage2_semi_supervised", {"ok": True, "metrics": stage2.metrics})
    # Phase 3.1: chain θ forward. P3 inherits P1's weights (θ-only, fresh
    # optimizer); P4 inherits P3's. This is init_from_checkpoint, NOT resume.
    stage3 = run_stage3_sft(
        steps=steps, batch_size=batch_size, out_dir=out / "checkpoints", device=device, n_cards=n_cards,
        init_from_checkpoint=stage1.checkpoint,
    )
    emit("stage3_sft", {"ok": True, "metrics": stage3.metrics, "checkpoint": stage3.checkpoint})
    stage4 = run_stage4_robust_rl(
        steps=steps, batch_size=batch_size, out_dir=out / "checkpoints", device=device, n_cards=n_cards,
        init_from_checkpoint=stage3.checkpoint,
    )
    emit("stage4_robust_rl", {"ok": True, "metrics": stage4.metrics, "checkpoint": stage4.checkpoint})
    stage5 = run_stage5_adaptive(steps=steps, out_dir=out, n_cards=n_cards)
    emit("stage5_adaptive", {"ok": True, "metrics": stage5.metrics})
    stage6 = run_stage6_league(out_dir=out)
    emit("stage6_league", {"ok": True, "metrics": stage6.metrics})
    stage7 = run_stage7_redteam(out_dir=out)
    emit("stage7_redteam", {"ok": True, "metrics": stage7.metrics})
    evaluation = run_evaluation_suite(out_dir=out / "evaluation", num_games=max(2, num_corpus_games))
    emit("evaluate", {"ok": True, "metrics": evaluation})
    emit("system_metrics_end", collect_system_metrics())

    # Honest algorithmic gate (Phase 0.2): re-execute the Phase 0.1 evaluator on
    # the checkpoint the pipeline actually produced, rather than declaring the run
    # OK on stage-0's import/verify check alone. `stage0.ok` only says the modules
    # loaded; `algorithmic_ok` says the trained P4 policy beats Random on a real,
    # computed mean score-diff. The overall summary is OK only if BOTH hold.
    algorithmic = _smoke_algorithmic_check(stage4.checkpoint, n_cards=n_cards, seed=seed)
    emit("algorithmic_gate", algorithmic)

    summary = {
        "ok": bool(stage0.ok) and bool(algorithmic["algorithmic_ok"]),
        "stage0_ok": bool(stage0.ok),
        "algorithmic_ok": bool(algorithmic["algorithmic_ok"]),
        "algorithmic_gate": algorithmic,
        "device": device,
        "n_cards": int(n_cards),
        "steps": int(steps),
        "batch_size": int(batch_size),
        "num_corpus_games": int(num_corpus_games),
        "event_log": str(sink.path),
        "event_count": sink.count(),
        "stage0": {"ok": stage0.ok, "checks": stage0.checks, "errors": stage0.errors},
        "build_corpus": corpus,
        "stage1_pretrain": as_stage_dict(stage1),
        "stage2_semi_supervised": as_stage_dict(stage2),
        "stage3_sft": as_stage_dict(stage3),
        "stage4_robust_rl": as_stage_dict(stage4),
        "stage5_adaptive": as_stage_dict(stage5),
        "stage6_league": as_stage_dict(stage6),
        "stage7_redteam": as_stage_dict(stage7),
        "evaluation": evaluation,
    }
    summary_path = out / "training_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary


def as_stage_dict(metrics: StageMetrics) -> dict[str, Any]:
    return {
        "stage": metrics.stage,
        "steps": metrics.steps,
        "metrics": metrics.metrics,
        "checkpoint": metrics.checkpoint,
    }


def iter_declared_stages() -> list[str]:
    from goofspiel.training.distributed import STAGE_SEQUENCE

    return list(STAGE_SEQUENCE)
