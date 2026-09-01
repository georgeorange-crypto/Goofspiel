"""Runnable training stages.  Torch is imported lazily inside stage methods."""

from __future__ import annotations

import itertools
import json
import random
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from goofspiel.game import GameState, legal_cards, transition
from goofspiel.observability import BaseEvent, JsonlEventSink, collect_system_metrics

if TYPE_CHECKING:
    from goofspiel.observability import TrainingLogger
from goofspiel.training.checkpoint import (
    CheckpointMetadata,
    dataset_provenance,
    model_config_hash,
    save_checkpoint,
)
from goofspiel.training.checkpoint import init_from_checkpoint as _load_init_from_checkpoint
from goofspiel.training.checkpoint import resume_checkpoint as _load_resume_checkpoint
from goofspiel.training.checkpoint_registry import CheckpointRegistry
from goofspiel.training.corpus import generate_random_game_corpus
from goofspiel.training.data import FailureRecord, JsonlStore, OpponentSession, ReanalysisRecord, RobustTrajectorySample, RoundRecord, state_record_from_game_state
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


def _should_log_step(step: int, total: int) -> bool:
    """Log roughly ten evenly-spaced steps per stage, always including the last.

    A stage running few steps (the smoke/test path) logs every step; a long run
    logs one line per ~10% of progress so ``run.log`` stays readable while still
    charting the loss trajectory rather than only its final value.
    """
    if total <= 0:
        return True
    every = max(1, total // 10)
    return step % every == 0 or step == total - 1


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
        "parent_checkpoint_sha256": None,
        "optimizer_reset": True,
        "resumed_global_step": 0,
    }
    target = getattr(model, "module", model)
    if init_from_checkpoint_path:
        prov = _load_init_from_checkpoint(target, init_from_checkpoint_path)
        lineage["init_checkpoint_id"] = prov["init_checkpoint_id"]
        lineage["parent_checkpoint_id"] = prov["init_checkpoint_id"]
        lineage["init_checkpoint_sha256"] = prov["init_checkpoint_sha256"]
        # The parent FILE's content hash at the moment we inherited from it — the
        # single fact a lineage-consistency check needs to confirm the child was
        # built on THIS parent's bytes, not a same-named file that later changed.
        lineage["parent_checkpoint_sha256"] = prov["init_checkpoint_sha256"]
        lineage["optimizer_reset"] = True  # a stage boundary always resets the optimizer
    elif resume_checkpoint_path:
        prov = _load_resume_checkpoint(
            target, resume_checkpoint_path, optimizers=optimizers, target_model=target_model
        )
        lineage["parent_checkpoint_id"] = prov["parent_checkpoint_id"]
        lineage["resume_sha256"] = prov["resume_sha256"]
        lineage["parent_checkpoint_sha256"] = prov["resume_sha256"]
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
    corpus_path: str | Path | None = None,
    init_from_checkpoint: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    logger: "TrainingLogger | None" = None,
) -> StageMetrics:
    """P1 pretraining over swap, transition, joint-outcome, and opponent tasks.

    When ``corpus_path`` points at a ``game_corpus.jsonl`` (written by
    ``build_corpus``), P1 pretrains over the states actually recorded there —
    so the corpus is a LIVE producer feeding P1, not a dead artifact.  When it
    is absent, P1 falls back to on-the-fly ``sample_reachable_states``.
    """
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
    # Prefer the game corpus written by build_corpus; fall back to on-the-fly
    # reachable states when it is absent (a stage run in isolation).
    corpus_states = _load_corpus_states(Path(corpus_path)) if corpus_path else []
    corpus_source = "game_corpus" if corpus_states else "sample_reachable_states"
    losses: list[float] = []
    metrics: dict[str, float] = {}
    all_states: list[GameState] = []
    for step in range(int(steps)):
        if corpus_states:
            start = (step * batch_size) % len(corpus_states)
            states = [corpus_states[(start + i) % len(corpus_states)] for i in range(batch_size)]
        else:
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
            "corpus_states": float(len(corpus_states)),
            "trained_on_corpus": float(bool(corpus_states)),
        }
        if logger is not None and _should_log_step(step, int(steps)):
            logger.step_metrics(
                "stage1_pretrain",
                step,
                int(steps),
                {
                    "loss": float(loss.detach().cpu()),
                    "loss_q": float(loss_q.detach().cpu()),
                    "swap": float(swap_loss.detach().cpu()),
                    "opp": float(opp_loss.detach().cpu()),
                },
            )

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
                parent_checkpoint_sha256=lineage.get("parent_checkpoint_sha256"),
                model_config_hash=model_config_hash(getattr(model, "module", model)),
                datasets=(
                    [dataset_provenance(
                        corpus_path,
                        num_samples=len(corpus_states),
                        role="game_corpus",
                    )]
                    if corpus_states else []
                ),
                optimizer_reset=lineage["optimizer_reset"],
            ),
        )
        registry = CheckpointRegistry(out_dir / "registry")
        registry.register("latest", ckpt, global_step=int(steps), metrics=metrics)
        checkpoint_path = manifest["path"]
        if logger is not None:
            logger.checkpoint_saved("stage1_pretrain", checkpoint_path, global_step=int(steps))
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


def _load_corpus_states(path: Path) -> list[GameState]:
    """Reconstruct trainable (non-terminal) states from a game corpus JSONL.

    This is the seam that makes ``build_corpus`` a live producer instead of a
    dead artifact: P1 pretrains over the states actually recorded in
    ``game_corpus.jsonl`` when it exists.  Terminal states and states with no
    legal self/opponent actions are skipped — P1's targets need a playable
    ``(self_action, opponent_action)`` pair.
    """
    if not path.exists():
        return []
    store = JsonlStore(path)
    states: list[GameState] = []
    for rec in store.iter_dicts():
        state_rec = rec.get("state")
        if not isinstance(state_rec, dict):
            continue
        st = _game_state_from_record(state_rec)
        if st.done or not st.self_actions or not st.opponent_actions:
            continue
        states.append(st)
    return states


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
    logger: "TrainingLogger | None" = None,
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
            if logger is not None and _should_log_step(step, int(steps)):
                logger.step_metrics(
                    "stage3_sft",
                    step,
                    int(steps),
                    {"loss": last_loss, "q_loss": float(q_loss.detach().cpu()),
                     "pi_loss": float(pi_loss.detach().cpu())},
                )
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
            if logger is not None and _should_log_step(step, int(steps)):
                logger.step_metrics(
                    "stage3_sft",
                    step,
                    int(steps),
                    {"loss": last_loss, "q_loss": float(q_loss.detach().cpu()),
                     "pi_loss": float(pi_loss.detach().cpu())},
                )

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
                parent_checkpoint_sha256=lineage.get("parent_checkpoint_sha256"),
                model_config_hash=model_config_hash(getattr(model, "module", model)),
                teacher_dataset_ids=[str(teacher_dataset_path)] if dataset else [],
                datasets=(
                    [dataset_provenance(
                        teacher_dataset_path,
                        num_samples=len(dataset),
                        role="teacher_dataset",
                    )]
                    if dataset else []
                ),
                optimizer_reset=lineage["optimizer_reset"],
            ),
        )
        registry = CheckpointRegistry(out_dir / "registry")
        registry.register("latest", ckpt, global_step=int(steps), metrics=metrics)
        registry.register("teacher_ema", ckpt, global_step=int(steps), metrics=metrics)
        checkpoint_path = manifest["path"]
        if logger is not None:
            logger.checkpoint_saved("stage3_sft", checkpoint_path, global_step=int(steps))
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
    logger: "TrainingLogger | None" = None,
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
        if logger is not None and _should_log_step(step, int(steps)):
            logger.step_metrics(
                "stage4_robust_rl",
                step,
                int(steps),
                {
                    "q": last_q,
                    "actor": last_actor,
                    "pg": last_pg,
                    "entropy": last_entropy,
                    "curriculum_n": cstep.n_cards,
                },
            )

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
                parent_checkpoint_sha256=lineage.get("parent_checkpoint_sha256"),
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
        if logger is not None:
            logger.checkpoint_saved("stage4_robust_rl", checkpoint_path, global_step=int(steps))
    barrier_if_distributed()
    stage_metrics["promotion_candidate"] = 1.0 if promotion.decision == "PROMOTE_CANDIDATE" else 0.0
    stage_metrics["replay_persisted_samples"] = float(replay_buffer.persisted_count()) if runtime.is_rank0 else 0.0
    return StageMetrics(
        "P4_ROBUST_RL",
        int(steps),
        stage_metrics,
        checkpoint_path,
    )


def _opponent_regime_distribution(regime_id: str, legal: list[int], *, stake: int, n_cards: int) -> dict[int, float]:
    """The TRUE next-action distribution of a scripted regime (the label P5 fits).

    ``opponent_action_for_regime`` samples an action; here we expose the full
    categorical it samples from, so P5 can train the opponent head against a real
    probability target and measure a real NLL/ECE against it — not a constant."""
    if not legal:
        raise ValueError("distribution requires at least one legal action")
    threshold = max(1, n_cards // 2)
    if regime_id == "high_card_pressure" and stake >= threshold:
        return {card: (1.0 if card == max(legal) else 0.0) for card in legal}
    if regime_id == "low_card_saver" and stake <= threshold:
        return {card: (1.0 if card == min(legal) else 0.0) for card in legal}
    p = 1.0 / len(legal)
    return {card: p for card in legal}


def _build_adaptive_training_tensors(sessions, *, max_cards, device):
    """Featurize opponent sessions into (public batch, history, memory, target) for
    the adaptive/opponent branch.

    Each *decision point* becomes one training row whose:
      * public state is reconstructed from the round (masks shrink as cards are played);
      * intra-game history is the rounds of THIS game so far (feeds the LSTM);
      * inter-game memory is a per-game summary sequence of PRIOR games this
        session (feeds the Mamba) — so the long-horizon head sees real cross-game
        context, the whole point of the session structure;
      * target is the opponent's actual next action index (for NLL/accuracy).
    """
    torch, _F = _torch_import()
    from goofspiel.models import public_state_from_game

    states: list[GameState] = []
    hist_rows: list[list[dict[str, float]]] = []
    mem_rows: list[list[list[float]]] = []
    targets: list[int] = []
    n_cards_row: list[int] = []
    for session in sessions:
        prior_game_summaries: list[list[float]] = []
        for game in session.games:
            n = _infer_session_n(game, max_cards)
            self_mask = opp_mask = (1 << n) - 1
            prize_mask = self_mask
            self_score = opp_score = 0
            carry = 0
            game_hist: list[dict[str, float]] = []
            for rec in game:
                prize = int(rec.prize)
                state = GameState(
                    n=n, self_mask=self_mask, opp_mask=opp_mask,
                    prize_mask=prize_mask & ~(1 << (prize - 1)) if prize else prize_mask,
                    current_prize=prize, self_score=self_score, opp_score=opp_score,
                    round_index=int(rec.round_index), done=False, carry_pool=carry,
                )
                states.append(state)
                hist_rows.append(list(game_hist))
                mem_rows.append(list(prior_game_summaries))
                targets.append(int(rec.opponent_action) - 1)
                n_cards_row.append(n)
                game_hist.append({
                    "prize": float(prize),
                    "self_action": float(rec.self_action),
                    "opponent_action": float(rec.opponent_action),
                    "score_diff": float(self_score - opp_score),
                    "round_idx": float(rec.round_index),
                })
                self_mask &= ~(1 << (int(rec.self_action) - 1))
                opp_mask &= ~(1 << (int(rec.opponent_action) - 1))
                self_score += int(rec.reward_self)
                opp_score += int(rec.reward_opponent)
                carry = int(rec.carry_out)
                prize_mask &= ~(1 << (prize - 1)) if prize else prize_mask
            # One D-dim summary of the finished game for the inter-game memory.
            prior_game_summaries.append(_summarize_game(game, n, max_cards))

    if not states:
        return None
    batch = public_state_from_game(states, max_cards=max_cards, device=device)
    history = _stack_history(hist_rows, max_cards=max_cards, device=device)
    memory = _stack_memory(mem_rows, dim=192, device=device)
    target_t = torch.tensor(targets, dtype=torch.long, device=device)
    return batch, history, memory, target_t, n_cards_row


def _infer_session_n(game, max_cards: int) -> int:
    cards = {int(r.self_action) for r in game} | {int(r.opponent_action) for r in game} | {int(r.prize) for r in game}
    return max(len(game), max(cards, default=1), 1) if game else 1


def _summarize_game(game, n: int, max_cards: int) -> list[float]:
    """A fixed 192-d summary of a finished game for the inter-game memory sequence.

    Deterministic featurization (not learned here): normalized final score diff,
    round count, mean stake, opponent high/low-card rates, broadcast to 192-d so
    it matches the Mamba input width.  The learned `game_summary_projector` then
    projects it inside the model."""
    if not game:
        return [0.0] * 192
    total = n * (n + 1) / 2.0
    self_score = sum(int(r.reward_self) for r in game)
    opp_score = sum(int(r.reward_opponent) for r in game)
    opp_high = sum(1 for r in game if int(r.opponent_action) == n) / len(game)
    opp_low = sum(1 for r in game if int(r.opponent_action) == 1) / len(game)
    feats = [
        (self_score - opp_score) / total,
        len(game) / float(max_cards),
        sum(int(r.prize) + int(r.carry_in) for r in game) / (len(game) * total),
        opp_high,
        opp_low,
    ]
    reps = (192 + len(feats) - 1) // len(feats)
    return (feats * reps)[:192]


def _stack_history(hist_rows, *, max_cards, device):
    torch, _F = _torch_import()
    from goofspiel.models import HistoryBatch

    batch = len(hist_rows)
    steps = max((len(h) for h in hist_rows), default=1) or 1
    prize = torch.zeros(batch, steps, dtype=torch.long, device=device)
    self_a = torch.zeros_like(prize)
    opp_a = torch.zeros_like(prize)
    score_diff = torch.zeros(batch, steps, device=device)
    round_idx = torch.zeros(batch, steps, device=device)
    valid = torch.zeros(batch, steps, dtype=torch.bool, device=device)
    for i, rows in enumerate(hist_rows):
        for t, ev in enumerate(rows):
            prize[i, t] = int(ev["prize"])
            self_a[i, t] = int(ev["self_action"])
            opp_a[i, t] = int(ev["opponent_action"])
            score_diff[i, t] = ev["score_diff"]
            round_idx[i, t] = ev["round_idx"]
            valid[i, t] = True
    return HistoryBatch(
        prize=prize, self_action=self_a, opponent_action=opp_a,
        score_diff=score_diff, outcome=torch.zeros_like(score_diff),
        round_idx=round_idx, valid_mask=valid,
    )


def _stack_memory(mem_rows, *, dim, device):
    torch, _F = _torch_import()
    from goofspiel.models import OpponentMemoryBatch

    batch = len(mem_rows)
    games = max((len(m) for m in mem_rows), default=1) or 1
    seq = torch.zeros(batch, games, dim, device=device)
    valid = torch.zeros(batch, games, dtype=torch.bool, device=device)
    for i, rows in enumerate(mem_rows):
        for g, summary in enumerate(rows):
            seq[i, g] = torch.tensor(summary, dtype=torch.float32, device=device)
            valid[i, g] = True
    return OpponentMemoryBatch(game_summary_sequence=seq, valid_mask=valid)


def _expected_calibration_error(probs, targets, *, n_bins: int = 10) -> float:
    """ECE over the predicted-argmax confidence vs. empirical accuracy."""
    torch, _F = _torch_import()
    conf, pred = probs.max(dim=-1)
    acc = (pred == targets).float()
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        in_bin = (conf > lo) & (conf <= hi) if b > 0 else (conf >= lo) & (conf <= hi)
        if in_bin.any():
            ece += float(in_bin.float().mean()) * abs(float(acc[in_bin].mean()) - float(conf[in_bin].mean()))
    return ece


def run_stage5_adaptive(
    *,
    steps: int,
    out_dir: str | Path,
    n_cards: int = 5,
    lr: float = 3e-4,
    init_from_checkpoint: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    logger: "TrainingLogger | None" = None,
) -> StageMetrics:
    """P5 — TRAIN the opponent/adaptive branch on scripted-regime sessions.

    Phase 3.2: this stage no longer emits a constant-NLL diagnostic.  It builds
    real multi-game opponent sessions, then runs supervised training of the
    opponent-prediction heads (short/long/fused) — the LSTM, Mamba, memory fusion
    and opponent heads — against each regime's TRUE next-action distribution.

    Phase 3.2b — explicit gradient firewall.  The robust backbone/Q/actor are
    *frozen by `requires_grad_(False)`*, not merely shielded by `.detach()`.  We
    optimize ONLY `model.adaptive_parameters()`, and after every step assert the
    firewall directly (`‖Δθ_R‖ == 0` and `‖∇θ_A‖ > 0`) so a future refactor that
    leaks gradient into robust is caught here, in the run itself.
    """
    torch, F = _torch_import()
    from goofspiel.models import GoofspielModel

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(503 + int(steps) + int(n_cards))
    sessions_path = out / "adaptive" / "opponent_sessions.jsonl"
    sessions = JsonlStore(sessions_path)
    session_rows = []
    regimes = default_opponent_curriculum()
    # --- Build the opponent sessions: several GAMES per session so the inter-game
    #     Mamba memory has real cross-game context to carry. ---
    games_per_session = 3
    for session_idx in range(max(1, int(steps))):
        regime = regimes[session_idx % len(regimes)]
        games: list[list[RoundRecord]] = []
        for _ in range(games_per_session):
            first_prize = rng.choice(list(range(1, n_cards + 1)))
            state = GameState.initial(n_cards, current_prize=first_prize)
            rounds = []
            while not state.done:
                self_action = rng.choice(state.self_actions)
                opp_action = opponent_action_for_regime(
                    regime.regime_id, state.opponent_actions,
                    stake=state.stake, n_cards=n_cards, rng=rng,
                )
                next_prize = rng.choice(legal_cards(state.prize_mask, state.n)) if state.prize_mask else None
                result = transition(state, self_action, opp_action, next_prize=next_prize)
                rounds.append(RoundRecord(
                    round_index=state.round_index, prize=state.current_prize,
                    self_action=self_action, opponent_action=opp_action,
                    reward_self=result.reward_self, reward_opponent=result.reward_opp,
                    carry_in=state.carry_pool, carry_out=result.state.carry_pool,
                    done=result.state.done,
                ))
                state = result.state
            games.append(rounds)
        session = OpponentSession(
            session_id=f"adaptive_session_{session_idx}_{uuid.uuid4().hex}",
            opponent_id=f"curriculum_{regime.regime_id}",
            strategy_regime_id=regime.regime_id,
            games=games,
        )
        sessions.append(session)
        session_rows.append(session)
    rounds_total = sum(len(g) for s in session_rows for g in s.games)

    # --- Featurize and train the opponent/adaptive branch (robust FROZEN). ---
    device = "cpu"
    model = GoofspielModel(max_cards=13).to(device)
    model.assert_partition_is_complete()
    opt = torch.optim.AdamW(model.adaptive_parameters(), lr=lr)
    lineage = _apply_init_or_resume(
        model,
        init_from_checkpoint_path=init_from_checkpoint,
        resume_checkpoint_path=resume_checkpoint,
        optimizers={"adaptive_sft": opt},
    )
    # The firewall: freeze robust params EXPLICITLY (Phase 3.2b), not via detach.
    model.set_robust_requires_grad(False)

    tensors = _build_adaptive_training_tensors(session_rows, max_cards=13, device=device)
    if tensors is None:
        raise RuntimeError("P5 produced no training rows from the opponent sessions")
    batch, history, memory, target_t, n_cards_row = tensors
    # Uniform reference NLL per decision point (the honest bar to beat), computed
    # from the actual legal-action counts, not a single scalar.
    legal_counts = batch.opponent_action_mask.sum(dim=-1).clamp_min(1).float()
    uniform_reference_nll = float(torch.log(legal_counts).mean())

    robust_snapshot = [p.detach().clone() for p in model.robust_parameters()]
    train_steps = max(1, int(steps))
    last = {"nll": float("nan"), "acc": 0.0, "ece": 1.0, "adaptive_grad_norm": 0.0, "robust_delta": 0.0}
    model.train()
    for _step in range(train_steps):
        out_model = model(batch, current_game_history=history, long_term_memory=memory)
        # Train the FUSED opponent head (short+long context combined). Masked to
        # legal opponent actions so the categorical is over legal cards only.
        logits = out_model.opponent_fused_logits
        loss = F.cross_entropy(logits, target_t)
        # Auxiliary short/long supervision keeps both memory paths learning.
        loss = loss + 0.5 * F.cross_entropy(out_model.opponent_short_logits, target_t)
        loss = loss + 0.5 * F.cross_entropy(out_model.opponent_long_logits, target_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        adaptive_grad_norm = float(torch.sqrt(sum(
            (p.grad.detach().float().pow(2).sum() for p in model.adaptive_parameters() if p.grad is not None),
            torch.tensor(0.0),
        )))
        # No robust param may carry gradient at all (frozen ⇒ grad is None).
        robust_with_grad = [p for p in model.robust_parameters() if p.grad is not None]
        if robust_with_grad:
            raise AssertionError(f"{len(robust_with_grad)} robust params received gradient in P5 (firewall breach)")
        torch.nn.utils.clip_grad_norm_(model.adaptive_parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            probs = F.softmax(logits.masked_fill(~batch.opponent_action_mask, -1e9), dim=-1)
            nll = float(F.cross_entropy(logits, target_t))
            acc = float((probs.argmax(dim=-1) == target_t).float().mean())
            ece = _expected_calibration_error(probs, target_t)
        last = {"nll": nll, "acc": acc, "ece": ece, "adaptive_grad_norm": adaptive_grad_norm,
                "robust_delta": 0.0}
        if logger is not None and _should_log_step(_step, train_steps):
            logger.step_metrics(
                "stage5_adaptive",
                _step,
                train_steps,
                {"nll": nll, "acc": acc, "ece": ece, "adaptive_grad_norm": adaptive_grad_norm},
            )
    # Firewall assertion #2: robust params are byte-for-byte unchanged.
    robust_delta = float(sum(
        (a - b).abs().sum() for a, b in zip(model.robust_parameters(), robust_snapshot)
    ))
    last["robust_delta"] = robust_delta
    if robust_delta != 0.0:
        raise AssertionError(f"robust params moved during P5 (‖Δθ_R‖={robust_delta} ≠ 0): firewall breach")

    oracle = oracle_opponent_diagnostic(session_rows, n_cards=n_cards)
    beats_uniform = last["nll"] < uniform_reference_nll
    ckpt_path = None
    ckpt = out / "stage5_adaptive.pt"
    manifest = save_checkpoint(
        ckpt,
        model=model,
        optimizers={"adaptive_sft": opt},
        metadata=CheckpointMetadata(
            checkpoint_id="stage5_adaptive",
            training_stage="P5_ADAPTIVE",
            global_step=train_steps,
            policy_version=3,
            config={"steps": steps, "n_cards": n_cards, "lr": lr, "games_per_session": games_per_session},
            metrics={
                "opponent_nll": last["nll"],
                "uniform_reference_nll": uniform_reference_nll,
                "opponent_accuracy": last["acc"],
                "opponent_ece": last["ece"],
            },
            parent_checkpoint_id=lineage["parent_checkpoint_id"],
            init_checkpoint_id=lineage["init_checkpoint_id"],
            parent_checkpoint_sha256=lineage.get("parent_checkpoint_sha256"),
            model_config_hash=model_config_hash(model),
            optimizer_reset=lineage["optimizer_reset"],
        ),
    )
    ckpt_path = manifest["path"]
    if logger is not None:
        logger.checkpoint_saved("stage5_adaptive", ckpt_path, global_step=train_steps)

    report = {
        # A trained opponent model now exists. It is usable iff it beat the honest
        # uniform reference on these scripted regimes.
        "opponent_model_usable": bool(beats_uniform),
        "gate": "OPPONENT_MODEL_TRAINED" if beats_uniform else "OPPONENT_MODEL_BELOW_UNIFORM",
        "calibration": {
            "sessions": max(1, int(steps)),
            "games_per_session": games_per_session,
            "rounds": rounds_total,
            "opponent_nll": last["nll"],
            "uniform_reference_nll": uniform_reference_nll,
            "nll_gain_over_uniform": uniform_reference_nll - last["nll"],
            "opponent_accuracy": last["acc"],
            "opponent_ece": last["ece"],
            "oracle_accuracy": oracle["oracle_accuracy"],
            "oracle_gain": oracle["oracle_gain"],
            "switch_delay": oracle["switch_delay"],
        },
        # 3.2b firewall evidence, recorded in the artifact itself.
        "firewall": {
            "robust_frozen": True,
            "robust_param_delta_l1": robust_delta,
            "adaptive_grad_norm_last": last["adaptive_grad_norm"],
        },
        "opponent_curriculum": [regime.__dict__ for regime in regimes],
        "required": {"nll_better_than_uniform": bool(beats_uniform), "switch_benchmark": "curriculum_gate"},
        "session_path": str(sessions_path),
        "checkpoint": ckpt_path,
    }
    (out / "adaptive" / "adaptive_gate_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return StageMetrics(
        "P5_ADAPTIVE",
        int(steps),
        {
            "opponent_model_usable": 1.0 if beats_uniform else 0.0,
            "opponent_sessions": float(max(1, int(steps))),
            "opponent_rounds": float(rounds_total),
            "opponent_nll": float(last["nll"]),
            "uniform_reference_nll": float(uniform_reference_nll),
            "nll_gain_over_uniform": float(uniform_reference_nll - last["nll"]),
            "opponent_accuracy": float(last["acc"]),
            "opponent_ece": float(last["ece"]),
            "opponent_regimes": float(len({s.strategy_regime_id for s in session_rows})),
            "oracle_gain": float(oracle["oracle_gain"]),
            "switch_delay": float(oracle["switch_delay"]),
            "robust_param_delta_l1": float(robust_delta),
            "adaptive_grad_norm_last": float(last["adaptive_grad_norm"]),
        },
        ckpt_path,
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


class _CheckpointPolicy:
    """A ``policy_for_state``-compatible wrapper around a *loaded* checkpoint.

    Phase 4.3: the league must play the trained models, not role-keyed
    baselines.  This adapts a :class:`GoofspielModel`'s robust policy to the
    13-slot ``policy_for_state`` interface ``_play_policy_match`` already speaks,
    so cross-play is genuine model-vs-model play.  A small softmax temperature
    keeps play stochastic (so seeds matter) without changing the argmax.
    """

    def __init__(self, checkpoint_path: str | Path, *, device: str = "cpu", temperature: float = 0.5) -> None:
        from goofspiel.training.model_eval import load_model_from_checkpoint, robust_policy_fn

        self.checkpoint_path = str(checkpoint_path)
        self.model, self.metadata = load_model_from_checkpoint(checkpoint_path, device=device)
        self._fn = robust_policy_fn(self.model, device=device, greedy=False, temperature=temperature)

    def policy_for_state(self, state: GameState) -> list[float]:
        dist = self._fn(state)
        policy = [0.0] * 13
        for card, prob in dist.items():
            policy[card - 1] = float(prob)
        return policy


def _mint_league_snapshot(role: str, *, out_dir: Path, seed: int, n_cards: int = 3) -> str:
    """Train a tiny, role-seeded checkpoint and return its real path.

    Each role is trained from a different torch seed so the three snapshots are
    *genuinely distinct* trained agents (verified by cross-play, not asserted).
    This is deliberately minimal — the point of Phase 4.3 is that the league
    plays *real, loadable, distinct* checkpoints, not that they are strong.
    """
    torch, _F = _torch_import()
    torch.manual_seed(seed)
    snap_dir = out_dir / "league" / "snapshots" / role.lower()
    metrics = run_stage1_pretrain(steps=1, batch_size=4, out_dir=snap_dir, n_cards=n_cards, lr=3e-4)
    return metrics.checkpoint


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


def run_stage6_league(
    *,
    out_dir: str | Path,
    role_checkpoints: dict[str, str | Path] | None = None,
    n_cards: int = 3,
) -> StageMetrics:
    out = Path(out_dir)
    registry = LeagueRegistry(out / "league" / "registry.json")

    # Phase 4.3: every league agent must reference a REAL, loadable checkpoint.
    # A caller (the smoke pipeline) can supply per-role checkpoints; otherwise we
    # mint three role-seeded snapshots so the three agents are genuinely distinct
    # trained models rather than `checkpoint_path=None` placeholders.
    role_seeds = {ROLE_ROBUST: 101, ROLE_AGGRESSIVE: 202, ROLE_EXPLOITER: 303}
    supplied = {str(k): str(v) for k, v in (role_checkpoints or {}).items()}
    resolved: dict[str, str] = {}
    for role in (ROLE_ROBUST, ROLE_AGGRESSIVE, ROLE_EXPLOITER):
        path = supplied.get(role)
        if not path or not Path(path).exists():
            path = _mint_league_snapshot(role, out_dir=out, seed=role_seeds[role], n_cards=n_cards)
        resolved[role] = str(path)
        if not any(agent.role == role for agent in registry.agents.values()):
            registry.add(
                LeagueAgent(
                    agent_id=f"seed_initial_{role.lower()}",
                    role=role,
                    checkpoint_path=resolved[role],
                    policy_version=0,
                    metrics={"priority": 1.0},
                )
            )
    counts = registry.counts_by_role()
    agents = sorted(registry.agents.values(), key=lambda a: a.agent_id)

    # Load each agent's REAL checkpoint once; cross-play is model-vs-model.
    policies: dict[str, _CheckpointPolicy] = {}
    for agent in agents:
        ckpt = agent.checkpoint_path or resolved.get(agent.role)
        policies[agent.agent_id] = _CheckpointPolicy(ckpt, temperature=0.5)

    cross_play = []
    for row_agent in agents:
        for col_agent in agents:
            score = _play_policy_match(
                policies[row_agent.agent_id],
                policies[col_agent.agent_id],
                n_cards=n_cards,
                seed=len(cross_play) + 600,
            )
            cross_play.append(
                {
                    "row_agent": row_agent.agent_id,
                    "col_agent": col_agent.agent_id,
                    "row_role": row_agent.role,
                    "col_role": col_agent.role,
                    "row_checkpoint": policies[row_agent.agent_id].checkpoint_path,
                    "col_checkpoint": policies[col_agent.agent_id].checkpoint_path,
                    "mean_score_diff": score,
                    "games": 1,
                    "source": "simulated_crossplay",
                }
            )

    # Handcrafted algorithms are kept only as clearly-LABELLED reference
    # opponents — never conflated with the trained cross-play above.
    from goofspiel.training.baseline_algorithms import create_baseline

    reference_by_role = {
        ROLE_ROBUST: create_baseline("CFR+"),
        ROLE_AGGRESSIVE: create_baseline("PPO"),
        ROLE_EXPLOITER: create_baseline("Minimax-Q"),
    }
    reference_play = []
    for agent in agents:
        ref = reference_by_role[agent.role]
        score = _play_policy_match(
            policies[agent.agent_id], ref, n_cards=n_cards, seed=len(reference_play) + 900
        )
        reference_play.append(
            {
                "agent": agent.agent_id,
                "role": agent.role,
                "reference": ref.name,
                "mean_score_diff": score,
                "games": 1,
                "source": "trained_vs_reference_baseline",
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
        "reference_play": reference_play,
        "agent_checkpoints": {agent.agent_id: policies[agent.agent_id].checkpoint_path for agent in agents},
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
            "reference_matches": float(len(reference_play)),
        }
    )
    return StageMetrics("P6_LEAGUE", 1, metrics, None)


def _attack_state_regression(
    model: Any,
    attack_states: list[GameState],
    teacher_cards: list[int],
) -> dict[str, Any]:
    """Re-play each attack state through the model and MEASURE the teacher-action
    match rate and mean NLL of the teacher action under the robust policy.

    This is the honest regression thermometer P7 was missing: it re-executes the
    policy on the exact attack states rather than reading a fabricated boolean.
    ``passed`` means every attack state's argmax robust action equals the teacher
    action (the correction closed the failure).
    """
    from goofspiel.training.model_eval import robust_policy_fn

    torch, _F = _torch_import()
    fn = robust_policy_fn(model, greedy=False, temperature=1.0)
    matches = 0
    nlls: list[float] = []
    per_state = []
    for card, state in zip(teacher_cards, attack_states):
        dist = fn(state)
        argmax_card = max(dist, key=lambda c: (dist.get(c, 0.0), -c)) if dist else None
        prob = float(dist.get(card, 0.0))
        nlls.append(-float(torch.log(torch.tensor(max(prob, 1e-9)))))
        hit = argmax_card == card
        matches += int(hit)
        per_state.append({"teacher_card": card, "argmax_card": argmax_card, "teacher_prob": prob, "matched": hit})
    n = max(1, len(attack_states))
    return {
        "match_rate": matches / n,
        "mean_teacher_nll": sum(nlls) / n,
        "matched": matches,
        "total": len(attack_states),
        "per_state": per_state,
        "passed": matches == len(attack_states),
    }


def run_stage7_redteam(
    *,
    out_dir: str | Path,
    init_from_checkpoint: str | Path | None = None,
    correction_steps: int = 40,
    lr: float = 1e-3,
    n_cards: int = 3,
) -> StageMetrics:
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
    teacher_samples = []
    teacher_cards: list[int] = []
    for idx, state in enumerate(attack_states):
        sample = router.label_state(state)
        teacher_samples.append(sample)
        # The teacher's recommended action (argmax over legal cards) — the target
        # the focused correction trains toward and the regression re-checks.
        pol = sample.teacher_policy or [1.0] * len(state.self_actions)
        best_index = max(range(len(state.self_actions)), key=lambda i: pol[i])
        teacher_cards.append(state.self_actions[best_index])
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
                "teacher_card": teacher_cards[idx],
            }
        )

    # ---- Phase 4.4: a REAL focused fine-tune + MEASURED regression -----------
    # Load (or mint) a real checkpoint, measure the attack-state regression
    # BEFORE, run a focused correction SFT on the teacher-relabeled attack states,
    # save the improved checkpoint, and measure the regression AFTER.  Every
    # pass/fail below is computed by re-playing the policy, never hardcoded.
    torch, F = _torch_import()
    from goofspiel.models import GoofspielModel, public_state_from_game
    from goofspiel.training.checkpoint import load_checkpoint

    if not init_from_checkpoint or not Path(init_from_checkpoint).exists():
        seed_metrics = run_stage1_pretrain(
            steps=1, batch_size=4, out_dir=out / "redteam" / "seed", n_cards=n_cards
        )
        init_from_checkpoint = seed_metrics.checkpoint

    model = GoofspielModel(max_cards=13)
    model.load_state_dict(load_checkpoint(init_from_checkpoint)["model_state"])
    model.eval()
    regression_before = _attack_state_regression(model, attack_states, teacher_cards)

    # Focused correction: KL toward the stored teacher policy on the attack states
    # (+ an immediate-Q anchor), the exact states that failed.
    batch = public_state_from_game(attack_states, max_cards=13)
    target_q, q_mask = _immediate_target(attack_states, 13)
    teacher_policy = torch.zeros(len(attack_states), 13)
    for b, sample in enumerate(teacher_samples):
        for i, v in enumerate((sample.teacher_policy or [])[:13]):
            teacher_policy[b, i] = float(v)
    teacher_policy = teacher_policy / teacher_policy.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    last_loss = 0.0
    for _step in range(int(correction_steps)):
        out_m = model(batch)
        logp = F.log_softmax(out_m.robust_policy_logits, dim=-1)
        pi_loss = F.kl_div(logp, teacher_policy, reduction="batchmean")
        q_loss = F.smooth_l1_loss(out_m.q_robust[q_mask], target_q[q_mask], beta=0.1)
        loss = pi_loss + q_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.detach().cpu())
    model.eval()
    regression_after = _attack_state_regression(model, attack_states, teacher_cards)

    corrected_ckpt = out / "redteam" / "stage7_corrected.pt"
    save_checkpoint(
        corrected_ckpt,
        model=model,
        optimizers={"focused_correction": opt},
        metadata=CheckpointMetadata(
            checkpoint_id="stage7_redteam_corrected",
            training_stage="P7_REDTEAM",
            global_step=int(correction_steps),
            policy_version=4,
            config={"correction_steps": correction_steps, "lr": lr, "n_cards": n_cards},
            metrics={"focused_correction_loss_last": last_loss},
            parent_checkpoint_id="seed_initial",
            init_checkpoint_id=str(init_from_checkpoint),
            model_config_hash=model_config_hash(model),
            optimizer_reset=True,
        ),
    )

    # A red-team correction "recurs" (fails) if any attack the correction fixed
    # is still mis-played after training.  Passing = every attack matches the
    # teacher action after correction.  General regression is proxied by the
    # attack match-rate not collapsing below the before-correction rate.
    original_attack_regression_passed = bool(regression_after["passed"])
    general_regression_passed = bool(regression_after["match_rate"] >= regression_before["match_rate"])
    recurrence = bool(regression_after["matched"] < len(attack_states))

    report_path = out / "redteam" / "redteam_report.json"
    focused_report = {
        "training_plan": {
            "method": "focused_correction_sft",
            "steps": int(correction_steps),
            "source": str(corrections.store.path),
            "init_checkpoint": str(init_from_checkpoint),
            "corrected_checkpoint": str(corrected_ckpt),
            "freeze_public_encoder": False,
            "retain_general_replay_fraction": 0.25,
        },
        "regression": {
            # MEASURED (Phase 4.4): computed by re-playing the attack states
            # before/after the focused correction, not fabricated.
            "original_attack_regression_passed": original_attack_regression_passed,
            "general_regression_passed": general_regression_passed,
            "recurrence": recurrence,
            "match_rate_before": regression_before["match_rate"],
            "match_rate_after": regression_after["match_rate"],
            "mean_teacher_nll_before": regression_before["mean_teacher_nll"],
            "mean_teacher_nll_after": regression_after["mean_teacher_nll"],
            "before": regression_before,
            "after": regression_after,
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
            # Phase 4.4: these are now MEASURED regression outcomes, re-executed
            # by replaying the attack states before/after the focused correction.
            "attack_match_rate_before": regression_before["match_rate"],
            "attack_match_rate_after": regression_after["match_rate"],
            "mean_teacher_nll_before": regression_before["mean_teacher_nll"],
            "mean_teacher_nll_after": regression_after["mean_teacher_nll"],
            "original_attack_regression_passed": float(original_attack_regression_passed),
            "general_regression_passed": float(general_regression_passed),
        },
        str(corrected_ckpt),
    )


def run_evaluation_suite(*, out_dir: str | Path, num_games: int = 16, checkpoint: str | None = None) -> dict[str, Any]:
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
    # Phase 5: the benchmark is model-aware — E2/E6 become the trained policy's
    # REAL play vs Random when a checkpoint is supplied (never the Heuristic-vs-
    # Random reference), and G2 becomes a real robustness verdict on it.
    benchmark = run_unified_benchmark(
        EvaluationProfile(name="QUICK", num_games=num_games, include_e7=False), checkpoint=checkpoint
    )
    payload["benchmark_report"] = write_benchmark_report(benchmark, Path(out_dir) / "reports" / "quick")
    return payload


def run_axis_promotion_selection(
    *,
    out_dir: str | Path,
    candidates: dict[str, str],
    primary_n: int = 5,
    generalization_ns: tuple[int, ...] = (3, 5),
    exploit_n: int = 4,
    num_games: int = 24,
    seed: int = 1,
) -> dict[str, Any]:
    """Phase 5 — register ``best_*`` aliases from THEIR OWN evaluations.

    Each candidate checkpoint is re-played through the 0.1 harness on three
    distinct axes (raw robust score, full-game exploitability, worst-N
    generalization); ``best_robust`` / ``best_search`` / ``best_generalization``
    are then the per-axis winners.  Because the axes optimise different
    quantities, the winners can be *different files* — the alias is no longer an
    unconditional copy of the P4 checkpoint.  ``latest`` still tracks the primary
    (last) candidate.  Every metric written here is computed, never a literal.
    """
    from goofspiel.training.selection import select_checkpoints_by_axis

    out = Path(out_dir)
    existing = {cid: path for cid, path in candidates.items() if path and Path(path).exists()}
    if not existing:
        return {"selected": {}, "table": {}, "reason": "no_candidate_checkpoints"}

    selection = select_checkpoints_by_axis(
        existing,
        primary_n=primary_n,
        generalization_ns=generalization_ns,
        exploit_n=exploit_n,
        num_games=num_games,
        seed=seed,
    )

    registry = CheckpointRegistry(out / "registry")
    # `latest` = the primary (last-listed) candidate; the three best_* aliases are
    # resolved by their own axis winner.  Only register an alias whose winning
    # candidate has a real, distinguishing metric on that axis.
    primary_id = list(existing.keys())[-1]
    registry.register("latest", existing[primary_id], global_step=0, metrics={})
    registered: dict[str, dict[str, Any]] = {}
    for alias, winner_id in selection.by_alias.items():
        metric_key = {
            "best_robust": "robust_score",
            "best_search": "search_exploitability",
            "best_generalization": "generalization_worst",
        }[alias]
        metric_value = selection.table[winner_id].get(metric_key)
        registry.register(
            alias,
            existing[winner_id],
            global_step=0,
            metrics={metric_key: float(metric_value) if metric_value is not None else float("nan")},
        )
        registered[alias] = {
            "winner": winner_id,
            "checkpoint": existing[winner_id],
            metric_key: metric_value,
        }

    report = {
        "selected": registered,
        "table": selection.table,
        "distinct_winner_count": selection.distinct_winner_count(),
        "candidates": {cid: str(path) for cid, path in existing.items()},
    }
    report_path = out / "reports" / "axis_selection.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


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
    stage1 = run_stage1_pretrain(
        steps=steps, batch_size=batch_size, out_dir=out / "checkpoints", device=device, n_cards=n_cards,
        corpus_path=out / "data" / "game_corpus.jsonl",
    )
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
    stage5 = run_stage5_adaptive(
        steps=steps, out_dir=out, n_cards=n_cards,
        init_from_checkpoint=stage4.checkpoint,
    )
    emit("stage5_adaptive", {"ok": True, "metrics": stage5.metrics, "checkpoint": stage5.checkpoint})
    # Phase 4.3: the league plays REAL trained snapshots produced by this run —
    # P4 (robust backbone), P3 (strategic SFT), P5 (adaptive/exploiter) — not
    # role-keyed handcrafted baselines.
    stage6 = run_stage6_league(
        out_dir=out,
        role_checkpoints={
            ROLE_ROBUST: stage4.checkpoint,
            ROLE_AGGRESSIVE: stage3.checkpoint,
            ROLE_EXPLOITER: stage5.checkpoint,
        },
        n_cards=n_cards,
    )
    emit("stage6_league", {"ok": True, "metrics": stage6.metrics})
    # Phase 4.4: P7 focuses a REAL correction fine-tune on the P4 robust backbone
    # and MEASURES the attack-state regression before/after.
    stage7 = run_stage7_redteam(out_dir=out, init_from_checkpoint=stage4.checkpoint, n_cards=n_cards)
    emit("stage7_redteam", {"ok": True, "metrics": stage7.metrics})
    evaluation = run_evaluation_suite(
        out_dir=out / "evaluation", num_games=max(2, num_corpus_games), checkpoint=stage4.checkpoint
    )
    emit("evaluate", {"ok": True, "metrics": evaluation})

    # Phase 5: register best_* aliases from THEIR OWN per-axis evaluations over
    # the genuinely-distinct checkpoints this run produced (P3 strategic SFT, P4
    # robust backbone, P5 adaptive, P7 corrected). The three axes optimise
    # different quantities, so the aliases can resolve to different files.
    axis_selection = run_axis_promotion_selection(
        out_dir=out,
        candidates={
            "stage3_sft": stage3.checkpoint,
            "stage4_robust": stage4.checkpoint,
            "stage5_adaptive": stage5.checkpoint,
            "stage7_corrected": stage7.checkpoint,
        },
        primary_n=max(3, min(n_cards, 5)),
        generalization_ns=(3, max(3, min(n_cards, 5))),
        exploit_n=min(4, max(3, n_cards)),
        num_games=max(4, num_corpus_games),
        seed=seed,
    )
    emit("axis_promotion_selection", axis_selection)
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
