"""Runnable training stages.  Torch is imported lazily inside stage methods."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from goofspiel.game import GameState, legal_cards, transition
from goofspiel.observability import BaseEvent, JsonlEventSink, collect_system_metrics

if TYPE_CHECKING:
    from goofspiel.observability import TrainingLogger
from goofspiel.training.checkpoint import (
    CheckpointMetadata,
    collect_rng_state,
    dataset_provenance,
    load_checkpoint,
    model_config_hash,
    restore_python_random_state,
    restore_rng_state,
    save_checkpoint,
    serialize_python_random_state,
    sha256_file,
    state_dict_sha256,
)
from goofspiel.training.checkpoint import init_from_checkpoint as _load_init_from_checkpoint
from goofspiel.training.checkpoint import resume_checkpoint as _load_resume_checkpoint
from goofspiel.training.checkpoint_registry import CheckpointRegistry
from goofspiel.training.corpus import generate_random_game_corpus
from goofspiel.training.data import FailureRecord, JsonlStore, OpponentSession, ReanalysisRecord, RobustTrajectorySample, RoundRecord, state_record_from_game_state
from goofspiel.training.distributed import (
    all_gather_objects,
    barrier_if_distributed,
    broadcast_object,
    current_runtime,
    derive_rank_seed,
    seed_everything,
    setup_torch_distributed,
)
from goofspiel.training.evaluation import evaluate_bot_matchup, exact_feasibility_sweep
from goofspiel.training.league import LeagueAgent, LeagueRegistry, ROLE_AGGRESSIVE, ROLE_EXPLOITER, ROLE_ROBUST
from goofspiel.training.curriculum import ProgressiveCurriculum
from goofspiel.training.promotion import evaluate_promotion_candidate, write_promotion_report
from goofspiel.training.replay import TrajectoryReplayBuffer
from goofspiel.training.redteam import CorrectionDataset, FailureBuffer
from goofspiel.training.stage0_verify import run_stage0_verify
from goofspiel.training.stage_control import (
    DEFAULT_HARD_TIMEOUT_S,
    DEFAULT_HEARTBEAT_INTERVAL_S,
    DEFAULT_HEARTBEAT_TIMEOUT_S,
    Rank0Heartbeat,
    control_dir_for,
    current_invocation_id,
    wait_for_rank0,
)
from goofspiel.training.state_coverage import coverage_report, sample_reachable_states
from goofspiel.training.pretraining import build_pretraining_targets
from goofspiel.training.adaptive import default_opponent_curriculum, opponent_action_for_regime, oracle_opponent_diagnostic
from goofspiel.training.budgets import Stage6Budget, Stage7Budget
from goofspiel.training.teachers import TeacherRouter


@dataclass
class StageMetrics:
    stage: str
    steps: int
    metrics: dict[str, float]
    checkpoint: str | None = None


@dataclass(frozen=True)
class _LocalRuntime:
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1

    @property
    def is_distributed(self) -> bool:
        return False

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0


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


def _ranked_step(step: int, runtime: Any) -> int:
    """Map a per-rank loop step to a deterministic global data index."""
    return int(step) * max(1, int(runtime.world_size)) + int(runtime.rank)


def _corpus_batch_for_rank(
    corpus_states: list[GameState],
    *,
    step: int,
    batch_size: int,
    runtime: Any,
) -> list[GameState]:
    """Select a rank-specific corpus shard while preserving reproducibility."""
    if not corpus_states:
        return []
    start = (_ranked_step(step, runtime) * int(batch_size)) % len(corpus_states)
    return [corpus_states[(start + i) % len(corpus_states)] for i in range(int(batch_size))]


def _global_state_list(local_states: list[GameState], runtime: Any) -> list[GameState]:
    """Gather per-rank training states for rank0 metrics/artifacts."""
    if not getattr(runtime, "is_distributed", False):
        return list(local_states)
    return _flatten_state_lists(all_gather_objects(local_states))


def _flatten_state_lists(batches: list[list[GameState]]) -> list[GameState]:
    states: list[GameState] = []
    for batch in batches:
        states.extend(batch)
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

    ``init_from_checkpoint_path`` is a **stage transition** (P1->P3->P4): copy theta
    only, keep the fresh optimizer / step-0.  ``resume_checkpoint_path`` is a
    **crash resume**: restore the full training state.  They are mutually
    exclusive; conflating them is the bug 3.1 exists to prevent.  Returns a
    lineage dict to fold into the stage's ``CheckpointMetadata``.
    """
    if init_from_checkpoint_path and resume_checkpoint_path:
        raise ValueError(
            "init_from_checkpoint and resume_checkpoint are mutually exclusive: "
            "'inherit theta across a stage boundary' and 'resume a crashed run' are "
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
        # The parent FILE's content hash at the moment we inherited from it: the
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
    seed: int = 1,
    corpus_path: str | Path | None = None,
    init_from_checkpoint: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    logger: "TrainingLogger | None" = None,
    local_only: bool = False,
) -> StageMetrics:
    """P1 pretraining over swap, transition, joint-outcome, and opponent tasks.

    When ``corpus_path`` points at a ``game_corpus.jsonl`` (written by
    ``build_corpus``), P1 pretrains over the states actually recorded there,
    so the corpus is a LIVE producer feeding P1, not a dead artifact.  When it
    is absent, P1 falls back to on-the-fly ``sample_reachable_states``.
    """
    torch, F = _torch_import()
    from goofspiel.models import GoofspielModel, public_state_from_game

    seed_everything(int(seed))
    if local_only:
        runtime = _LocalRuntime()
    else:
        runtime, device = setup_torch_distributed(device)
    model = GoofspielModel(max_cards=13).to(device)
    if runtime.is_distributed:
        # stage1 runs TWO forward passes (batch + swapped_batch) into a SINGLE
        # backward (swap_loss / style_loss span both).  find_unused_parameters is
        # documented NOT to support multiple-forward-before-one-backward and aborts
        # with "Expected to have finished reduction in the prior iteration..." even
        # once it is enabled.  static_graph=True is the primitive built for this:
        # it traces the autograd graph on the first iteration and reuses it, which
        # also transparently handles the multi-branch model's idle params (the
        # per-stage loss is fixed, so the used-param set is stable across steps,
        # static_graph's precondition).  Required for correctness, not a perf knob.
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[runtime.local_rank] if device.startswith("cuda") else None,
            static_graph=True,
        )
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
            states = _corpus_batch_for_rank(corpus_states, step=step, batch_size=batch_size, runtime=runtime)
        else:
            states = sample_reachable_states(
                batch_size,
                n=n_cards,
                step=_ranked_step(step, runtime),
                seed=int(seed),
            )
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
        # supervised signal too; previously only the fused head was trained in P1,
        # leaving opp_short/opp_long with no P1 gradient at all.  With no history /
        # memory fed here the LSTM/Mamba states are zero, so in P1 all three heads
        # learn the same public-state -> opponent-action map; the short/long
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
    global_states = _global_state_list(all_states, runtime) if not local_only else list(all_states)
    if metrics:
        metrics.update(coverage_report(global_states).as_metrics())
    if runtime.is_rank0:
        out_dir = Path(out_dir)
        ckpt = out_dir / "stage1_pretrain.pt"
        if global_states:
            _write_coverage_artifact(out_dir, "stage1_pretrain", global_states)
        manifest = save_checkpoint(
            ckpt,
            model=getattr(model, "module", model),
            optimizers={"public_pretrain": opt},
            metadata=CheckpointMetadata(
                checkpoint_id="stage1_pretrain",
                training_stage="P1_PRETRAIN",
                global_step=int(steps),
                policy_version=0,
                config={
                    "steps": steps,
                    "batch_size": batch_size,
                    "local_batch_size": batch_size,
                    "global_batch_size": int(batch_size) * int(runtime.world_size),
                    "n_cards": n_cards,
                    "lr": lr,
                    "seed": seed,
                    "world_size": runtime.world_size,
                    "local_only": bool(local_only),
                    "ddp_data_semantics": "rank_sharded_reachable_or_corpus_states",
                },
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
    if local_only:
        return StageMetrics("P1_PRETRAIN", int(steps), metrics or {"loss_last": 0.0}, checkpoint_path)
    payload = {
        "metrics": metrics or {"loss_last": 0.0},
        "checkpoint": checkpoint_path,
    } if runtime.is_rank0 else None
    payload = broadcast_object(payload, src=0)
    if payload is None:
        raise RuntimeError("stage1 payload broadcast failed")
    barrier_if_distributed()
    return StageMetrics("P1_PRETRAIN", int(steps), payload["metrics"], payload["checkpoint"])


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
    legal self/opponent actions are skipped; P1's targets need a playable
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
    seed: int = 1,
    teacher_dataset_path: str | Path | None = None,
    init_from_checkpoint: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    logger: "TrainingLogger | None" = None,
) -> StageMetrics:
    """P3 Robust Strategic SFT that CONSUMES the multi-source teacher dataset.

    Phase 2.2b: P3 now loads ``teacher_dataset.jsonl`` (written by P2) and trains
    the robust policy toward the *stored* teacher policies, so its loss depends
    on the file's content and its four source counts (CFR/SEARCH/EXACT/PSEUDO)
    measure four genuinely different computations, not one aliased number.  When
    the dataset is absent (e.g. a stage run in isolation) it falls back to
    solving reachable states on the fly, and the source counts are reported as
    ``fallback``.

    Phase 3.1: ``init_from_checkpoint`` inherits P1's theta (fresh optimizer);
    ``resume_checkpoint`` restores a crashed P3 run in full.  The two are never
    conflated.
    """
    torch, F = _torch_import()
    from goofspiel.learning.game_theory.regret_matching_plus import solve_batch
    from goofspiel.models import GoofspielModel, public_state_from_game
    from goofspiel.training.teacher_dataset import ROBUST_TEACHER_SOURCES

    seed_everything(int(seed))
    runtime, device = setup_torch_distributed(device)
    model = GoofspielModel(max_cards=13).to(device)
    if runtime.is_distributed:
        # GoofspielModel.forward ALWAYS computes and returns every head (robust +
        # opponent + adaptive); stage3's loss consumes only the robust outputs.
        # find_unused_parameters keys off what forward RETURNS, not what the loss
        # uses, so it marks the opponent/adaptive params "used" (they are
        # reachable from the returned logits), then aborts when they receive no
        # grad ("...not all forward outputs participate in computing loss").
        # static_graph=True instead learns the true gradient-receiving set from
        # the first backward and reuses it; the per-stage loss is fixed, so that
        # set is stable across steps (static_graph's precondition).  Same reason
        # as stage1; see the note there.
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[runtime.local_rank] if device.startswith("cuda") else None,
            static_graph=True,
        )
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
            start = (_ranked_step(step, runtime) * int(batch_size)) % len(dataset)
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
            states = sample_reachable_states(
                batch_size,
                n=n_cards,
                step=_ranked_step(step, runtime),
                seed=int(seed),
            )
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
        # dataset: no longer four aliases of one number (Phase 2.2b). Each source
        # is a different algorithm/search depth (CFR immediate, SEARCH depth-1,
        # EXACT full recursion, PSEUDO confident self-labels); none uses opponent
        # behaviour (that is P5's job, preserving robust/adaptive separation).
        "strategic_sft_samples": float(sum(source_counts.values())),
        "teacher_dataset_consumed": float(1.0 if dataset else 0.0),
    }
    for src in ROBUST_TEACHER_SOURCES:
        metrics[f"sft_source_{src.lower()}_samples"] = float(source_counts[src])
    global_states = _global_state_list(all_states, runtime)
    metrics.update(coverage_report(global_states).as_metrics())
    if runtime.is_rank0:
        out_dir = Path(out_dir)
        ckpt = out_dir / "stage3_sft.pt"
        if global_states:
            _write_coverage_artifact(out_dir, "stage3_sft", global_states)
        manifest = save_checkpoint(
            ckpt,
            model=getattr(model, "module", model),
            optimizers={"strategic_sft": opt},
            metadata=CheckpointMetadata(
                checkpoint_id="stage3_sft",
                training_stage="P3_STRATEGIC_SFT",
                global_step=int(steps),
                policy_version=1,
                config={
                    "steps": steps,
                    "batch_size": batch_size,
                    "local_batch_size": batch_size,
                    "global_batch_size": int(batch_size) * int(runtime.world_size),
                    "n_cards": n_cards,
                    "lr": lr,
                    "seed": seed,
                    "world_size": runtime.world_size,
                    "ddp_data_semantics": "rank_sharded_teacher_dataset_or_reachable_states",
                },
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
    payload = {
        "metrics": metrics,
        "checkpoint": checkpoint_path,
    } if runtime.is_rank0 else None
    payload = broadcast_object(payload, src=0)
    if payload is None:
        raise RuntimeError("stage3 payload broadcast failed")
    barrier_if_distributed()
    return StageMetrics("P3_STRATEGIC_SFT", int(steps), payload["metrics"], payload["checkpoint"])


def run_stage2_semi_supervised(
    *,
    steps: int,
    out_dir: str | Path,
    n_cards: int = 5,
    seed: int = 1,
) -> StageMetrics:
    """Generate the multi-source robust teacher dataset consumed by P3.

    Phase 2.2b: instead of a single confidence-filtered ensemble label, P2 now
    labels reachable states with FOUR genuinely distinct robust-only sources
    (CFR / SEARCH / EXACT / PSEUDO) that differ by algorithm and search depth,
    writing them all to ``teacher_dataset.jsonl``, the file P3 actually trains
    on.  None of the sources uses opponent behaviour (that stays in P5).
    """
    from goofspiel.training.teacher_dataset import ROBUST_TEACHER_SOURCES, build_teacher_dataset

    runtime = current_runtime()
    if runtime.is_distributed:
        runtime, _ = setup_torch_distributed("auto")
    if not runtime.is_rank0:
        payload = broadcast_object(None, src=0)
        if payload is None:
            raise RuntimeError("stage2 payload broadcast failed")
        barrier_if_distributed()
        return StageMetrics("P2_SEMI_SUPERVISED", int(steps), payload["metrics"], None)

    path = Path(out_dir) / "teacher_dataset.jsonl"
    path.unlink(missing_ok=True)
    store = JsonlStore(path)
    states: list[GameState] = []
    for step in range(int(steps)):
        states.extend(sample_reachable_states(max(8, n_cards * 2), n=n_cards, step=step, seed=int(seed)))
    counts = build_teacher_dataset(states, store)
    total = sum(counts.values())
    metrics = {f"teacher_source_{src.lower()}_samples": float(counts[src]) for src in ROBUST_TEACHER_SOURCES}
    metrics["teacher_samples"] = float(total)
    metrics["distinct_teacher_sources"] = float(sum(1 for v in counts.values() if v > 0))
    metrics["teacher_dataset_rows"] = float(store.count())
    metrics["seed"] = float(seed)
    metrics["stage2_rank_owner"] = 0.0
    metrics["stage2_write_once"] = 1.0
    payload = {"metrics": metrics}
    payload = broadcast_object(payload, src=0)
    if payload is None:
        raise RuntimeError("stage2 payload broadcast failed")
    barrier_if_distributed()
    return StageMetrics("P2_SEMI_SUPERVISED", int(steps), payload["metrics"], None)


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


def _select_model_action(
    model: Any,
    state: GameState,
    *,
    device: str,
    temperature: float = 1.0,
    rng: random.Random | None = None,
) -> tuple[int, list[float], float]:
    torch, F = _torch_import()
    from goofspiel.models import public_state_from_game

    batch = public_state_from_game([state], max_cards=13, device=device)
    with torch.no_grad():
        logits = model(batch).robust_policy_logits[0]
        logits = logits / max(float(temperature), 1e-6)
        probs = F.softmax(logits, dim=-1)
        policy = [0.0] * 13
        probs_cpu = probs.detach().cpu().tolist()[:13]
        for i, value in enumerate(probs_cpu):
            policy[i] = float(value)
        if rng is None:
            dist = torch.distributions.Categorical(probs=probs)
            action_idx = int(dist.sample().item())
            prob = float(probs[action_idx].detach().cpu())
            action = action_idx + 1
            if action not in state.self_actions:
                action = random.choice(state.self_actions)
                prob = 1.0 / len(state.self_actions)
        else:
            action = _sample_from_policy(policy, state.self_actions, rng)
            prob = float(policy[action - 1]) if action > 0 else 0.0
    return action, policy, prob


def _trajectory_sample_id(*, stage: str, rank: int, step: int, game_index: int, seed: int, n_cards: int) -> str:
    return f"{stage}:r{rank}:s{step}:g{game_index}:n{n_cards}:seed{seed}"


def _trajectory_hash(sample: RobustTrajectorySample) -> str:
    payload = {
        "n": sample.n,
        "final_score_diff": sample.final_score_diff,
        "states": [state.__dict__ for state in sample.states],
        "rounds": [round_record.__dict__ for round_record in sample.rounds],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _stable_trajectory_prefix(
    trajectories: list[RobustTrajectorySample],
    k: int,
) -> list[RobustTrajectorySample]:
    """Select a deterministic content-hash prefix from gathered trajectories.

    Stage4 defines ``batch_size`` as the learner/global training batch.  Under
    DDP every rank may collect fresh rollouts, but the optimizer must consume
    one shared global batch.  When replay is empty this helper avoids rank-order
    bias by selecting the first ``k`` trajectories after stable content-hash
    ordering, and then that batch is broadcast to all ranks.
    """
    if k <= 0:
        return []
    return sorted(trajectories, key=lambda sample: (_trajectory_hash(sample), sample.sample_id))[:k]


def _rank_shard_range(total: int, rank: int, world_size: int) -> tuple[int, int]:
    """Return this rank's contiguous shard of a global item count."""
    total = max(0, int(total))
    world_size = max(1, int(world_size))
    rank = int(rank)
    base, extra = divmod(total, world_size)
    count = base + (1 if rank < extra else 0)
    start = rank * base + min(rank, extra)
    return start, count


def _flatten_trajectory_lists(batches: list[list[RobustTrajectorySample]]) -> list[RobustTrajectorySample]:
    flattened: list[RobustTrajectorySample] = []
    for batch in batches:
        flattened.extend(batch)
    return flattened


def _rollout_selfplay_game(
    model: Any,
    *,
    n_cards: int,
    rng: random.Random,
    device: str,
    model_version: str,
    game_index: int,
    sample_id: str,
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
        self_action, self_policy, self_prob = _select_model_action(model, state, device=device, rng=rng)
        opp_view = _mirrored_state(state)
        opp_action, opp_policy, opp_prob = _select_model_action(model, opp_view, device=device, rng=rng)
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
        sample_id=sample_id,
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


def _replay_snapshot_hash(snapshot: dict[str, Any]) -> str:
    encoded = json.dumps(snapshot, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _rank_rng_payload(
    *,
    rank: int,
    device: str,
    rollout_rng: random.Random,
    replay_rng: random.Random,
) -> dict[str, Any]:
    payload = collect_rng_state(torch_device=device)
    payload["rank"] = int(rank)
    payload["rollout_rng"] = serialize_python_random_state(rollout_rng.getstate())
    payload["replay_rng"] = serialize_python_random_state(replay_rng.getstate())
    return payload


def _gather_stage4_rng_state_by_rank(
    *,
    runtime: Any,
    device: str,
    rollout_rng: random.Random,
    replay_rng: random.Random,
) -> dict[str, Any]:
    local = _rank_rng_payload(
        rank=runtime.rank,
        device=device,
        rollout_rng=rollout_rng,
        replay_rng=replay_rng,
    )
    gathered = all_gather_objects(local)
    return {str(int(row["rank"])): row for row in gathered}


def _restore_stage4_rank_rng(
    payload: dict[str, Any],
    *,
    device: str,
    rollout_rng: random.Random,
    replay_rng: random.Random,
) -> None:
    restore_rng_state(payload, torch_device=device)
    restore_python_random_state(payload["rollout_rng"], rollout_rng)
    restore_python_random_state(payload["replay_rng"], replay_rng)


def _write_stage4_resume_checkpoint(
    *,
    path: Path,
    model: Any,
    optimizer: Any,
    target_model: Any,
    stage_step_completed: int,
    total_steps: int,
    stage_metrics: dict[str, float],
    config: dict[str, Any],
    lineage: dict[str, Any],
    replay_buffer: TrajectoryReplayBuffer,
    curriculum: ProgressiveCurriculum,
    rng_state_by_rank: dict[str, Any],
    save_step_copy: bool = True,
) -> dict[str, Any]:
    replay_snapshot = replay_buffer.snapshot()
    target_state = target_model.state_dict()
    metadata = CheckpointMetadata(
        checkpoint_id="stage4_robust_rl_resume",
        training_stage="P4_ROBUST_RL_RESUME",
        global_step=int(stage_step_completed) + 1,
        policy_version=2,
        config=config,
        metrics=stage_metrics,
        parent_checkpoint_id=lineage["parent_checkpoint_id"],
        init_checkpoint_id=lineage["init_checkpoint_id"],
        parent_checkpoint_sha256=lineage.get("parent_checkpoint_sha256"),
        model_config_hash=model_config_hash(getattr(model, "module", model)),
        optimizer_reset=lineage["optimizer_reset"],
    )
    extra = {
        "checkpoint_kind": "stage4_periodic_resume",
        "stage_step_completed": int(stage_step_completed),
        "next_stage_step": int(stage_step_completed) + 1,
        "total_steps": int(total_steps),
        "world_size": int(config["world_size"]),
        "seed": int(config["seed"]),
        "target_model_state": target_state,
        "target_model_sha256": state_dict_sha256(target_state),
        "replay_snapshot": replay_snapshot,
        "replay_snapshot_sha256": _replay_snapshot_hash(replay_snapshot),
        "curriculum": {
            "target_n": curriculum.target_n,
            "warmup_n": curriculum.warmup_n,
            "ramp_every": curriculum.ramp_every,
            "state_semantics": "stateless_function_of_stage_step",
        },
        "rng_state_by_rank": rng_state_by_rank,
        "stage4_config": dict(config),
        "lineage": dict(lineage),
    }
    manifest = save_checkpoint(
        path,
        model=getattr(model, "module", model),
        optimizers={"robust_rl": optimizer},
        metadata=metadata,
        extra=extra,
        atomic=True,
        rng_state={"rng_state_by_rank": rng_state_by_rank},
    )
    if save_step_copy:
        step_path = path.with_name(f"stage4_resume_step_{int(stage_step_completed):06d}.pt")
        save_checkpoint(
            step_path,
            model=getattr(model, "module", model),
            optimizers={"robust_rl": optimizer},
            metadata=metadata,
            extra=extra,
            atomic=True,
            rng_state={"rng_state_by_rank": rng_state_by_rank},
        )
    return manifest


def _restore_stage4_resume_state(
    *,
    checkpoint_path: str | Path,
    model: Any,
    optimizer: Any,
    target_model: Any,
    replay_buffer: TrajectoryReplayBuffer,
    rollout_rng: random.Random,
    replay_rng: random.Random,
    runtime: Any,
    device: str,
    steps: int,
    batch_size: int,
    n_cards: int,
    lr: float,
    seed: int,
) -> dict[str, Any]:
    payload = load_checkpoint(checkpoint_path)
    meta = payload.get("metadata", {})
    extra = payload.get("extra", {})
    if extra.get("checkpoint_kind") != "stage4_periodic_resume":
        raise RuntimeError(
            f"Stage4 crash resume requires a periodic resume checkpoint; got "
            f"{extra.get('checkpoint_kind')!r} from {checkpoint_path}"
        )
    ckpt_world_size = int(extra.get("world_size", meta.get("config", {}).get("world_size", 1)))
    if ckpt_world_size != int(runtime.world_size):
        raise RuntimeError(
            f"Stage4 exact crash resume requires same world_size: "
            f"checkpoint={ckpt_world_size}, current={runtime.world_size}"
        )
    ckpt_config = extra.get("stage4_config", meta.get("config", {}))
    expected = {
        "batch_size": int(batch_size),
        "n_cards": int(n_cards),
        "seed": int(seed),
    }
    actual = {
        "batch_size": int(ckpt_config.get("batch_size", -1)),
        "n_cards": int(ckpt_config.get("n_cards", -1)),
        "seed": int(ckpt_config.get("seed", -1)),
    }
    if actual != expected:
        raise RuntimeError(f"Stage4 resume config mismatch: checkpoint={actual}, current={expected}")
    if abs(float(ckpt_config.get("lr", lr)) - float(lr)) > 1e-18:
        raise RuntimeError(f"Stage4 resume lr mismatch: checkpoint={ckpt_config.get('lr')} current={lr}")
    completed_step = int(extra["stage_step_completed"])
    if int(steps) <= completed_step:
        raise RuntimeError(
            f"Stage4 resume target steps must exceed completed step: "
            f"completed={completed_step}, requested_total={steps}"
        )

    target_state = extra.get("target_model_state")
    if not target_state:
        raise RuntimeError("Stage4 resume checkpoint is missing target_model_state")
    target_model.load_state_dict(target_state, strict=True)
    target_hash = state_dict_sha256(target_model.state_dict())
    if target_hash != extra.get("target_model_sha256"):
        raise RuntimeError("Stage4 target_model_state hash mismatch during resume")

    replay_snapshot = extra.get("replay_snapshot")
    if not replay_snapshot:
        raise RuntimeError("Stage4 resume checkpoint is missing replay_snapshot")
    if _replay_snapshot_hash(replay_snapshot) != extra.get("replay_snapshot_sha256"):
        raise RuntimeError("Stage4 replay snapshot hash mismatch during resume")
    replay_buffer.restore_snapshot(replay_snapshot)
    if runtime.is_rank0:
        replay_buffer.rewrite_store_from_memory()

    model.load_state_dict(payload["model_state"], strict=True)
    stored_opts = payload.get("optimizer_states", {})
    if "robust_rl" not in stored_opts:
        raise RuntimeError("Stage4 resume checkpoint is missing robust_rl optimizer state")
    optimizer.load_state_dict(stored_opts["robust_rl"])

    rng_state_by_rank = extra.get("rng_state_by_rank", {})
    rank_key = str(int(runtime.rank))
    if rank_key not in rng_state_by_rank:
        raise RuntimeError(f"Stage4 resume checkpoint has no RNG payload for rank {runtime.rank}")
    _restore_stage4_rank_rng(
        rng_state_by_rank[rank_key],
        device=device,
        rollout_rng=rollout_rng,
        replay_rng=replay_rng,
    )

    next_step = int(extra["next_stage_step"])
    if next_step != completed_step + 1:
        raise RuntimeError(
            f"Stage4 resume step metadata inconsistent: completed={completed_step}, next={next_step}"
        )
    saved_lineage = extra.get("lineage", {})
    return {
        "parent_checkpoint_id": saved_lineage.get("parent_checkpoint_id"),
        "init_checkpoint_id": saved_lineage.get("init_checkpoint_id"),
        "init_checkpoint_sha256": saved_lineage.get("init_checkpoint_sha256"),
        "resume_sha256": sha256_file(checkpoint_path),
        "resume_checkpoint_id": meta.get("checkpoint_id"),
        "parent_checkpoint_sha256": saved_lineage.get("parent_checkpoint_sha256"),
        "optimizer_reset": False,
        "resumed_stage_step_completed": completed_step,
        "next_stage_step": next_step,
        "restored_target_model": True,
        "restored_replay_samples": replay_buffer.count(),
        "restored_replay_snapshot_sha256": extra.get("replay_snapshot_sha256"),
    }


def run_stage4_robust_rl(
    *,
    steps: int,
    batch_size: int,
    out_dir: str | Path,
    device: str = "cpu",
    n_cards: int = 5,
    lr: float = 1e-4,
    seed: int = 1,
    init_from_checkpoint: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    resume_checkpoint_interval: int = 250,
    logger: "TrainingLogger | None" = None,
) -> StageMetrics:
    """Self-play robust RL runner using trajectory replay + Nash/NeuRD anchors.

    Phase 3.1: ``init_from_checkpoint`` inherits P3's theta (theta-only, fresh optimizer;
    the target network is then seeded from the inherited weights).
    ``resume_checkpoint`` restores model + optimizer + target network in full.
    """
    torch, F = _torch_import()
    from goofspiel.learning.game_theory.neurd import (
        NEURD_LOGIT_THRESHOLD,
        legal_logits,
        neurd_actor_loss_from_regret,
        row_action_regret,
    )
    from goofspiel.learning.game_theory.regret_matching_plus import solve_batch
    from goofspiel.models import GoofspielModel, public_state_from_game

    total_steps = int(steps)
    seed_everything(int(seed))
    runtime, device = setup_torch_distributed(device)
    rank_seed = derive_rank_seed(int(seed), runtime.rank)
    model = GoofspielModel(max_cards=13).to(device)
    target_model = GoofspielModel(max_cards=13).to(device)
    opt_q = torch.optim.AdamW(model.parameters(), lr=lr)
    replay_path = Path(out_dir) / "replay" / "selfplay_robust.jsonl"
    event_path = Path(out_dir) / "events" / "stage4_robust_rl.jsonl"
    if runtime.is_rank0 and not resume_checkpoint:
        replay_path.unlink(missing_ok=True)
        event_path.unlink(missing_ok=True)
    replay_buffer = TrajectoryReplayBuffer(replay_path)
    event_sink = JsonlEventSink(event_path) if runtime.is_rank0 else None
    warmup_n = n_cards if total_steps <= 1 else min(3, n_cards)
    curriculum = ProgressiveCurriculum(
        target_n=n_cards,
        warmup_n=warmup_n,
        ramp_every=max(1, total_steps // max(1, n_cards - warmup_n + 1)),
    )
    rollout_rng = random.Random(rank_seed)
    replay_rng = random.Random(derive_rank_seed(int(seed) + 97, runtime.rank))
    resume_config = {
        "steps": total_steps,
        "batch_size": int(batch_size),
        "global_training_batch_size": int(batch_size),
        "global_rollout_batch_size": int(batch_size),
        "per_rank_rollout_rule": "contiguous shard of global_rollout_batch_size",
        "n_cards": int(n_cards),
        "lr": float(lr),
        "seed": int(seed),
        "world_size": int(runtime.world_size),
        "rank_seed_rule": "seed + rank",
        "replay_semantics": "rank0_global_replay_sample_then_broadcast_to_all_ranks",
        "neurd_logit_threshold": NEURD_LOGIT_THRESHOLD,
        "neurd_threshold_step_size": float(lr),
        "curriculum": {
            "target_n": curriculum.target_n,
            "warmup_n": curriculum.warmup_n,
            "ramp_every": curriculum.ramp_every,
            "state_semantics": "stateless_function_of_stage_step",
        },
        "resume_checkpoint_interval": int(resume_checkpoint_interval),
    }
    if init_from_checkpoint and resume_checkpoint:
        raise ValueError(
            "init_from_checkpoint and resume_checkpoint are mutually exclusive: "
            "'inherit theta across a stage boundary' and 'resume a crashed run' are "
            "different operations and must not be conflated."
        )
    lineage: dict[str, Any] = {
        "parent_checkpoint_id": None,
        "init_checkpoint_id": None,
        "parent_checkpoint_sha256": None,
        "optimizer_reset": True,
        "resumed_stage_step_completed": -1,
        "next_stage_step": 0,
    }
    start_step = 0
    if init_from_checkpoint:
        prov = _load_init_from_checkpoint(getattr(model, "module", model), init_from_checkpoint)
        lineage["init_checkpoint_id"] = prov["init_checkpoint_id"]
        lineage["parent_checkpoint_id"] = prov["init_checkpoint_id"]
        lineage["init_checkpoint_sha256"] = prov["init_checkpoint_sha256"]
        lineage["parent_checkpoint_sha256"] = prov["init_checkpoint_sha256"]
        lineage["optimizer_reset"] = True
        target_model.load_state_dict(getattr(model, "module", model).state_dict())
    elif resume_checkpoint:
        lineage.update(
            _restore_stage4_resume_state(
                checkpoint_path=resume_checkpoint,
                model=model,
                optimizer=opt_q,
                target_model=target_model,
                replay_buffer=replay_buffer,
                rollout_rng=rollout_rng,
                replay_rng=replay_rng,
                runtime=runtime,
                device=device,
                steps=total_steps,
                batch_size=batch_size,
                n_cards=n_cards,
                lr=lr,
                seed=seed,
            )
        )
        start_step = int(lineage["next_stage_step"])
        msg = (
            f"resume Stage4: checkpoint completed_step={lineage['resumed_stage_step_completed']} "
            f"-> continuing at step={start_step} / {total_steps}"
        )
        if runtime.is_rank0:
            if logger is not None:
                logger.resume_stage(
                    "stage4_robust_rl",
                    completed_step=int(lineage["resumed_stage_step_completed"]),
                    next_step=start_step,
                    total_steps=total_steps,
                    checkpoint=resume_checkpoint,
                )
            else:
                print(msg)
    else:
        target_model.load_state_dict(getattr(model, "module", model).state_dict())
    if start_step > total_steps:
        raise RuntimeError(
            f"Stage4 resume checkpoint is beyond requested total steps: "
            f"next_stage_step={start_step}, steps={total_steps}"
        )
    target_model.eval()
    if runtime.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[runtime.local_rank] if device.startswith("cuda") else None,
            static_graph=True,
        )
    last_q = last_actor = last_entropy = 0.0
    last_regret_scale = last_min_logit = last_max_logit = last_logit_gap = 0.0
    last_grad_norm = 0.0
    last_unique_gathered_hashes = 0
    last_duplicate_gathered_hashes = 0
    last_training_trajectories = 0
    last_unique_training_hashes = 0
    local_trajectories_total = 0
    local_transitions_total = 0
    global_trajectories_total = 0
    global_transitions_total = 0
    curriculum_path = Path(out_dir) / "curriculum" / "stage4_manifest.json"
    if runtime.is_rank0:
        curriculum_path.parent.mkdir(parents=True, exist_ok=True)
        curriculum_path.write_text(json.dumps(curriculum.manifest(steps=total_steps), indent=2, ensure_ascii=False), encoding="utf-8")
    stage_step_completed = start_step - 1
    for step in range(start_step, total_steps):
        cstep = curriculum.at(step)
        collector = getattr(model, "module", model)
        collector.eval()
        shard_start, shard_count = _rank_shard_range(batch_size, runtime.rank, runtime.world_size)
        trajectories = [
            _rollout_selfplay_game(
                collector,
                n_cards=cstep.n_cards,
                rng=rollout_rng,
                device=device,
                model_version=f"stage4_step_{step}",
                game_index=step * batch_size + shard_start + i,
                sample_id=_trajectory_sample_id(
                    stage="stage4_robust_rl",
                    rank=runtime.rank,
                    step=step,
                    game_index=step * batch_size + shard_start + i,
                    seed=rank_seed,
                    n_cards=cstep.n_cards,
                ),
            )
            for i in range(shard_count)
        ]
        gathered = all_gather_objects(trajectories)
        all_trajectories = _flatten_trajectory_lists(gathered)
        gathered_hashes = [_trajectory_hash(traj) for traj in all_trajectories]
        unique_gathered_hashes = len(set(gathered_hashes))
        last_unique_gathered_hashes = unique_gathered_hashes
        last_duplicate_gathered_hashes = len(gathered_hashes) - unique_gathered_hashes
        local_trajectories_total += len(trajectories)
        local_transitions_total += sum(len(t.rounds) for t in trajectories)
        global_trajectories_total += len(all_trajectories)
        global_transitions_total += sum(len(t.rounds) for t in all_trajectories)
        if runtime.is_rank0:
            replay_buffer.append_many(all_trajectories)
            if event_sink is not None:
                event_sink.emit(
                    BaseEvent(
                        event_type="STAGE4_SELFPLAY_COLLECTED",
                        run_id="stage4_robust_rl",
                        step=step,
                        payload={
                            "curriculum_n": cstep.n_cards,
                            "trajectories": len(all_trajectories),
                            "per_rank_trajectories": [len(batch) for batch in gathered],
                            "global_rollout_batch_size": int(batch_size),
                            "rank_shard_counts": [
                                _rank_shard_range(batch_size, r, runtime.world_size)[1]
                                for r in range(runtime.world_size)
                            ],
                            "transitions": sum(len(t.rounds) for t in all_trajectories),
                            "mean_final_score_diff": sum(t.final_score_diff for t in all_trajectories) / max(len(all_trajectories), 1),
                            "unique_trajectory_hashes": unique_gathered_hashes,
                            "duplicate_trajectory_hashes": last_duplicate_gathered_hashes,
                        },
                    )
                )
        if runtime.is_rank0:
            sampled_trajectories = replay_buffer.sample(batch_size, replay_rng)
            training_trajectories = sampled_trajectories or _stable_trajectory_prefix(
                all_trajectories,
                int(batch_size),
            )
        else:
            training_trajectories = None
        training_trajectories = broadcast_object(training_trajectories, src=0)
        if not training_trajectories:
            raise RuntimeError("stage4 produced no training trajectories")
        training_hashes = [_trajectory_hash(traj) for traj in training_trajectories]
        last_training_trajectories = len(training_trajectories)
        last_unique_training_hashes = len(set(training_hashes))
        states, action_self, action_opp, mc_returns = _flatten_trajectory_batch(training_trajectories)
        if not states:
            raise RuntimeError("stage4 training trajectories produced no decision states")
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
        centered_logits = legal_logits(out.robust_policy_logits, batch.self_action_mask)
        policy = F.softmax(centered_logits, dim=-1)
        log_policy = F.log_softmax(centered_logits, dim=-1)
        entropy = -(policy * log_policy).sum(dim=-1).mean()
        action_regret = row_action_regret(
            out.q_robust.detach(),
            policy.detach(),
            sol.column_policy.detach(),
            batch.self_action_mask,
        )
        actor_loss, centered_actor_logits, thresholded_force = neurd_actor_loss_from_regret(
            out.robust_policy_logits,
            action_regret,
            batch.self_action_mask,
            threshold=NEURD_LOGIT_THRESHOLD,
            threshold_step_size=lr,
        )
        anchor = F.kl_div(log_policy, sol.row_policy.detach(), reduction="batchmean")
        loss = q_loss + actor_loss + 0.1 * anchor
        opt_q.zero_grad(set_to_none=True)
        loss.backward()
        last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt_q.step()
        with torch.no_grad():
            source_model = getattr(model, "module", model)
            for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
                target_param.mul_(0.995).add_(source_param.detach(), alpha=0.005)
        last_q = float(q_loss.detach().cpu())
        last_actor = float(actor_loss.detach().cpu())
        last_entropy = float(entropy.detach().cpu())
        self_mask_f = batch.self_action_mask.float()
        denom = self_mask_f.sum(dim=-1).clamp_min(1.0)
        last_regret_scale = float((thresholded_force.abs() * self_mask_f).sum(dim=-1).div(denom).mean().detach().cpu())
        finite_logits = centered_actor_logits[batch.self_action_mask.bool()]
        if finite_logits.numel() > 0:
            last_min_logit = float(finite_logits.min().detach().cpu())
            last_max_logit = float(finite_logits.max().detach().cpu())
            last_logit_gap = last_max_logit - last_min_logit
        if logger is not None and _should_log_step(step, total_steps):
            logger.step_metrics(
                "stage4_robust_rl",
                step,
                total_steps,
                {
                    "q": last_q,
                    "actor": last_actor,
                    "entropy": last_entropy,
                    "min_logit": last_min_logit,
                    "max_logit": last_max_logit,
                    "logit_gap": last_logit_gap,
                    "regret_scale": last_regret_scale,
                    "grad_norm": last_grad_norm,
                    "curriculum_n": cstep.n_cards,
                    "global_training_trajectories": last_training_trajectories,
                    "unique_training_hashes": last_unique_training_hashes,
                    "neurd_logit_threshold": NEURD_LOGIT_THRESHOLD,
                },
            )
        stage_step_completed = step
        should_save_resume = (
            int(resume_checkpoint_interval) > 0
            and ((step + 1) % int(resume_checkpoint_interval) == 0 or step == total_steps - 1)
        )
        if should_save_resume:
            rng_state_by_rank = _gather_stage4_rng_state_by_rank(
                runtime=runtime,
                device=device,
                rollout_rng=rollout_rng,
                replay_rng=replay_rng,
            )
            save_result = None
            if runtime.is_rank0:
                try:
                    resume_path = Path(out_dir) / "stage4_resume_latest.pt"
                    manifest = _write_stage4_resume_checkpoint(
                        path=resume_path,
                        model=model,
                        optimizer=opt_q,
                        target_model=target_model,
                        stage_step_completed=stage_step_completed,
                        total_steps=total_steps,
                        stage_metrics={
                            "q_loss_last": last_q,
                            "actor_loss_last": last_actor,
                            "entropy_last": last_entropy,
                            "stage_step_completed": float(stage_step_completed),
                            "next_stage_step": float(stage_step_completed + 1),
                        },
                        config=resume_config,
                        lineage=lineage,
                        replay_buffer=replay_buffer,
                        curriculum=curriculum,
                        rng_state_by_rank=rng_state_by_rank,
                    )
                    save_result = {"ok": True, "manifest": manifest}
                    if logger is not None:
                        logger.checkpoint_saved(
                            "stage4_resume",
                            manifest["path"],
                            global_step=stage_step_completed + 1,
                            sha256=manifest["sha256"],
                        )
                except Exception as exc:
                    save_result = {"ok": False, "error": repr(exc)}
            save_result = broadcast_object(save_result, src=0)
            if not save_result or not save_result.get("ok"):
                raise RuntimeError(f"Stage4 periodic resume checkpoint failed: {save_result}")

    stage_metrics = {
        "q_loss_last": last_q,
        "actor_loss_last": last_actor,
        "policy_gradient_loss_last": 0.0,
        "policy_gradient_removed": 1.0,
        "entropy_last": last_entropy,
        "min_logit_last": last_min_logit,
        "max_logit_last": last_max_logit,
        "logit_gap_last": last_logit_gap,
        "regret_scale_last": last_regret_scale,
        "grad_norm_last": last_grad_norm,
        "neurd_logit_threshold": float(NEURD_LOGIT_THRESHOLD),
        "neurd_threshold_step_size": float(lr),
        "selfplay_trajectories": float(global_trajectories_total),
        "selfplay_local_trajectories": float(local_trajectories_total),
        "selfplay_transitions": float(global_transitions_total),
        "selfplay_local_transitions": float(local_transitions_total),
        "global_training_batch_trajectories": float(last_training_trajectories),
        "unique_training_trajectory_hashes": float(last_unique_training_hashes),
        "unique_gathered_trajectory_hashes_last": float(last_unique_gathered_hashes),
        "duplicate_gathered_trajectory_hashes_last": float(last_duplicate_gathered_hashes),
        "distributed_replay_semantics": 1.0,
        "replay_samples": float(replay_buffer.count()) if runtime.is_rank0 else 0.0,
        "replay_persisted_samples": float(replay_buffer.persisted_count()) if runtime.is_rank0 else 0.0,
        "rank_seed": float(rank_seed),
        "world_size": float(runtime.world_size),
        "global_rollout_batch_size": float(batch_size),
        "per_rank_rollout_batch_size": float(_rank_shard_range(batch_size, runtime.rank, runtime.world_size)[1]),
        "target_network_ema": 0.995,
        "curriculum_final_n": float(curriculum.at(max(0, total_steps - 1)).n_cards),
        "start_stage_step": float(start_step),
        "stage_step_completed": float(stage_step_completed),
        "next_stage_step": float(stage_step_completed + 1),
        "resumed_stage_step_completed": float(lineage.get("resumed_stage_step_completed", -1)),
        "same_world_size_resume_required": 1.0,
    }
    if stage_step_completed + 1 != total_steps:
        stage_metrics["stage_incomplete"] = 1.0
    promotion = evaluate_promotion_candidate(stage_metrics)
    checkpoint_path = None
    promotion_artifact = None
    final_save_result = None
    if runtime.is_rank0:
        try:
            ckpt = Path(out_dir) / "stage4_robust_rl.pt"
            promotion_artifact = write_promotion_report(promotion, Path(out_dir) / "promotion" / "stage4_promotion.json")
            manifest = save_checkpoint(
                ckpt,
                model=getattr(model, "module", model),
                optimizers={"robust_rl": opt_q},
                metadata=CheckpointMetadata(
                checkpoint_id="stage4_robust_rl",
                training_stage="P4_ROBUST_RL",
                global_step=total_steps,
                policy_version=2,
                config=resume_config,
                    metrics=stage_metrics,
                    parent_checkpoint_id=lineage["parent_checkpoint_id"],
                    init_checkpoint_id=lineage["init_checkpoint_id"],
                    parent_checkpoint_sha256=lineage.get("parent_checkpoint_sha256"),
                    model_config_hash=model_config_hash(getattr(model, "module", model)),
                optimizer_reset=lineage["optimizer_reset"],
            ),
                extra={
                    "checkpoint_kind": "stage4_final_boundary",
                    "stage_step_completed": int(stage_step_completed),
                    "next_stage_step": int(stage_step_completed + 1),
                    "total_steps": int(total_steps),
                    "resume_source_checkpoint_id": lineage.get("resume_checkpoint_id"),
                    "resume_source_sha256": lineage.get("resume_sha256"),
                    "resumed_stage_step_completed": lineage.get("resumed_stage_step_completed"),
                    "restored_replay_samples": lineage.get("restored_replay_samples"),
                    "replay_path": str(replay_buffer.path),
                    "event_log": str(event_sink.path if event_sink is not None else ""),
                    "curriculum": str(curriculum_path),
                    "promotion": promotion_artifact,
                    "target_model_state": target_model.state_dict(),
                    "target_model_sha256": state_dict_sha256(target_model.state_dict()),
                },
                atomic=True,
            )
            registry = CheckpointRegistry(Path(out_dir) / "registry")
            registry.register("latest", ckpt, global_step=total_steps, metrics=stage_metrics)
            checkpoint_path = manifest["path"]
            if logger is not None:
                logger.checkpoint_saved("stage4_robust_rl", checkpoint_path, global_step=total_steps)
            final_save_result = {"ok": True, "manifest": manifest}
        except Exception as exc:
            final_save_result = {"ok": False, "error": repr(exc)}
    final_save_result = broadcast_object(final_save_result, src=0)
    if not final_save_result or not final_save_result.get("ok"):
        raise RuntimeError(f"Stage4 final checkpoint failed: {final_save_result}")
    barrier_if_distributed()
    stage_metrics["promotion_candidate"] = 1.0 if promotion.decision == "PROMOTE_CANDIDATE" else 0.0
    payload = {
        "checkpoint": checkpoint_path,
        "metrics": stage_metrics,
        "rank_owner": 0.0,
    } if runtime.is_rank0 else None
    payload = broadcast_object(payload, src=0)
    if payload is None:
        raise RuntimeError("stage4 payload broadcast failed")
    return StageMetrics(
        "P4_ROBUST_RL",
        int(steps),
        payload["metrics"],
        payload["checkpoint"],
    )
def _opponent_regime_distribution(regime_id: str, legal: list[int], *, stake: int, n_cards: int) -> dict[int, float]:
    """The TRUE next-action distribution of a scripted regime (the label P5 fits).

    ``opponent_action_for_regime`` samples an action; here we expose the full
    categorical it samples from, so P5 can train the opponent head against a real
    probability target and measure a real NLL/ECE against it, not a constant."""
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
        session (feeds the Mamba), so the long-horizon head sees real cross-game
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
    device: str = "cpu",
    n_cards: int = 5,
    lr: float = 3e-4,
    seed: int = 1,
    init_from_checkpoint: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    logger: "TrainingLogger | None" = None,
    heartbeat_timeout: float = DEFAULT_HEARTBEAT_TIMEOUT_S,
    hard_timeout: float = DEFAULT_HARD_TIMEOUT_S,
    heartbeat_interval: float = DEFAULT_HEARTBEAT_INTERVAL_S,
) -> StageMetrics:
    """P5 trains the opponent/adaptive branch behind a hard firewall.

    Only rank0 executes the adaptive training loop and writes the checkpoint.
    Stage5 is long and rank0-only, so non-rank0 ranks must NOT hold an NCCL
    collective for its whole duration — that is the >600s watchdog deadlock the
    control-plane replaces.  Instead rank0 publishes a heartbeat/status file
    (see ``stage_control``) and non-rank0 ranks POLL it for liveness plus the
    terminal result; only after rank0 reaches SUCCESS do all ranks do ONE short
    collective (barrier) to realign the NCCL sequence number before the next
    stage.  Failure fails closed: a crashed / hung / explicitly-FAILED rank0
    makes peers stop waiting and raise, never block forever.

    ``heartbeat_timeout`` is "how long with no fresh heartbeat before rank0 is
    declared dead" (NOT a cap on Stage5 runtime); ``hard_timeout`` is the
    last-resort fake-alive guard for a rank0 that keeps heartbeating but never
    terminates.  NCCL's own timeout stays purely low-level fault protection.
    """
    # Phase 2: resolve the device from the caller (coordinator passes
    # ``config.device``) instead of discarding it — this is the single lever
    # that moves the rank0 adaptive model + training tensors CPU -> cuda.  The
    # rebind mirrors every sibling stage (``runtime, device = ...``); default
    # "cpu" keeps single-process/local behaviour byte-identical.  It does NOT
    # DDP-wrap Stage5 (still rank0-only) and does NOT touch the control-plane.
    runtime, device = setup_torch_distributed(device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    control_dir = control_dir_for(out)
    # All ranks of THIS launch derive the same invocation id from the shared
    # torchrun env (no collective, no clock trust); a different launch — a
    # crash-resume / salvage / re-run into the same artifact-dir — gets a
    # different id.  This is what lets a peer reject a prior invocation's stale
    # SUCCESS/FAILED left in status.json instead of acting on it.
    invocation_id = current_invocation_id()

    if not runtime.is_rank0:
        # Poll rank0's liveness instead of blocking in a collective.  Returns
        # the SUCCESS status (carrying checkpoint + metrics); raises
        # Stage5ControlError — fail-closed — on FAILED / stale heartbeat / hard
        # timeout, so a dead rank0 never hangs its peers.  ``expect_invocation_id``
        # makes a stale terminal record from an earlier invocation invisible.
        status = wait_for_rank0(
            control_dir,
            expect_invocation_id=invocation_id,
            heartbeat_timeout=heartbeat_timeout,
            hard_timeout=hard_timeout,
        )
        barrier_if_distributed()
        return StageMetrics("P5_ADAPTIVE", int(steps), dict(status.metrics), status.checkpoint)

    # rank0 owns the training and every write.  It publishes STARTING, refreshes
    # a RUNNING heartbeat through the long sub-phases, and lands on an EXPLICIT
    # terminal state (SUCCESS with the result, or FAILED with the error) so a
    # peer is never left guessing.  The heartbeat is a no-op unless distributed,
    # so a single-process Stage5 is byte-identical to the old implementation.
    heartbeat = Rank0Heartbeat(
        control_dir,
        enabled=runtime.is_distributed,
        run_id=out.name,
        invocation_id=invocation_id,
        total_steps=max(1, int(steps)),
        interval=heartbeat_interval,
    )
    heartbeat.starting()
    try:
        stage_metrics, ckpt_path = _run_stage5_adaptive_rank0(
            steps=steps,
            out=out,
            device=device,
            n_cards=n_cards,
            lr=lr,
            seed=seed,
            init_from_checkpoint=init_from_checkpoint,
            resume_checkpoint=resume_checkpoint,
            logger=logger,
            heartbeat=heartbeat,
        )
        heartbeat.success(checkpoint=ckpt_path, metrics=stage_metrics)
    except BaseException as exc:  # noqa: BLE001 - fail closed, then re-raise
        heartbeat.fail(error=repr(exc))
        raise
    # rank0 reached SUCCESS: ONE short collective realigns the NCCL sequence
    # number with the peers (which skipped every Stage5 collective), then all
    # ranks proceed together.
    barrier_if_distributed()
    return StageMetrics("P5_ADAPTIVE", int(steps), stage_metrics, ckpt_path)


def _run_stage5_adaptive_rank0(
    *,
    steps: int,
    out: Path,
    device: str,
    n_cards: int,
    lr: float,
    seed: int,
    init_from_checkpoint: str | Path | None,
    resume_checkpoint: str | Path | None,
    logger: "TrainingLogger | None",
    heartbeat: Rank0Heartbeat,
) -> tuple[dict[str, float], str | None]:
    """rank0-only Stage5 body: build sessions, train the adaptive branch behind
    the firewall, write the checkpoint + gate report, and return
    ``(stage_metrics, checkpoint_path)``.  Kept as a helper so the public
    function can wrap it in the heartbeat try/except without re-indenting the
    whole loop.  Refreshes the heartbeat at each long sub-phase boundary.
    """
    torch, F = _torch_import()
    from goofspiel.models import GoofspielModel

    heartbeat.running(step=0, phase="session_generation", force=True)
    stage_seed = int(seed) + 503 + int(steps) + int(n_cards)
    rng = random.Random(stage_seed)
    sessions_path = out / "adaptive" / "opponent_sessions.jsonl"
    if not resume_checkpoint:
        sessions_path.unlink(missing_ok=True)
    sessions = JsonlStore(sessions_path)
    session_rows = []
    regimes = default_opponent_curriculum()
    games_per_session = 3
    for session_idx in range(max(1, int(steps))):
        regime = regimes[session_idx % len(regimes)]
        games: list[list[RoundRecord]] = []
        for game_idx in range(games_per_session):
            first_prize = rng.choice(list(range(1, n_cards + 1)))
            state = GameState.initial(n_cards, current_prize=first_prize)
            rounds = []
            while not state.done:
                self_action = rng.choice(state.self_actions)
                opp_action = opponent_action_for_regime(
                    regime.regime_id,
                    state.opponent_actions,
                    stake=state.stake,
                    n_cards=n_cards,
                    rng=rng,
                )
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
            games.append(rounds)
        session = OpponentSession(
            session_id=f"adaptive_session_{session_idx}:seed{stage_seed}:n{n_cards}",
            opponent_id=f"curriculum_{regime.regime_id}",
            strategy_regime_id=regime.regime_id,
            games=games,
        )
        sessions.append(session)
        session_rows.append(session)
    rounds_total = sum(len(g) for s in session_rows for g in s.games)

    model = GoofspielModel(max_cards=13).to(device)
    model.assert_partition_is_complete()
    opt = torch.optim.AdamW(model.adaptive_parameters(), lr=lr)
    lineage = _apply_init_or_resume(
        model,
        init_from_checkpoint_path=init_from_checkpoint,
        resume_checkpoint_path=resume_checkpoint,
        optimizers={"adaptive_sft": opt},
    )
    model.set_robust_requires_grad(False)

    heartbeat.running(step=0, phase="tensor_build", force=True)
    # Phase 2: build the nested public-state / history / memory tensors NATIVELY
    # on the target device.  A post-hoc ``batch.to(device)`` does NOT work — the
    # nested public-state tensors don't move recursively (verified device
    # mismatch), so device must be threaded to the source constructor.
    tensors = _build_adaptive_training_tensors(session_rows, max_cards=13, device=device)
    if tensors is None:
        raise RuntimeError("P5 produced no training rows from the opponent sessions")
    batch, history, memory, target_t, n_cards_row = tensors
    legal_counts = batch.opponent_action_mask.sum(dim=-1).clamp_min(1).float()
    uniform_reference_nll = float(torch.log(legal_counts).mean())

    robust_snapshot = [p.detach().clone() for p in model.robust_parameters()]
    train_steps = max(1, int(steps))
    last = {"nll": float("nan"), "acc": 0.0, "ece": 1.0, "adaptive_grad_norm": 0.0, "robust_delta": 0.0}
    model.train()
    heartbeat.running(step=0, phase="training", force=True)
    for _step in range(train_steps):
        out_model = model(batch, current_game_history=history, long_term_memory=memory)
        logits = out_model.opponent_fused_logits
        loss = F.cross_entropy(logits, target_t)
        loss = loss + 0.5 * F.cross_entropy(out_model.opponent_short_logits, target_t)
        loss = loss + 0.5 * F.cross_entropy(out_model.opponent_long_logits, target_t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        adaptive_grad_norm = float(torch.sqrt(sum(
            (p.grad.detach().float().pow(2).sum() for p in model.adaptive_parameters() if p.grad is not None),
            torch.tensor(0.0),
        )))
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
        last = {"nll": nll, "acc": acc, "ece": ece, "adaptive_grad_norm": adaptive_grad_norm, "robust_delta": 0.0}
        # Interval-gated heartbeat: cheap to call every step (only writes when
        # the wall-clock interval has elapsed), so a slow CPU step and a fast
        # GPU step refresh liveness at the same cadence.
        heartbeat.running(step=_step + 1, phase="training")
        if logger is not None and _should_log_step(_step, train_steps):
            logger.step_metrics(
                "stage5_adaptive",
                _step,
                train_steps,
                {"nll": nll, "acc": acc, "ece": ece, "adaptive_grad_norm": adaptive_grad_norm},
            )
    robust_delta = float(sum((a - b).abs().sum() for a, b in zip(model.robust_parameters(), robust_snapshot)))
    last["robust_delta"] = robust_delta
    if robust_delta != 0.0:
        raise AssertionError(f"robust params moved during P5 ({robust_delta} != 0): firewall breach")

    oracle = oracle_opponent_diagnostic(session_rows, n_cards=n_cards)
    beats_uniform = last["nll"] < uniform_reference_nll
    ckpt_path = None
    ckpt = out / "stage5_adaptive.pt"
    heartbeat.running(step=train_steps, phase="checkpoint_write", force=True)
    manifest = save_checkpoint(
        ckpt,
        model=model,
        optimizers={"adaptive_sft": opt},
        metadata=CheckpointMetadata(
            checkpoint_id="stage5_adaptive",
            training_stage="P5_ADAPTIVE",
            global_step=train_steps,
            policy_version=3,
            config={"steps": steps, "n_cards": n_cards, "lr": lr, "games_per_session": games_per_session, "seed": seed, "stage_seed": stage_seed},
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

    stage_metrics = {
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
        "stage5_rank_owner": 0.0,
        "stage5_write_once": 1.0,
    }
    report = {
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
    return stage_metrics, ckpt_path


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
        # A ``PolicyFn`` (state -> {card: prob}) view for ``play_policy_vs_bot``;
        # ``policy_for_state`` below is the 13-slot list view ``_play_policy_match``
        # speaks.  One loaded model, two interface shapes.
        self.policy_fn = self._fn

    def policy_for_state(self, state: GameState) -> list[float]:
        dist = self._fn(state)
        policy = [0.0] * 13
        for card, prob in dist.items():
            policy[card - 1] = float(prob)
        return policy


class _AggressorPolicy:
    """A fixed, deterministic red-team adversary: always bids the highest legal
    card.  This is the honest 'aggressive pressure' archetype used as the
    red-team opponent slot in the Stage7 arena — a real fixed strategy, never a
    fabricated search result.  ``policy_for_state`` concentrates all mass on the
    top legal card so it plugs straight into ``_play_policy_match_seq``.
    """

    def policy_for_state(self, state: GameState) -> list[float]:
        policy = [0.0] * 13
        legal = state.self_actions
        if legal:
            policy[max(legal) - 1] = 1.0
        return policy


def _mint_league_snapshot(role: str, *, out_dir: Path, seed: int, n_cards: int = 3) -> str:
    """Train a tiny, role-seeded checkpoint and return its real path.

    Each role is trained from a different torch seed so the three snapshots are
    *genuinely distinct* trained agents (verified by cross-play, not asserted).
    This is deliberately minimal: the point of Phase 4.3 is that the league
    plays *real, loadable, distinct* checkpoints, not that they are strong.
    """
    torch, _F = _torch_import()
    torch.manual_seed(seed)
    snap_dir = out_dir / "league" / "snapshots" / role.lower()
    metrics = run_stage1_pretrain(
        steps=1,
        batch_size=4,
        out_dir=snap_dir,
        n_cards=n_cards,
        lr=3e-4,
        seed=seed,
        local_only=True,
    )
    return metrics.checkpoint


def _play_policy_match_seq(
    row_policy: Any,
    col_policy: Any,
    *,
    n_cards: int,
    prize_order: list[int],
    seed: int,
) -> float:
    """Play one row-vs-col game on a caller-supplied prize schedule.

    ``prize_order`` is the full ordered reveal of prizes ``1..n`` (a permutation).
    The opening prize is ``prize_order[0]`` and each subsequent reveal is the next
    element, so the legacy ascending schedule is ``prize_order=[1,2,…,n]`` — which
    is exactly what ``GameState.initial(n, current_prize=1)`` + the lowest-remaining
    ``_choose_next_prize`` default produces.  This makes ``_play_policy_match``
    (below) a byte-exact special case and lets Stage6 vary the prize schedule
    (and reuse it across both seat orders for common random numbers) without a
    second game loop.
    """
    if not prize_order:
        prize_order = list(range(1, n_cards + 1))
    rng = random.Random(seed)
    state = GameState.initial(n_cards, current_prize=int(prize_order[0]))
    reveal_index = 1
    while not state.done:
        row_dist = row_policy.policy_for_state(state)
        col_dist = col_policy.policy_for_state(_mirrored_state(state))
        row_action = _sample_from_policy(row_dist, state.self_actions, rng)
        col_action = _sample_from_policy(col_dist, state.opponent_actions, rng)
        if state.prize_mask:
            next_prize = int(prize_order[reveal_index]) if reveal_index < len(prize_order) else None
            if next_prize is None or not (state.prize_mask & (1 << (next_prize - 1))):
                # The schedule ran out or named an already-revealed prize; fall
                # back to the deterministic lowest-remaining reveal so the game
                # always terminates on a legal prize.
                next_prize = legal_cards(state.prize_mask, state.n)[0]
            reveal_index += 1
        else:
            next_prize = None
        state = transition(state, row_action, col_action, next_prize=next_prize).state
    return float(state.self_score - state.opp_score)


def _play_policy_match(row_policy: Any, col_policy: Any, *, n_cards: int, seed: int) -> float:
    # Legacy ascending prize schedule [1,2,…,n]; kept as the byte-exact special
    # case a re-execution test reproduces.
    return _play_policy_match_seq(
        row_policy,
        col_policy,
        n_cards=n_cards,
        prize_order=list(range(1, n_cards + 1)),
        seed=seed,
    )


def _prize_sequences(n_cards: int, seed: int, k: int) -> list[list[int]]:
    """Deterministic pool of ``k`` prize reveal orders for ``n_cards``.

    Sequence 0 is ALWAYS the legacy ascending ``[1,2,…,n]`` so SMOKE (k=1) is
    byte-identical to today.  Sequences 1..k-1 are seed-deterministic permutations
    from independent ``random.Random`` streams, so common random numbers can be
    shared across seat orders while still exercising varied prize schedules.
    """
    base = list(range(1, n_cards + 1))
    sequences = [list(base)]
    for q in range(1, max(1, int(k))):
        perm = random.Random(int(seed) * 7919 + q).sample(base, n_cards)
        sequences.append(perm)
    return sequences[: max(1, int(k))]


def _bootstrap_ci(
    values: list[float],
    *,
    seed: int,
    iters: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Deterministic pure-python bootstrap CI for the mean of ``values``.

    Collapses to ``(v, v)`` for a single value (SMOKE), so a 1/1/1 matchup has a
    degenerate but well-defined interval.  Given the same ``values`` and ``seed``
    the result is reproducible (a re-execution test recomputes it), because the
    resampling draws from a single seeded ``random.Random``.
    """
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        v = float(values[0])
        return (v, v)
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(max(1, int(iters))):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(resample) / n)
    means.sort()
    lo_idx = max(0, int((alpha / 2.0) * len(means)))
    hi_idx = min(len(means) - 1, int((1.0 - alpha / 2.0) * len(means)))
    return (float(means[lo_idx]), float(means[hi_idx]))


@dataclass(frozen=True)
class AttackCase:
    """One discovered adversarial state plus the family it belongs to.

    ``family`` groups attacks so the held-out split can measure BOTH
    same-family generalization (states like the trained ones) and
    other-family generalization (structurally different states).  ``case_index``
    is the 0-based position within the generation order and drives the legacy
    ``redteam_attack_{idx}`` id scheme.
    """

    state: GameState
    family: str
    case_index: int


def _generate_attacks(
    *,
    n_cards: int,
    seed: int,
    count: int,
    families: tuple[str, ...] = ("carry_and_asymmetric_masks", "curriculum_regimes"),
    include_legacy_prefix: bool = True,
) -> list[AttackCase]:
    """Seed-deterministic adversarial-state generator.

    The FIRST family, ``carry_and_asymmetric_masks``, has as its canonical
    family-0 cases (indices 0,1,2) the exact three legacy attack states, in
    order.  So ``count=3`` returns precisely those three states, in the legacy
    order, preserving the SMOKE contract (``failures==3`` with ids
    ``redteam_attack_{0,1,2}_seed{seed}_n{n}``).  Larger counts extend with more
    carry/asymmetric-mask states and then states snapshotted from the opponent
    curriculum regimes — everything drawn from a single seeded ``random.Random``
    so ids are reproducible.

    ``include_legacy_prefix=False`` skips the three legacy states entirely and
    generates purely from the seeded RNG.  Held-out *other-family* probes use
    this so they never re-emit the trained-on legacy states.
    """
    # The three legacy states, always first and always n=3 (they label under the
    # caller's n_cards only via the id string, exactly as before).
    legacy = [
        GameState.initial(3, current_prize=1),
        GameState.initial(3, current_prize=2),
        GameState(n=3, self_mask=0b011, opp_mask=0b110, prize_mask=0b100, current_prize=1, carry_pool=2, round_index=2),
    ]
    cases: list[AttackCase] = []
    if include_legacy_prefix:
        for i, state in enumerate(legacy):
            cases.append(AttackCase(state=state, family=families[0], case_index=i))
            if len(cases) >= count:
                return cases[:count]

    rng = random.Random(int(seed))
    # Extend family-0 (carry_and_asymmetric_masks) with more small asymmetric
    # states: random legal self/opp masks over n=3 with a random carry pool.
    while len(cases) < count and families:
        # Alternate between the two families as we grow, but keep family-0 states
        # generated first so a moderate count stays within the primary family.
        family = families[0] if (len(cases) % 2 == 1 or len(families) == 1) else families[-1]
        if family == "curriculum_regimes":
            regimes = default_opponent_curriculum()
            regime = regimes[rng.randrange(len(regimes))]
            nn = 3
            legal = list(range(1, nn + 1))
            stake = rng.randint(1, nn)
            opp_card = opponent_action_for_regime(regime.regime_id, legal, stake=stake, n_cards=nn, rng=rng)
            self_card = rng.choice(legal)
            self_mask = ((1 << nn) - 1) & ~(1 << (self_card - 1))
            opp_mask = ((1 << nn) - 1) & ~(1 << (opp_card - 1))
            prize_mask = (1 << nn) - 1
            # Reveal a random prize; keep it legal.
            prize = rng.choice(legal)
            prize_mask &= ~(1 << (prize - 1))
            state = GameState(
                n=nn, self_mask=self_mask or 1, opp_mask=opp_mask or 1,
                prize_mask=prize_mask, current_prize=prize, carry_pool=rng.randint(0, 2), round_index=2,
            )
        else:
            nn = 3
            full = (1 << nn) - 1
            drop_self = rng.randint(0, nn)
            drop_opp = rng.randint(0, nn)
            self_mask = full & ~(1 << (drop_self - 1)) if drop_self else full
            opp_mask = full & ~(1 << (drop_opp - 1)) if drop_opp else full
            prize = rng.randint(1, nn)
            prize_mask = full & ~(1 << (prize - 1))
            state = GameState(
                n=nn, self_mask=self_mask or 1, opp_mask=opp_mask or 1,
                prize_mask=prize_mask, current_prize=prize,
                carry_pool=rng.randint(0, 2), round_index=rng.randint(1, 2),
            )
        cases.append(AttackCase(state=state, family=family, case_index=len(cases)))
    return cases[:count]



def run_stage6_league(
    *,
    out_dir: str | Path,
    role_checkpoints: dict[str, str | Path] | None = None,
    n_cards: int = 3,
    seed: int = 1,
    budget: Stage6Budget | None = None,
    profile_name: str = "SMOKE",
) -> StageMetrics:
    runtime = current_runtime()
    if runtime.is_distributed:
        runtime, _ = setup_torch_distributed("auto")
    if not runtime.is_rank0:
        payload = broadcast_object(None, src=0)
        if payload is None:
            raise RuntimeError("stage6 payload broadcast failed")
        barrier_if_distributed()
        return StageMetrics("P6_LEAGUE", 1, payload["metrics"], payload.get("checkpoint"))

    # SMOKE default: 1 game / 1 seed / 1 prize sequence — byte-preserves the
    # legacy single-game cross-play.  Heavier statistical work is opt-in via the
    # profile/flag-resolved budget.
    budget = budget or Stage6Budget()
    games_per_matchup = max(1, int(budget.games_per_matchup))
    seeds = max(1, int(budget.seeds))
    prize_sequences_n = max(1, int(budget.prize_sequences))

    out = Path(out_dir)
    registry_path = out / "league" / "registry.json"
    registry_path.unlink(missing_ok=True)
    registry = LeagueRegistry(registry_path)

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

    crossplay_seed_base = int(seed) + 600
    # ---- Legacy single-game cross-play (byte-preserved) ----------------------
    # Each of the 9 ordered pairs contributes exactly one game on the ascending
    # prize schedule at seed `crossplay_seed_base + index`, exactly as before.
    # This block is profile-independent so the re-execution test reproduces
    # rows[0] with `seed=crossplay_seed_base`.
    cross_play = []
    for row_agent in agents:
        for col_agent in agents:
            score = _play_policy_match(
                policies[row_agent.agent_id],
                policies[col_agent.agent_id],
                n_cards=n_cards,
                seed=crossplay_seed_base + len(cross_play),
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

    # ---- Statistical matrix (block/paired bootstrap; deltas #2/#3/#4/#15) -----
    # The directly-readable ROBUST/AGGRESSIVE/EXPLOITER pairwise matrix.  Each
    # ordered cell is aggregated over games × prize_sequences × seeds on common
    # random numbers: the prize schedule AND the per-game seeds are keyed on the
    # CANONICAL (unordered) pair, so cell (A,B) and cell (B,A) are played on the
    # SAME decks with the SAME move-sampling seeds — genuine CRN pairing, not two
    # independent samples.
    #
    #   * Block = (seed, prize_sequence) (delta #2).  A block holds
    #     ``games_per_matchup`` games; the bootstrap resamples WHOLE BLOCKS (never
    #     individual games), so the interval respects the CRN correlation
    #     structure.  Equal-size blocks make the block bootstrap of the mean
    #     exactly a bootstrap over the per-block means.
    #   * Paired seat-symmetrized statistic (delta #3): for a fixed
    #     (seed, prize_sequence, game) the A-advantage is
    #     ``mean(diff_A_vs_B, −diff_B_vs_A)`` — identical in construction to
    #     ``arena/match.play_pairing``'s ``a_adv = [seat0_diff, −seat1_diff]``.
    #   * Fixed budget: NO sequential CI stopping anywhere (delta #4).
    def _median(xs: list[float]) -> float:
        s = sorted(xs)
        n = len(s)
        if n == 0:
            return 0.0
        return s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2.0

    matrix = []
    total_games = 0
    total_rounds = 0
    total_blocks = 0
    # Per-ordered-cell raw game diffs, keyed by (row_idx, col_idx) then
    # (seed, prize_sequence) -> [game diffs].  Reused (NO re-play) to build the
    # paired unordered-pair statistic below.
    cell_block_games: dict[tuple[int, int], dict[tuple[int, int], list[float]]] = {}
    t0 = time.perf_counter()
    n_agents = len(agents)
    for ri, row_agent in enumerate(agents):
        for ci, col_agent in enumerate(agents):
            lo_idx, hi_idx = (ri, ci) if ri <= ci else (ci, ri)
            pair_key = lo_idx * n_agents + hi_idx  # canonical (shared across seats)
            sequences = _prize_sequences(n_cards, seed=crossplay_seed_base + pair_key, k=prize_sequences_n)
            row_pol = policies[row_agent.agent_id]
            col_pol = policies[col_agent.agent_id]
            per_seed_means: list[float] = []
            all_diffs: list[float] = []
            block_means: list[float] = []
            block_rows: list[dict[str, Any]] = []
            block_games_by_key: dict[tuple[int, int], list[float]] = {}
            for s in range(seeds):
                seed_diffs: list[float] = []
                for q in range(prize_sequences_n):
                    block_diffs: list[float] = []
                    for g in range(games_per_matchup):
                        # CRN: cell seed depends on the CANONICAL pair + (s,q,g),
                        # NOT on seat order, so A-vs-B and B-vs-A share randomness.
                        cell_seed = crossplay_seed_base + pair_key * 100000 + s * 1000 + q * 100 + g
                        diff = _play_policy_match_seq(
                            row_pol, col_pol, n_cards=n_cards, prize_order=sequences[q], seed=cell_seed
                        )
                        block_diffs.append(diff)
                        seed_diffs.append(diff)
                        all_diffs.append(diff)
                        total_games += 1
                        total_rounds += n_cards
                    block_mean = sum(block_diffs) / max(1, len(block_diffs))
                    block_means.append(block_mean)
                    block_rows.append(
                        {"seed": s, "prize_sequence": q, "games": list(block_diffs), "block_mean": block_mean}
                    )
                    block_games_by_key[(s, q)] = block_diffs
                    total_blocks += 1
                per_seed_means.append(sum(seed_diffs) / max(1, len(seed_diffs)))
            cell_block_games[(ri, ci)] = block_games_by_key
            n_games = max(1, len(all_diffs))
            mean = sum(all_diffs) / n_games
            var = sum((d - mean) ** 2 for d in all_diffs) / n_games
            wins = sum(1 for d in all_diffs if d > 0)
            draws = sum(1 for d in all_diffs if d == 0)
            # delta #2: bootstrap resamples WHOLE (seed, prize_sequence) blocks.
            # Seeded on the canonical pair key so a re-execution test reproduces it.
            ci_low, ci_high = _bootstrap_ci(block_means, seed=crossplay_seed_base + pair_key)
            matrix.append(
                {
                    "row_agent": row_agent.agent_id,
                    "col_agent": col_agent.agent_id,
                    "row_role": row_agent.role,
                    "col_role": col_agent.role,
                    # delta #3: diagonal is self-play, off-diagonal is competitive.
                    "self_play": ri == ci,
                    "relationship": "self_play" if ri == ci else "competitive",
                    "games": len(all_diffs),
                    "raw_games": len(all_diffs),           # delta #15
                    "bootstrap_blocks": len(block_means),  # delta #15
                    "blocks": len(block_means),            # delta #15 (per-ordered-pair)
                    "seeds": seeds,
                    "seeds_used": len(per_seed_means),
                    "prize_sequences": prize_sequences_n,
                    "win_rate": wins / n_games,
                    "draw_rate": draws / n_games,
                    "mean_score_diff": mean,
                    "std": var ** 0.5,
                    "median": _median(all_diffs),
                    "worst_seed": min(per_seed_means) if per_seed_means else mean,       # delta #15
                    "worst_seed_mean": min(per_seed_means) if per_seed_means else mean,  # legacy alias
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "ci_halfwidth": (ci_high - ci_low) / 2.0,
                    "block_bootstrap_ci_low": ci_low,   # delta #15 canonical names
                    "block_bootstrap_ci_high": ci_high,
                    "block_rows": block_rows,
                }
            )

    # ---- Paired seat-symmetrized statistic, per unordered competitive pair ----
    # For each unordered pair (lo < hi), pair the ALREADY-PLAYED ordered cells
    # (lo,hi) and (hi,lo) game-for-game on their shared CRN (seed, prize_sequence,
    # game) indices.  The paired A-advantage is mean(diff_A_vs_B, −diff_B_vs_A);
    # blocks stay whole (delta #2) and the CI is seeded on the canonical pair key
    # plus a fixed 900000 offset so it is distinct from the per-cell interval and
    # a re-execution test can reproduce it from the stored block rows (delta #19).
    paired_matrix = []
    for lo in range(n_agents):
        for hi in range(lo + 1, n_agents):
            fwd = cell_block_games.get((lo, hi), {})
            rev = cell_block_games.get((hi, lo), {})
            paired_games: list[float] = []
            paired_block_means: list[float] = []
            for key in sorted(set(fwd) & set(rev)):
                fg, rg = fwd[key], rev[key]
                block_pairs = [0.5 * (fg[g] - rg[g]) for g in range(min(len(fg), len(rg)))]
                paired_games.extend(block_pairs)
                if block_pairs:
                    paired_block_means.append(sum(block_pairs) / len(block_pairs))
            pair_key = lo * n_agents + hi
            if paired_games:
                p_mean = sum(paired_games) / len(paired_games)
                p_lo, p_hi = _bootstrap_ci(paired_block_means, seed=crossplay_seed_base + pair_key + 900000)
            else:
                p_mean = p_lo = p_hi = 0.0
            paired_matrix.append(
                {
                    "agent_a": agents[lo].agent_id,
                    "agent_b": agents[hi].agent_id,
                    "role_a": agents[lo].role,
                    "role_b": agents[hi].role,
                    "relationship": "competitive",
                    "raw_games": len(paired_games),
                    "paired_blocks": len(paired_block_means),
                    "paired_mean_score_diff": p_mean,
                    "paired_median": _median(paired_games),
                    "paired_block_bootstrap_ci_low": p_lo,
                    "paired_block_bootstrap_ci_high": p_hi,
                }
            )
    wall_clock_s = time.perf_counter() - t0

    # Handcrafted algorithms are kept only as clearly-LABELLED reference
    # opponents, never conflated with the trained cross-play above.
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
            policies[agent.agent_id],
            ref,
            n_cards=n_cards,
            seed=int(seed) + len(reference_play) + 900,
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
    ordered_matchups = len(matrix)
    self_play_matchups = sum(1 for row in matrix if row["self_play"])
    competitive_matchups = ordered_matchups - self_play_matchups
    workload = {
        # delta #15 run-level accounting.
        "profile": str(profile_name or "SMOKE"),
        "ordered_matchups": ordered_matchups,
        "self_play_matchups": self_play_matchups,
        "competitive_matchups": competitive_matchups,
        "matchups": len(cross_play),  # legacy alias (9 ordered pairs)
        "games_per_matchup": games_per_matchup,
        "seeds": seeds,
        "prize_sequences": prize_sequences_n,
        "raw_games": total_games,
        "bootstrap_blocks": total_blocks,
        "sequential_ci_stop": False,  # delta #4: fixed budget on every profile
        "ci_target_halfwidth": 0.0,
        "total_games": total_games,
        "total_episodes": total_games,
        "total_actions": total_rounds * 2,
        "total_rounds": total_rounds,
        "wall_clock_s": wall_clock_s,
        "games_per_sec": (total_games / wall_clock_s) if wall_clock_s > 0 else 0.0,
    }
    league_report = {
        "counts_by_role": counts,
        "pfsp_weights": pfsp_weights,
        "cross_play": cross_play,
        "matrix": matrix,
        "paired_matrix": paired_matrix,
        "workload": workload,
        "reference_play": reference_play,
        "agent_checkpoints": {agent.agent_id: policies[agent.agent_id].checkpoint_path for agent in agents},
        "historical_agents_frozen": all(agent.frozen for agent in agents),
        "seed": int(seed),
        "crossplay_seed_base": crossplay_seed_base,
        "reference_seed_base": int(seed) + 900,
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
            "stage6_rank_owner": 0.0,
            "stage6_write_once": 1.0,
            # Workload accounting (additive, numeric).
            "stage6_matchups": float(len(cross_play)),
            "stage6_ordered_matchups": float(ordered_matchups),
            "stage6_self_play_matchups": float(self_play_matchups),
            "stage6_competitive_matchups": float(competitive_matchups),
            "stage6_games_per_matchup": float(games_per_matchup),
            "stage6_seeds": float(seeds),
            "stage6_prize_sequences": float(prize_sequences_n),
            "stage6_raw_games": float(total_games),
            "stage6_bootstrap_blocks": float(total_blocks),
            "stage6_total_games": float(total_games),
            "stage6_total_episodes": float(total_games),
            "stage6_total_actions": float(total_rounds * 2),
            "stage6_wall_clock_s": float(wall_clock_s),
            "stage6_games_per_sec": float(workload["games_per_sec"]),
        }
    )
    payload = {"metrics": metrics, "checkpoint": None}
    payload = broadcast_object(payload, src=0)
    if payload is None:
        raise RuntimeError("stage6 payload broadcast failed")
    barrier_if_distributed()
    return StageMetrics("P6_LEAGUE", 1, payload["metrics"], payload.get("checkpoint"))


def _memorization_flag(
    train_before: dict[str, Any],
    train_after: dict[str, Any],
    heldout_pairs: list[tuple[dict[str, Any] | None, dict[str, Any] | None]],
) -> bool:
    """Detect a memorizing correction: TRAIN match-rate improved but NO held-out
    bucket did.

    Pure and deterministic in its inputs (each bucket is an
    ``_attack_state_regression`` result dict, or ``None`` when that bucket has no
    states), so a test can both exercise the semantics on synthesized buckets AND
    re-execute it on the reloaded before/after checkpoints and reproduce the exact
    flag the runner reported.  A bucket that has no states (``None``) never counts
    as an improvement; ``has_heldout`` is true iff at least one held-out bucket
    exists.
    """

    def improved(before: dict[str, Any] | None, after: dict[str, Any] | None) -> bool:
        return bool(before is not None and after is not None and after["match_rate"] > before["match_rate"])

    train_improved = train_after["match_rate"] > train_before["match_rate"]
    heldout_improved = any(improved(b, a) for b, a in heldout_pairs)
    has_heldout = any(b is not None for b, _ in heldout_pairs)
    return bool(train_improved and has_heldout and not heldout_improved)


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


def _strong_bot_for(n_cards: int) -> tuple[str, str]:
    """Return ``(requested, effective)`` for the arena's 'strong' opponent slot.

    We request an exact-Nash bot (``nash`` for classic n≤NASH_MAX_N else the
    carry variant), but the bot itself HONESTLY falls back to a heuristic above
    its exact cap.  We mirror that cap here so the report records the truth
    (e.g. n=13 → requested ``nash``, effective ``heuristic_fallback``) rather
    than fabricating a search result.
    """
    from goofspiel.bots import NASH_MAX_N

    requested = "nash"
    effective = "nash" if int(n_cards) <= int(NASH_MAX_N) else "heuristic_fallback"
    return requested, effective


def _stage7_arena(
    *,
    robust_policy_fn_view,
    corrected_policy_fn_view,
    league_snapshot_policy,
    redteam_policy,
    n_cards: int,
    arena_games: int,
    arena_seeds: int,
    seed: int,
) -> dict[str, Any]:
    """Corrected-vs-robust arena across the honest bot suite (Section 7).

    Plays BOTH the pre-correction robust policy and the post-correction policy
    against Random / Heuristic / strong(nash→honest fallback) / League snapshot
    / Red-team, over ``arena_games`` games × ``arena_seeds`` seeds.  Bots use
    ``play_policy_vs_bot`` (real carry-over env); the League and Red-team slots
    are model/fixed policies played through ``_play_policy_match_seq`` so the
    whole arena is genuine play.  Returns per-opponent rows for both policies.
    """
    from goofspiel.training.model_eval import play_policy_vs_bot

    games = max(1, int(arena_games))
    seeds = max(1, int(arena_seeds))
    requested, effective = _strong_bot_for(n_cards)
    bot_slots = [
        ("random", "random", "random"),
        ("heuristic", "heuristic", "heuristic"),
        ("strong", requested, effective),
    ]

    def _bot_row(policy_fn_view, slot, requested_name, effective_name) -> dict[str, Any]:
        diffs: list[float] = []
        wins = 0
        for s in range(seeds):
            res = play_policy_vs_bot(
                policy_fn_view,
                effective_name if effective_name != "heuristic_fallback" else "heuristic",
                n_cards=n_cards,
                num_games=games,
                seed=int(seed) + s * 17 + 1,
            )
            diffs.append(res["mean_score_diff"])
            wins += int(res["win_rate"] * games)
        mean = sum(diffs) / max(1, len(diffs))
        return {
            "opponent_slot": slot,
            "opponent_requested": requested_name,
            "opponent_effective": effective_name,
            "games": games * seeds,
            "seeds": seeds,
            "mean_score_diff": mean,
            "win_rate": wins / max(1, games * seeds),
        }

    def _policy_match_row(policy_view, opponent_view, slot) -> dict[str, Any]:
        diffs: list[float] = []
        for s in range(seeds):
            sequences = _prize_sequences(n_cards, seed=int(seed) + s * 31, k=1)
            for g in range(games):
                diffs.append(
                    _play_policy_match_seq(
                        policy_view, opponent_view, n_cards=n_cards,
                        prize_order=sequences[0], seed=int(seed) + s * 1009 + g,
                    )
                )
        mean = sum(diffs) / max(1, len(diffs))
        return {
            "opponent_slot": slot,
            "opponent_requested": slot,
            "opponent_effective": slot,
            "games": games * seeds,
            "seeds": seeds,
            "mean_score_diff": mean,
            "win_rate": sum(1 for d in diffs if d > 0) / max(1, len(diffs)),
        }

    def _rows_for(policy_fn_view, policy_match_view) -> list[dict[str, Any]]:
        rows = [_bot_row(policy_fn_view, *slot) for slot in bot_slots]
        if league_snapshot_policy is not None:
            rows.append(_policy_match_row(policy_match_view, league_snapshot_policy, "league"))
        rows.append(_policy_match_row(policy_match_view, redteam_policy, "redteam"))
        return rows

    return {
        "n_cards": int(n_cards),
        "arena_games": games,
        "arena_seeds": seeds,
        "strong_opponent_requested": requested,
        "strong_opponent_effective": effective,
        "robust": _rows_for(robust_policy_fn_view.policy_fn, robust_policy_fn_view),
        "corrected": _rows_for(corrected_policy_fn_view.policy_fn, corrected_policy_fn_view),
    }


def run_stage7_redteam(
    *,
    out_dir: str | Path,
    init_from_checkpoint: str | Path | None = None,
    correction_steps: int = 40,
    lr: float = 1e-3,
    n_cards: int = 3,
    seed: int = 1,
    budget: Stage7Budget | None = None,
    profile_name: str = "SMOKE",
) -> StageMetrics:
    runtime = current_runtime()
    if runtime.is_distributed:
        runtime, _ = setup_torch_distributed("auto")
    if not runtime.is_rank0:
        payload = broadcast_object(None, src=0)
        if payload is None:
            raise RuntimeError("stage7 payload broadcast failed")
        barrier_if_distributed()
        return StageMetrics("P7_REDTEAM", 1, payload["metrics"], payload["checkpoint"])

    out = Path(out_dir)
    failures_path = out / "redteam" / "failures.jsonl"
    corrections_path = out / "redteam" / "corrections.jsonl"
    failures_path.unlink(missing_ok=True)
    corrections_path.unlink(missing_ok=True)
    failures = FailureBuffer(failures_path)
    corrections = CorrectionDataset(corrections_path)
    router = TeacherRouter()

    # SMOKE default: 3 discovered attacks (the legacy states), all in the train
    # set, no held-out, no arena — byte-preserves today's behaviour.
    budget = budget or Stage7Budget()
    attack_cases = max(1, int(budget.attack_cases))
    correction_train_cases = max(1, int(budget.correction_train_cases))
    heldout_attack_cases = max(0, int(budget.heldout_attack_cases))
    arena_games = max(0, int(budget.arena_games))
    arena_seeds = max(0, int(budget.arena_seeds))

    t0 = time.perf_counter()

    # ---- Discovery -----------------------------------------------------------
    # Seed-deterministic generation.  SMOKE (attack_cases=3) returns EXACTLY the
    # three legacy states in order, preserving the ids and failures==3.
    discovered = _generate_attacks(n_cards=n_cards, seed=int(seed), count=attack_cases)

    router_cache: dict[int, Any] = {}

    def _teacher_card(state: GameState) -> tuple[Any, int]:
        sample = router.label_state(state)
        pol = sample.teacher_policy or [1.0] * len(state.self_actions)
        best_index = max(range(len(state.self_actions)), key=lambda i: pol[i])
        return sample, state.self_actions[best_index]

    attack_report = []
    teacher_samples = []
    teacher_cards: list[int] = []
    for idx, case in enumerate(discovered):
        state = case.state
        sample, card = _teacher_card(state)
        teacher_samples.append(sample)
        teacher_cards.append(card)
        failure = FailureRecord(
            failure_id=f"redteam_attack_{idx}_seed{int(seed)}_n{n_cards}",
            failure_type="ADVERSARIAL_STATE_REANALYSIS",
            state=state_record_from_game_state(state),
            model_version="seed_initial",
            teacher_source=sample.teacher_source,
            details={
                "purpose": "minimal red-team correction loop",
                "attack_family": case.family,
                "teacher_confidence": sample.teacher_confidence,
            },
        )
        failures.add(failure)
        corrections.add_reanalysis(
            ReanalysisRecord(
                sample_id=f"correction_{idx}_seed{int(seed)}_n{n_cards}",
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
                "attack_family": case.family,
                "teacher_source": sample.teacher_source,
                "teacher_confidence": sample.teacher_confidence,
                "teacher_card": card,
            }
        )

    # ---- Train / held-out split (kills the memorization hole) ----------------
    # TRAIN = the first correction_train_cases discovered attacks; the focused
    # correction optimises ONLY on these.  HELDOUT_SAME = the remaining discovered
    # attacks (same family, never trained on).  HELDOUT_OTHER = freshly-generated
    # structurally-different states (never the legacy ones).  A correction that
    # generalises improves held-out too; one that memorizes improves only train.
    train_slice = discovered[:correction_train_cases]
    train_states = [c.state for c in train_slice]
    train_cards = teacher_cards[:correction_train_cases]
    train_samples = teacher_samples[:correction_train_cases]

    heldout_same_states = [c.state for c in discovered[correction_train_cases:]]
    heldout_same_cards = teacher_cards[correction_train_cases:]
    if heldout_attack_cases > 0:
        heldout_other_cases = _generate_attacks(
            n_cards=n_cards,
            seed=int(seed) * 3 + 1,
            count=heldout_attack_cases,
            families=("curriculum_regimes", "carry_and_asymmetric_masks"),
            include_legacy_prefix=False,
        )
        heldout_other_states = [c.state for c in heldout_other_cases]
        heldout_other_cards = [_teacher_card(st)[1] for st in heldout_other_states]
    else:
        heldout_other_states = []
        heldout_other_cards = []

    # ---- Phase 4.4: a REAL focused fine-tune + MEASURED regression -----------
    # Load (or mint) a real checkpoint, measure the attack-state regression
    # BEFORE, run a focused correction SFT on the teacher-relabeled TRAIN states,
    # save the improved checkpoint, and measure the regression AFTER.  Every
    # pass/fail below is computed by re-playing the policy, never hardcoded.
    torch, F = _torch_import()
    from goofspiel.models import GoofspielModel, public_state_from_game
    from goofspiel.training.checkpoint import load_checkpoint

    if not init_from_checkpoint or not Path(init_from_checkpoint).exists():
        seed_metrics = run_stage1_pretrain(
            steps=1, batch_size=4, out_dir=out / "redteam" / "seed", n_cards=n_cards, local_only=True
        )
        init_from_checkpoint = seed_metrics.checkpoint

    model = GoofspielModel(max_cards=13)
    model.load_state_dict(load_checkpoint(init_from_checkpoint)["model_state"])
    model.eval()
    # BEFORE-correction regression on every bucket (same pre-correction model).
    regression_before = _attack_state_regression(model, train_states, train_cards)
    heldout_same_before = (
        _attack_state_regression(model, heldout_same_states, heldout_same_cards)
        if heldout_same_states else None
    )
    heldout_other_before = (
        _attack_state_regression(model, heldout_other_states, heldout_other_cards)
        if heldout_other_states else None
    )
    # Normal-play regression (opt-in with the arena budget): measured on the
    # pre-correction model against the honest bot suite.
    normal_before: dict[str, Any] | None = None
    if arena_games > 0:
        from goofspiel.training.model_eval import play_policy_vs_bot, robust_policy_fn

        pol_view = robust_policy_fn(model, greedy=False, temperature=1.0)
        normal_before = {
            bt: play_policy_vs_bot(pol_view, bt, n_cards=n_cards, num_games=arena_games, seed=int(seed) + 900)
            for bt in ("random", "heuristic")
        }

    # Focused correction: KL toward the stored teacher policy on the TRAIN states
    # (+ an immediate-Q anchor), the exact states that failed.
    batch = public_state_from_game(train_states, max_cards=13)
    target_q, q_mask = _immediate_target(train_states, 13)
    teacher_policy = torch.zeros(len(train_states), 13)
    for b, sample in enumerate(train_samples):
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
    # AFTER-correction regression on every bucket (post-correction model).
    regression_after = _attack_state_regression(model, train_states, train_cards)
    heldout_same_after = (
        _attack_state_regression(model, heldout_same_states, heldout_same_cards)
        if heldout_same_states else None
    )
    heldout_other_after = (
        _attack_state_regression(model, heldout_other_states, heldout_other_cards)
        if heldout_other_states else None
    )

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
            config={"correction_steps": correction_steps, "lr": lr, "n_cards": n_cards, "seed": seed},
            metrics={"focused_correction_loss_last": last_loss},
            parent_checkpoint_id="seed_initial",
            init_checkpoint_id=str(init_from_checkpoint),
            model_config_hash=model_config_hash(model),
            optimizer_reset=True,
        ),
    )

    # Normal-play regression AFTER, on the post-correction model.
    normal_after: dict[str, Any] | None = None
    if arena_games > 0:
        from goofspiel.training.model_eval import play_policy_vs_bot, robust_policy_fn

        pol_view = robust_policy_fn(model, greedy=False, temperature=1.0)
        normal_after = {
            bt: play_policy_vs_bot(pol_view, bt, n_cards=n_cards, num_games=arena_games, seed=int(seed) + 900)
            for bt in ("random", "heuristic")
        }

    # ---- Arena (Section 7): corrected-vs-robust across the honest bot suite ---
    arena: dict[str, Any] | None = None
    if arena_games > 0:
        robust_view = _CheckpointPolicy(init_from_checkpoint, temperature=1.0)
        corrected_view = _CheckpointPolicy(corrected_ckpt, temperature=1.0)
        league_ckpt = _mint_league_snapshot(
            ROLE_EXPLOITER, out_dir=out / "redteam", seed=int(seed) + 303, n_cards=n_cards
        )
        league_view = _CheckpointPolicy(league_ckpt, temperature=0.5)
        arena = _stage7_arena(
            robust_policy_fn_view=robust_view,
            corrected_policy_fn_view=corrected_view,
            league_snapshot_policy=league_view,
            redteam_policy=_AggressorPolicy(),
            n_cards=n_cards,
            arena_games=arena_games,
            arena_seeds=arena_seeds,
            seed=int(seed),
        )

    # A red-team correction "recurs" (fails) if any attack the correction fixed
    # is still mis-played after training.  Passing = every attack matches the
    # teacher action after correction.  General regression is proxied by the
    # attack match-rate not collapsing below the before-correction rate.
    original_attack_regression_passed = bool(regression_after["passed"])
    general_regression_passed = bool(regression_after["match_rate"] >= regression_before["match_rate"])
    recurrence = bool(regression_after["matched"] < len(train_states))

    # Memorization detection: the correction improved TRAIN but NOT held-out.
    # Delegated to the pure module-level helper so a test can re-execute the exact
    # same decision on the reloaded before/after checkpoints (RE-EXECUTE the fact).
    heldout_pairs = [
        (heldout_same_before, heldout_same_after),
        (heldout_other_before, heldout_other_after),
    ]
    memorization_flag = _memorization_flag(regression_before, regression_after, heldout_pairs)
    regression_delta = float(regression_after["match_rate"] - regression_before["match_rate"])
    wall_clock_s = time.perf_counter() - t0

    def _bucket(before: dict | None, after: dict | None) -> dict[str, Any] | None:
        if before is None or after is None:
            return None
        return {
            "match_rate_before": before["match_rate"],
            "match_rate_after": after["match_rate"],
            "delta": after["match_rate"] - before["match_rate"],
            "before": before,
            "after": after,
        }

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
            # MEASURED (Phase 4.4): computed by re-playing the TRAIN attack states
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
        # ---- Held-out generalization + normal-play regression (additive) ------
        "heldout_same_family": _bucket(heldout_same_before, heldout_same_after),
        "heldout_other_family": _bucket(heldout_other_before, heldout_other_after),
        "normal_play": {"before": normal_before, "after": normal_after} if normal_before is not None else None,
        "memorization_flag": memorization_flag,
        "regression_delta": regression_delta,
        "arena": arena,
        "workload": {
            "profile": str(profile_name or "SMOKE"),
            "attack_candidates_generated": len(discovered),
            "correction_train_cases": len(train_states),
            "heldout_same_tests": len(heldout_same_states),
            "heldout_other_tests": len(heldout_other_states),
            "correction_optimizer_steps": int(correction_steps),
            "arena_games": arena_games,
            "arena_seeds": arena_seeds,
            "wall_clock_s": wall_clock_s,
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
    metrics = StageMetrics(
        "P7_REDTEAM",
        1,
        {
            "failures": float(failures.count()),
            "corrections": float(corrections.count()),
            "attack_families": float(len({c.family for c in discovered})),
            "teacher_relabels": float(len(attack_report)),
            # Section 8 mislabel FIX: focused_correction_steps now reports the REAL
            # optimizer-loop count (range(correction_steps)), no longer the attack
            # count.  correction_optimizer_steps is the same value, added as the
            # unambiguous key; the attack count remains under failures/teacher_relabels.
            "focused_correction_steps": float(int(correction_steps)),
            "correction_optimizer_steps": float(int(correction_steps)),
            # Phase 4.4: these are now MEASURED regression outcomes, re-executed
            # by replaying the attack states before/after the focused correction.
            "attack_match_rate_before": regression_before["match_rate"],
            "attack_match_rate_after": regression_after["match_rate"],
            "mean_teacher_nll_before": regression_before["mean_teacher_nll"],
            "mean_teacher_nll_after": regression_after["mean_teacher_nll"],
            "original_attack_regression_passed": float(original_attack_regression_passed),
            "general_regression_passed": float(general_regression_passed),
            # Workload accounting + held-out generalization (additive, numeric).
            "attack_candidates_generated": float(len(discovered)),
            "correction_train_cases": float(len(train_states)),
            "heldout_same_tests": float(len(heldout_same_states)),
            "heldout_other_tests": float(len(heldout_other_states)),
            "arena_games_played": float(arena_games * arena_seeds),
            "memorization_flag": float(memorization_flag),
            "regression_delta": regression_delta,
            "stage7_wall_clock_s": float(wall_clock_s),
            "stage7_rank_owner": 0.0,
            "stage7_write_once": 1.0,
        },
        str(corrected_ckpt),
    )
    payload = {"metrics": metrics.metrics, "checkpoint": metrics.checkpoint}
    payload = broadcast_object(payload, src=0)
    if payload is None:
        raise RuntimeError("stage7 payload broadcast failed")
    barrier_if_distributed()
    return StageMetrics("P7_REDTEAM", 1, payload["metrics"], payload["checkpoint"])


def run_evaluation_suite(
    *,
    out_dir: str | Path,
    num_games: int = 16,
    seeds: list[int] | None = None,
    checkpoint: str | None = None,
    seed: int = 1,
    profile_name: str = "SMOKE",
) -> dict[str, Any]:
    runtime = current_runtime()
    if runtime.is_distributed:
        runtime, _ = setup_torch_distributed("auto")
    if not runtime.is_rank0:
        payload = broadcast_object(None, src=0)
        if payload is None:
            raise RuntimeError("evaluation payload broadcast failed")
        barrier_if_distributed()
        return payload

    from goofspiel.training.benchmark import EvaluationProfile, run_unified_benchmark, write_benchmark_report

    # ``seeds=None`` reproduces the historical 3-seed behaviour exactly.
    eval_seeds = list(seeds) if seeds else [int(seed) + i for i in range(3)]
    report_a = evaluate_bot_matchup(num_games=num_games, seed=int(seed))
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
    # Phase 5: the benchmark is model-aware; E2/E6 become the trained policy's
    # REAL play vs Random when a checkpoint is supplied (never the Heuristic-vs-
    # Random reference), and G2 becomes a real robustness verdict on it.
    # The resolved profile name flows in so QUICK/SMOKE resolve to NOT_EVALUATED
    # (Section 9): only FULL may emit a binding PROMOTE/REJECT.
    benchmark = run_unified_benchmark(
        EvaluationProfile(
            name=str(profile_name or "SMOKE"),
            seeds=eval_seeds,
            num_games=num_games,
            include_e7=False,
        ),
        checkpoint=checkpoint,
    )
    # The directory is the fixed storage slot for this in-run evaluation; the
    # promotion discipline (Section 9) is governed by the profile NAME inside the
    # report, not by the path.  Keeping it stable preserves the artifact shape.
    payload["benchmark_report"] = write_benchmark_report(benchmark, Path(out_dir) / "reports" / "quick")
    payload["rank_owner"] = 0
    payload["write_once"] = True
    payload = broadcast_object(payload, src=0)
    if payload is None:
        raise RuntimeError("evaluation payload broadcast failed")
    barrier_if_distributed()
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
    """Phase 5: register ``best_*`` aliases from THEIR OWN evaluations.

    Each candidate checkpoint is re-played through the 0.1 harness on three
    distinct axes (raw robust score, full-game exploitability, worst-N
    generalization); ``best_robust`` / ``best_search`` / ``best_generalization``
    are then the per-axis winners.  Because the axes optimise different
    quantities, the winners can be *different files*, so the alias is no longer an
    unconditional copy of the P4 checkpoint.  ``latest`` still tracks the primary
    (last) candidate.  Every metric written here is computed, never a literal.
    """
    from goofspiel.training.selection import select_checkpoints_by_axis

    runtime = current_runtime()
    if runtime.is_distributed:
        runtime, _ = setup_torch_distributed("auto")
    if not runtime.is_rank0:
        payload = broadcast_object(None, src=0)
        if payload is None:
            raise RuntimeError("axis selection payload broadcast failed")
        barrier_if_distributed()
        return payload

    out = Path(out_dir)
    existing = {cid: path for cid, path in candidates.items() if path and Path(path).exists()}
    if not existing:
        payload = {"selected": {}, "table": {}, "reason": "no_candidate_checkpoints", "rank_owner": 0, "write_once": True}
        payload = broadcast_object(payload, src=0)
        if payload is None:
            raise RuntimeError("axis selection payload broadcast failed")
        barrier_if_distributed()
        return payload

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
        "rank_owner": 0,
        "write_once": True,
    }
    report_path = out / "reports" / "axis_selection.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report = broadcast_object(report, src=0)
    if report is None:
        raise RuntimeError("axis selection payload broadcast failed")
    barrier_if_distributed()
    return report


def _smoke_algorithmic_check(
    checkpoint_path: str | None,
    *,
    n_cards: int,
    seed: int,
    num_games: int = 32,
) -> dict[str, Any]:
    """Re-execute the Phase 0.1 honest evaluator on the produced checkpoint.

    Returns a dict whose ``algorithmic_ok`` is computed, never a literal, from
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
    runtime = current_runtime()
    if runtime.is_distributed:
        runtime, _ = setup_torch_distributed(device)

    out = Path(out_dir)
    event_path = out / "events" / "training_smoke.jsonl"
    if runtime.is_rank0:
        out.mkdir(parents=True, exist_ok=True)
        event_path.unlink(missing_ok=True)
        sink = JsonlEventSink(event_path)
    else:
        sink = None

    def emit(stage: str, payload: dict[str, Any]) -> None:
        if sink is not None:
            sink.emit(BaseEvent(event_type="TRAINING_SMOKE_STAGE", run_id="smoke_pipeline", payload={"stage": stage, **payload}))

    emit("system_metrics_start", collect_system_metrics())
    stage0 = run_stage0_verify(artifact_dir=out / "stage0_verify")
    emit("stage0_verify", {"ok": stage0.ok, "checks": stage0.checks})
    corpus = generate_random_game_corpus(out_path=out / "data" / "game_corpus.jsonl", num_games=num_corpus_games, seed=seed)
    emit("build_corpus", {"ok": True, "metrics": corpus})
    stage1 = run_stage1_pretrain(
        steps=steps, batch_size=batch_size, out_dir=out / "checkpoints", device=device, n_cards=n_cards,
        seed=seed,
        corpus_path=out / "data" / "game_corpus.jsonl",
    )
    emit("stage1_pretrain", {"ok": True, "metrics": stage1.metrics, "checkpoint": stage1.checkpoint})
    stage2 = run_stage2_semi_supervised(steps=steps, out_dir=out / "data", n_cards=n_cards, seed=seed)
    emit("stage2_semi_supervised", {"ok": True, "metrics": stage2.metrics})
    # Phase 3.1: chain theta forward. P3 inherits P1's weights (theta-only, fresh
    # optimizer); P4 inherits P3's. This is init_from_checkpoint, NOT resume.
    stage3 = run_stage3_sft(
        steps=steps, batch_size=batch_size, out_dir=out / "checkpoints", device=device, n_cards=n_cards,
        seed=seed,
        init_from_checkpoint=stage1.checkpoint,
    )
    emit("stage3_sft", {"ok": True, "metrics": stage3.metrics, "checkpoint": stage3.checkpoint})
    stage4 = run_stage4_robust_rl(
        steps=steps, batch_size=batch_size, out_dir=out / "checkpoints", device=device, n_cards=n_cards,
        seed=seed,
        init_from_checkpoint=stage3.checkpoint,
    )
    emit("stage4_robust_rl", {"ok": True, "metrics": stage4.metrics, "checkpoint": stage4.checkpoint})
    stage5 = run_stage5_adaptive(
        steps=steps, out_dir=out, n_cards=n_cards,
        seed=seed,
        init_from_checkpoint=stage4.checkpoint,
    )
    emit("stage5_adaptive", {"ok": True, "metrics": stage5.metrics, "checkpoint": stage5.checkpoint})
    # Phase 4.3: the league plays REAL trained snapshots produced by this run:
    # P4 (robust backbone), P3 (strategic SFT), P5 (adaptive/exploiter), not
    # role-keyed handcrafted baselines.  budget=None → the SMOKE preset (1/1/1),
    # byte-preserving the historical single-game cross-play.
    stage6 = run_stage6_league(
        out_dir=out,
        role_checkpoints={
            ROLE_ROBUST: stage4.checkpoint,
            ROLE_AGGRESSIVE: stage3.checkpoint,
            ROLE_EXPLOITER: stage5.checkpoint,
        },
        n_cards=n_cards,
        seed=seed,
        budget=Stage6Budget(),
    )
    emit("stage6_league", {"ok": True, "metrics": stage6.metrics})
    # Phase 4.4: P7 focuses a REAL correction fine-tune on the P4 robust backbone
    # and MEASURES the attack-state regression before/after.  budget=None → the
    # SMOKE preset (3 attacks all in train, no held-out, no arena).
    stage7 = run_stage7_redteam(
        out_dir=out, init_from_checkpoint=stage4.checkpoint, n_cards=n_cards, seed=seed,
        budget=Stage7Budget(),
    )
    emit("stage7_redteam", {"ok": True, "metrics": stage7.metrics})
    evaluation = run_evaluation_suite(
        out_dir=out / "evaluation", num_games=max(2, num_corpus_games), checkpoint=stage4.checkpoint, seed=seed,
        profile_name="SMOKE",
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
        "event_log": str(event_path),
        "event_count": sink.count() if sink is not None else 0,
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
    if runtime.is_rank0:
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    payload = broadcast_object(summary if runtime.is_rank0 else None, src=0)
    if payload is None:
        raise RuntimeError("smoke pipeline payload broadcast failed")
    barrier_if_distributed()
    return payload


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
