"""Top-level training coordinator and CLI-callable orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .corpus import generate_random_game_corpus
from .distributed import STAGE_SEQUENCE, broadcast_object, current_runtime, barrier_if_distributed
from .league import ROLE_AGGRESSIVE, ROLE_EXPLOITER, ROLE_ROBUST
from ..observability import TrainingLogger
from .stage0_verify import run_stage0_verify
from .stages import (
    iter_declared_stages,
    run_axis_promotion_selection,
    run_evaluation_suite,
    run_smoke_pipeline,
    run_stage1_pretrain,
    run_stage2_semi_supervised,
    run_stage3_sft,
    run_stage4_robust_rl,
    run_stage5_adaptive,
    run_stage6_league,
    run_stage7_redteam,
)

# ---------------------------------------------------------------------------
# The θ-inheritance chain.  These constants are the single source of truth for
# "which stage's learned weights feed which next stage", and where each stage
# writes the checkpoint a later stage must load.  Auto-wiring reads ONLY these
# maps, so a filename or ordering change is a one-line edit here, asserted by
# `test_coordinator_autowires_theta_chain`.
#
#   stage1_pretrain ──θ──▶ stage3_sft ──θ──▶ stage4_robust_rl ──θ──▶ stage5_adaptive
#
# stage2_semi_supervised is a DATA stage (no θ), so stage3 inherits stage1
# directly.  build_corpus / stage6 / stage7 / evaluate carry no θ at all.
THETA_PRODUCERS = ("stage1_pretrain", "stage3_sft", "stage4_robust_rl", "stage5_adaptive")
THETA_PARENT = {
    "stage3_sft": "stage1_pretrain",
    "stage4_robust_rl": "stage3_sft",
    "stage5_adaptive": "stage4_robust_rl",
}
# Where each θ-producing stage writes the checkpoint that its child loads.
# Paths are relative to the run's artifact_dir.  (stage5 is terminal — nothing
# inherits from it — so only the parents that get loaded need an entry, but we
# list all producers for completeness / disk discovery.)
_THETA_CHECKPOINT_RELPATH = {
    "stage1_pretrain": "checkpoints/stage1_pretrain.pt",
    "stage3_sft": "checkpoints/stage3_sft.pt",
    "stage4_robust_rl": "checkpoints/stage4_robust_rl.pt",
    "stage5_adaptive": "stage5_adaptive.pt",
}
# Aliases that request the whole auto-wired sequence rather than one stage.
FULL_SEQUENCE_ALIASES = ("all", "full", "full_sequence")


@dataclass
class TrainingRunConfig:
    artifact_dir: str = "artifacts/runs/manual"
    seed: int = 1
    stage: str = "stage0_verify"
    steps: int = 10
    batch_size: int = 8
    device: str = "cpu"
    num_corpus_games: int = 32
    n_cards: int = 5
    dry_run: bool = False
    # Phase 3.1 — two DISTINCT, never-conflated checkpoint seams:
    #   init_from_checkpoint: stage transition (θ-only, fresh optimizer/step-0)
    #   resume_checkpoint:    crash recovery (full model+optimizer+target+step)
    init_from_checkpoint: str | None = None
    resume_checkpoint: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class TrainingCoordinator:
    def __init__(self, config: TrainingRunConfig) -> None:
        self.config = config
        self.artifact_dir = Path(config.artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.logger: TrainingLogger | None = None

    def write_resolved_config(self) -> Path:
        path = self.artifact_dir / "resolved_config.json"
        path.write_text(json.dumps(asdict(self.config), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _build_logger(self) -> TrainingLogger:
        """Construct the run-level dual-channel logger (rank0-only writes).

        Rank is read from the torchrun env vars via ``current_runtime`` — a
        lightweight check that needs no torch.distributed init.  Non-rank0 ranks
        get a logger whose every method is a no-op, so multi-GPU runs neither
        spam the console nor race on ``run.log`` / ``events/run.jsonl``.
        """
        runtime = current_runtime()
        logger = TrainingLogger(
            self.artifact_dir,
            is_rank0=runtime.is_rank0,
            run_id=Path(self.artifact_dir).name or "local",
        )
        self.logger = logger
        return logger

    # ------------------------------------------------------------------
    # θ auto-wiring
    # ------------------------------------------------------------------
    def _discover_parent_on_disk(self, stage: str) -> str | None:
        """Find the parent θ-stage's checkpoint on disk, if this stage has one.

        Used when stages are run as SEPARATE process invocations into a shared
        ``artifact_dir`` (e.g. one ``torchrun`` per stage): the child stage can
        still inherit the parent's weights by convention instead of silently
        training from scratch.  Returns ``None`` for stages with no θ-parent
        (stage1) or when the parent checkpoint is not present yet.
        """
        parent = THETA_PARENT.get(stage)
        if parent is None:
            return None
        candidate = self.artifact_dir / _THETA_CHECKPOINT_RELPATH[parent]
        return str(candidate) if candidate.exists() else None

    def _resolve_checkpoint(self, stage: str, produced: dict[str, str | None] | None) -> str | None:
        """Resolve a θ-producing stage's checkpoint for a DOWNSTREAM consumer.

        Prefers the in-memory ``produced`` map (the full-sequence path, where the
        stage just ran in this process); falls back to the on-disk convention
        path (the per-stage-process path, where an earlier invocation wrote it
        into the shared ``artifact_dir``).  Returns ``None`` only when neither
        exists — the caller then degrades honestly (mints its own seed) rather
        than pretending a checkpoint was consumed.
        """
        if produced is not None:
            ckpt = produced.get(stage)
            if ckpt and Path(ckpt).exists():
                return ckpt
        relpath = _THETA_CHECKPOINT_RELPATH.get(stage)
        if relpath is not None:
            candidate = self.artifact_dir / relpath
            if candidate.exists():
                return str(candidate)
        return None

    # Human-readable role each θ-producer plays for the downstream stages, used
    # only to make a strict-mode lineage failure name what is actually missing.
    _DOWNSTREAM_ROLE = {
        "stage4_robust_rl": "robust backbone (P4)",
        "stage3_sft": "strategic/aggressive policy (P3)",
        "stage5_adaptive": "adaptive exploiter (P5)",
    }

    def _require_checkpoint(
        self, stage: str, produced: dict[str, str | None] | None, *, consumer: str
    ) -> str:
        """Strict resolve: the upstream product MUST exist or the lineage is broken.

        Full-sequence mode calls this instead of tolerating a ``None`` from
        ``_resolve_checkpoint``.  A missing product here means an earlier stage
        did not write the checkpoint this consumer is contractually built on, so
        we refuse rather than silently substitute a freshly-minted seed and pass
        it off as THIS run's result.
        """
        ckpt = self._resolve_checkpoint(stage, produced)
        if ckpt is None:
            role = self._DOWNSTREAM_ROLE.get(stage, stage)
            raise RuntimeError(
                f"lineage broken: {consumer!r} requires the {role} produced by "
                f"{stage!r}, but no such checkpoint was found (neither in this "
                f"run's produced map nor on disk at "
                f"{self.artifact_dir / _THETA_CHECKPOINT_RELPATH.get(stage, '?')}). "
                f"Refusing to run {consumer!r} on a throwaway seed in full-sequence mode."
            )
        return ckpt

    def _dispatch_stage(
        self,
        stage: str,
        *,
        init_from_checkpoint: str | None,
        produced: dict[str, str | None] | None = None,
        strict: bool = False,
    ) -> dict[str, Any]:
        """Run exactly one stage.  θ-stages receive ``init_from_checkpoint``.

        This is the single dispatch ladder shared by the single-stage ``run``
        path and the in-process ``run_full_sequence`` path, so the two can never
        drift in how a stage is invoked.  ``produced`` carries the checkpoints
        produced earlier in a full-sequence run so the non-θ downstream stages
        (league / red-team / evaluate) consume THIS run's real snapshots instead
        of silently minting their own.

        ``strict`` selects the lineage contract for the DOWNSTREAM (non-θ) stages:

          * Standalone mode (``strict=False``, the single-stage ``run`` path):
            a downstream stage that cannot resolve an upstream checkpoint degrades
            honestly — it mints its own seed and says so in the result.  This is
            correct when the user deliberately runs one stage in isolation.

          * Full-sequence mode (``strict=True``, ``run_full_sequence``): every
            required upstream product MUST be present.  A missing one is a broken
            lineage, not a fallback opportunity, so we raise rather than quietly
            evaluate/league/red-team a throwaway seed and label it as this run's
            result.  θ-stages already hard-fail earlier in ``run_full_sequence``;
            this extends the same refusal to league / red-team / evaluate.
        """
        resume = self.config.resume_checkpoint
        if stage == "stage0_verify":
            report = run_stage0_verify(artifact_dir=self.artifact_dir / "stage0_verify")
            return {"stage": stage, "ok": report.ok, "checks": report.checks, "metrics": report.metrics, "errors": report.errors}
        if stage == "build_corpus":
            metrics = generate_random_game_corpus(
                out_path=self.artifact_dir / "data" / "game_corpus.jsonl",
                num_games=self.config.num_corpus_games,
                seed=self.config.seed,
            )
            return {"stage": stage, "ok": True, "metrics": metrics}
        if stage == "stage1_pretrain":
            metrics = run_stage1_pretrain(
                steps=self.config.steps,
                batch_size=self.config.batch_size,
                out_dir=self.artifact_dir / "checkpoints",
                device=self.config.device,
                n_cards=self.config.n_cards,
                seed=self.config.seed,
                corpus_path=self.artifact_dir / "data" / "game_corpus.jsonl",
                init_from_checkpoint=init_from_checkpoint,
                resume_checkpoint=resume,
                logger=self.logger,
            )
            return {"stage": stage, "ok": True, "metrics": asdict(metrics)}
        if stage == "stage2_semi_supervised":
            metrics = run_stage2_semi_supervised(
                steps=self.config.steps,
                out_dir=self.artifact_dir / "data",
                n_cards=self.config.n_cards,
                seed=self.config.seed,
            )
            return {"stage": stage, "ok": True, "metrics": asdict(metrics)}
        if stage == "stage3_sft":
            metrics = run_stage3_sft(
                steps=self.config.steps,
                batch_size=self.config.batch_size,
                out_dir=self.artifact_dir / "checkpoints",
                device=self.config.device,
                n_cards=self.config.n_cards,
                seed=self.config.seed,
                init_from_checkpoint=init_from_checkpoint,
                resume_checkpoint=resume,
                logger=self.logger,
            )
            return {"stage": stage, "ok": True, "metrics": asdict(metrics)}
        if stage == "stage4_robust_rl":
            metrics = run_stage4_robust_rl(
                steps=self.config.steps,
                batch_size=self.config.batch_size,
                out_dir=self.artifact_dir / "checkpoints",
                device=self.config.device,
                n_cards=self.config.n_cards,
                seed=self.config.seed,
                init_from_checkpoint=init_from_checkpoint,
                resume_checkpoint=resume,
                logger=self.logger,
            )
            return {"stage": stage, "ok": True, "metrics": asdict(metrics)}
        if stage == "stage5_adaptive":
            metrics = run_stage5_adaptive(
                steps=self.config.steps,
                out_dir=self.artifact_dir,
                n_cards=self.config.n_cards,
                seed=self.config.seed,
                init_from_checkpoint=init_from_checkpoint,
                resume_checkpoint=resume,
                logger=self.logger,
            )
            return {"stage": stage, "ok": True, "metrics": asdict(metrics)}
        if stage == "stage6_league":
            # Phase 4.3: the league must play THIS run's real trained snapshots —
            # P4 (robust), P3 (aggressive/strategic), P5 (exploiter) — not freshly
            # minted role seeds.
            #   Standalone: only supply roles whose checkpoint resolved; a missing
            #     one lets run_stage6_league mint that single seed honestly.
            #   Full-sequence (strict): all three are required products of this
            #     run — a missing one is a broken lineage, not a fallback.
            role_checkpoints: dict[str, str] = {}
            if strict:
                role_checkpoints[ROLE_ROBUST] = self._require_checkpoint(
                    "stage4_robust_rl", produced, consumer="stage6_league"
                )
                role_checkpoints[ROLE_AGGRESSIVE] = self._require_checkpoint(
                    "stage3_sft", produced, consumer="stage6_league"
                )
                role_checkpoints[ROLE_EXPLOITER] = self._require_checkpoint(
                    "stage5_adaptive", produced, consumer="stage6_league"
                )
            else:
                robust = self._resolve_checkpoint("stage4_robust_rl", produced)
                aggressive = self._resolve_checkpoint("stage3_sft", produced)
                exploiter = self._resolve_checkpoint("stage5_adaptive", produced)
                if robust:
                    role_checkpoints[ROLE_ROBUST] = robust
                if aggressive:
                    role_checkpoints[ROLE_AGGRESSIVE] = aggressive
                if exploiter:
                    role_checkpoints[ROLE_EXPLOITER] = exploiter
            metrics = run_stage6_league(
                out_dir=self.artifact_dir,
                role_checkpoints=role_checkpoints or None,
                n_cards=self.config.n_cards,
                seed=self.config.seed,
            )
            result = {"stage": stage, "ok": True, "metrics": asdict(metrics)}
            result["role_checkpoints"] = role_checkpoints
            return result
        if stage == "stage7_redteam":
            # Phase 4.4: focus the red-team correction on THIS run's P4 robust
            # backbone, not a throwaway seed minted inside the stage.  Strict
            # full-sequence mode requires P4; standalone tolerates its absence.
            if strict:
                init_p4 = self._require_checkpoint(
                    "stage4_robust_rl", produced, consumer="stage7_redteam"
                )
            else:
                init_p4 = self._resolve_checkpoint("stage4_robust_rl", produced)
            metrics = run_stage7_redteam(
                out_dir=self.artifact_dir,
                init_from_checkpoint=init_p4,
                n_cards=self.config.n_cards,
                seed=self.config.seed,
            )
            result = {"stage": stage, "ok": True, "metrics": asdict(metrics)}
            result["init_from_checkpoint"] = init_p4
            result["init_inherited"] = init_p4 is not None
            return result
        if stage == "evaluate":
            # Phase 5: evaluate THIS run's robust checkpoint so E2/E6 are real
            # trained-model play and G2 is a computed verdict (not the heuristic
            # reference with G2 unrun).  Strict full-sequence mode requires P4;
            # standalone tolerates its absence (heuristic-reference evaluation).
            if strict:
                eval_ckpt = self._require_checkpoint(
                    "stage4_robust_rl", produced, consumer="evaluate"
                )
            else:
                eval_ckpt = self._resolve_checkpoint("stage4_robust_rl", produced)
            payload = run_evaluation_suite(
                out_dir=self.artifact_dir, num_games=16, checkpoint=eval_ckpt, seed=self.config.seed
            )
            result = {"stage": stage, "ok": True, "metrics": payload}
            result["evaluated_checkpoint"] = eval_ckpt
            return result
        if stage == "smoke_pipeline":
            payload = run_smoke_pipeline(
                out_dir=self.artifact_dir,
                steps=self.config.steps,
                batch_size=self.config.batch_size,
                device=self.config.device,
                n_cards=self.config.n_cards,
                num_corpus_games=self.config.num_corpus_games,
                seed=self.config.seed,
            )
            return {"stage": stage, "ok": payload["ok"], "metrics": payload}
        raise ValueError(f"unknown stage={stage!r}; allowed={iter_declared_stages() + ['build_corpus', 'evaluate']}")

    def run(self) -> dict[str, Any]:
        self.write_resolved_config()
        if self.config.dry_run:
            return {
                "dry_run": True,
                "declared_stages": iter_declared_stages(),
                "artifact_dir": str(self.artifact_dir),
            }

        stage = self.config.stage
        if stage in FULL_SEQUENCE_ALIASES:
            return self.run_full_sequence()

        # Single-stage invocation.  An explicit init_from_checkpoint always wins;
        # a resume takes over the θ seam entirely (mutually exclusive downstream).
        # Otherwise, auto-discover the parent θ-checkpoint on disk so a per-stage
        # launch into a shared artifact_dir still inherits weights automatically.
        init_ckpt = self.config.init_from_checkpoint
        auto_discovered = False
        if init_ckpt is None and self.config.resume_checkpoint is None:
            init_ckpt = self._discover_parent_on_disk(stage)
            auto_discovered = init_ckpt is not None

        logger = self._build_logger()
        logger.run_start(asdict(self.config))
        logger.stage_start(
            stage,
            init_from=init_ckpt,
            inherited=THETA_PARENT.get(stage) if init_ckpt else None,
        )
        started = time.perf_counter()
        try:
            result = self._dispatch_stage(stage, init_from_checkpoint=init_ckpt)
        except Exception as exc:
            logger.error(stage, f"stage {stage!r} raised: {exc}", exc=exc)
            logger.stage_end(stage, ok=False, elapsed_s=round(time.perf_counter() - started, 3))
            raise
        stage_ckpt = None
        stage_metrics = result.get("metrics")
        if isinstance(stage_metrics, dict):
            stage_ckpt = stage_metrics.get("checkpoint")
        logger.stage_end(
            stage,
            ok=bool(result.get("ok", False)),
            metrics=stage_metrics if isinstance(stage_metrics, dict) else None,
            checkpoint=stage_ckpt,
            elapsed_s=round(time.perf_counter() - started, 3),
        )
        if stage in THETA_PARENT:
            # Make θ-inheritance auditable even on the single-stage path: never
            # silent about whether this stage stood on the previous one.
            result["init_from_checkpoint"] = init_ckpt
            result["init_inherited"] = init_ckpt is not None
            result["init_auto_discovered"] = auto_discovered
        logger.run_end({"ok": result.get("ok", False), "stages_run": [stage]})
        logger.close()
        return result

    def run_full_sequence(self) -> dict[str, Any]:
        """Run the entire frozen STAGE_SEQUENCE in one process, auto-wiring θ.

        Each θ-producing stage's checkpoint is threaded forward, in memory, as
        the next θ-stage's ``init_from_checkpoint`` — so the learned weights of
        every stage feed the next.  A θ-stage that should inherit but finds no
        parent checkpoint is a HARD ERROR: we refuse to silently train a stage
        from scratch, because "each stage trains itself in isolation" is exactly
        the failure this method exists to prevent.

        The same refusal extends to the downstream (non-θ) stages via
        ``strict=True``: league / red-team / evaluate MUST consume this run's
        real snapshots (``_require_checkpoint``), never a throwaway seed.  This
        is the difference between the two modes — the single-stage ``run`` path
        degrades honestly when a stage is deliberately run in isolation; the
        full sequence treats any missing required product as a broken lineage.

        Works for single-GPU and for ``torchrun`` multi-GPU alike: every rank
        runs the same sequence, and each stage's own ``barrier_if_distributed``
        + rank0 checkpoint write + ``init_from_checkpoint`` load coordinate the
        hand-off across the stage boundary.
        """
        self.write_resolved_config()
        logger = self._build_logger()
        logger.run_start(asdict(self.config))
        logger.system_metrics("run_start")
        produced: dict[str, str | None] = {}
        results: dict[str, Any] = {}
        lineage: list[dict[str, Any]] = []
        ok = True

        for stage in STAGE_SEQUENCE:
            if stage == "smoke_pipeline":
                # smoke is a self-contained miniature of this sequence, not a
                # member of the real chain; running it here would double-run.
                continue

            init_ckpt: str | None = None
            parent = THETA_PARENT.get(stage)
            if parent is not None:
                init_ckpt = produced.get(parent)
                if init_ckpt is None:
                    msg = (
                        f"θ auto-wiring broken: stage {stage!r} must inherit weights "
                        f"from {parent!r}, but {parent!r} produced no checkpoint. "
                        f"Refusing to train {stage!r} from scratch."
                    )
                    logger.error(stage, msg)
                    raise RuntimeError(msg)
            elif stage == "stage1_pretrain":
                # The chain's head may be seeded from an external checkpoint, but
                # normally starts from a fresh init.  resume takes precedence.
                if self.config.resume_checkpoint is None:
                    init_ckpt = self.config.init_from_checkpoint

            logger.stage_start(stage, init_from=init_ckpt, inherited=parent)
            started = time.perf_counter()
            try:
                result = self._dispatch_stage(
                    stage, init_from_checkpoint=init_ckpt, produced=produced, strict=True
                )
            except Exception as exc:
                logger.error(stage, f"stage {stage!r} raised: {exc}", exc=exc)
                logger.stage_end(stage, ok=False, elapsed_s=round(time.perf_counter() - started, 3))
                raise
            results[stage] = result
            ok = ok and bool(result.get("ok", False))

            checkpoint = None
            metrics = result.get("metrics")
            if isinstance(metrics, dict):
                checkpoint = metrics.get("checkpoint")
            # A θ-producing stage writes its checkpoint on RANK0 ONLY, so on every
            # other rank `metrics["checkpoint"]` is None even though the file now
            # exists on the shared disk (guaranteed: each such stage ends with a
            # post-write `barrier_if_distributed()`).  `produced[stage]` must mean
            # "did THIS RUN produce it" — true on all ranks — not "did MY rank
            # write it"; otherwise the θ-inheritance check below raises on rank1+
            # with "<parent> produced no checkpoint" while rank0 sails through.
            # Fall back to the on-disk convention path (the same resolution the
            # downstream strict stages already use) so the map is rank-symmetric.
            if checkpoint is None:
                relpath = _THETA_CHECKPOINT_RELPATH.get(stage)
                if relpath is not None:
                    candidate = self.artifact_dir / relpath
                    if candidate.exists():
                        checkpoint = str(candidate)
            produced[stage] = checkpoint

            logger.stage_end(
                stage,
                ok=bool(result.get("ok", False)),
                metrics=metrics if isinstance(metrics, dict) else None,
                checkpoint=checkpoint,
                elapsed_s=round(time.perf_counter() - started, 3),
            )

            if stage in THETA_PRODUCERS:
                lineage.append({
                    "stage": stage,
                    "inherited_from": parent,
                    "init_checkpoint": init_ckpt,
                    "produced_checkpoint": checkpoint,
                })
                if parent is not None:
                    logger.theta_wired(
                        stage, parent, init_ckpt=init_ckpt, produced_ckpt=checkpoint
                    )

        # Phase 5: register best_* aliases from THIS run's genuinely-distinct
        # checkpoints (P3 strategic SFT, P4 robust, P5 adaptive, P7 corrected),
        # each re-evaluated on its own axis.  smoke does this too; the full
        # sequence must not skip it or the run produces no per-axis promotion.
        axis_candidates: dict[str, str] = {}
        for cid, stage_name in (
            ("stage3_sft", "stage3_sft"),
            ("stage4_robust", "stage4_robust_rl"),
            ("stage5_adaptive", "stage5_adaptive"),
        ):
            ckpt = self._resolve_checkpoint(stage_name, produced)
            if ckpt:
                axis_candidates[cid] = ckpt
        stage7_metrics = results.get("stage7_redteam", {}).get("metrics")
        stage7_ckpt = stage7_metrics.get("checkpoint") if isinstance(stage7_metrics, dict) else None
        if stage7_ckpt and Path(stage7_ckpt).exists():
            axis_candidates["stage7_corrected"] = stage7_ckpt

        axis_selection: dict[str, Any] | None = None
        if axis_candidates:
            axis_selection = run_axis_promotion_selection(
                out_dir=self.artifact_dir,
                candidates=axis_candidates,
                primary_n=max(3, min(self.config.n_cards, 5)),
                generalization_ns=(3, max(3, min(self.config.n_cards, 5))),
                exploit_n=min(4, max(3, self.config.n_cards)),
                num_games=max(4, self.config.num_corpus_games),
                seed=self.config.seed,
            )

        summary = {
            "stage": "full_sequence",
            "ok": ok,
            "stages_run": [s for s in STAGE_SEQUENCE if s != "smoke_pipeline"],
            "lineage_chain": lineage,
            "axis_selection": axis_selection,
            "artifact_dir": str(self.artifact_dir),
        }
        runtime = current_runtime()

        # Priority ⑥: assemble THIS run's checkpoints into a lineage tree and
        # record whether it is internally consistent — every child descends from
        # the parent it names, that parent file is unchanged since the child
        # inherited, and the architecture is stable across every edge.  Computed
        # by re-hashing on disk, not by trusting a flag.
        if runtime.is_rank0:
            from .lineage import build_lineage_from_run

            tree = build_lineage_from_run(self.artifact_dir)
            summary["lineage_consistent"] = tree.is_consistent()
            summary["lineage_inconsistencies"] = tree.inconsistencies()
            summary["lineage_order"] = tree.chain_order()
            logger.lineage_verdict(
                tree.is_consistent(),
                inconsistencies=tree.inconsistencies(),
                order=tree.chain_order(),
            )
        else:
            summary["lineage_consistent"] = None
            summary["lineage_inconsistencies"] = []
            summary["lineage_order"] = []

        # Surface the two log products on the summary, mirroring smoke's
        # event_log/event_count convention, so a run's observability outputs are
        # discoverable straight from the summary JSON.
        summary["run_log"] = str(logger.run_log_path)
        summary["event_log"] = str(logger.event_log_path)
        summary["event_count"] = logger.event_count

        if runtime.is_rank0:
            (self.artifact_dir / "full_sequence_summary.json").write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        logger.system_metrics("run_end")
        logger.run_end(summary)
        logger.close()
        payload = broadcast_object(summary if runtime.is_rank0 else None, src=0)
        if payload is None:
            raise RuntimeError("full sequence summary broadcast failed")
        barrier_if_distributed()
        return payload
