"""Top-level training coordinator and CLI-callable orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .corpus import generate_random_game_corpus
from .distributed import STAGE_SEQUENCE
from .stage0_verify import run_stage0_verify
from .stages import (
    iter_declared_stages,
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

    def write_resolved_config(self) -> Path:
        path = self.artifact_dir / "resolved_config.json"
        path.write_text(json.dumps(asdict(self.config), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

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

    def _dispatch_stage(self, stage: str, *, init_from_checkpoint: str | None) -> dict[str, Any]:
        """Run exactly one stage.  θ-stages receive ``init_from_checkpoint``.

        This is the single dispatch ladder shared by the single-stage ``run``
        path and the in-process ``run_full_sequence`` path, so the two can never
        drift in how a stage is invoked.
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
                init_from_checkpoint=init_from_checkpoint,
                resume_checkpoint=resume,
            )
            return {"stage": stage, "ok": True, "metrics": asdict(metrics)}
        if stage == "stage2_semi_supervised":
            metrics = run_stage2_semi_supervised(
                steps=self.config.steps,
                out_dir=self.artifact_dir / "data",
                n_cards=self.config.n_cards,
            )
            return {"stage": stage, "ok": True, "metrics": asdict(metrics)}
        if stage == "stage3_sft":
            metrics = run_stage3_sft(
                steps=self.config.steps,
                batch_size=self.config.batch_size,
                out_dir=self.artifact_dir / "checkpoints",
                device=self.config.device,
                n_cards=self.config.n_cards,
                init_from_checkpoint=init_from_checkpoint,
                resume_checkpoint=resume,
            )
            return {"stage": stage, "ok": True, "metrics": asdict(metrics)}
        if stage == "stage4_robust_rl":
            metrics = run_stage4_robust_rl(
                steps=self.config.steps,
                batch_size=self.config.batch_size,
                out_dir=self.artifact_dir / "checkpoints",
                device=self.config.device,
                n_cards=self.config.n_cards,
                init_from_checkpoint=init_from_checkpoint,
                resume_checkpoint=resume,
            )
            return {"stage": stage, "ok": True, "metrics": asdict(metrics)}
        if stage == "stage5_adaptive":
            metrics = run_stage5_adaptive(
                steps=self.config.steps,
                out_dir=self.artifact_dir,
                n_cards=self.config.n_cards,
                init_from_checkpoint=init_from_checkpoint,
                resume_checkpoint=resume,
            )
            return {"stage": stage, "ok": True, "metrics": asdict(metrics)}
        if stage == "stage6_league":
            metrics = run_stage6_league(out_dir=self.artifact_dir)
            return {"stage": stage, "ok": True, "metrics": asdict(metrics)}
        if stage == "stage7_redteam":
            metrics = run_stage7_redteam(out_dir=self.artifact_dir)
            return {"stage": stage, "ok": True, "metrics": asdict(metrics)}
        if stage == "evaluate":
            payload = run_evaluation_suite(out_dir=self.artifact_dir, num_games=16)
            return {"stage": stage, "ok": True, "metrics": payload}
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

        result = self._dispatch_stage(stage, init_from_checkpoint=init_ckpt)
        if stage in THETA_PARENT:
            # Make θ-inheritance auditable even on the single-stage path: never
            # silent about whether this stage stood on the previous one.
            result["init_from_checkpoint"] = init_ckpt
            result["init_inherited"] = init_ckpt is not None
            result["init_auto_discovered"] = auto_discovered
        return result

    def run_full_sequence(self) -> dict[str, Any]:
        """Run the entire frozen STAGE_SEQUENCE in one process, auto-wiring θ.

        Each θ-producing stage's checkpoint is threaded forward, in memory, as
        the next θ-stage's ``init_from_checkpoint`` — so the learned weights of
        every stage feed the next.  A θ-stage that should inherit but finds no
        parent checkpoint is a HARD ERROR: we refuse to silently train a stage
        from scratch, because "each stage trains itself in isolation" is exactly
        the failure this method exists to prevent.

        Works for single-GPU and for ``torchrun`` multi-GPU alike: every rank
        runs the same sequence, and each stage's own ``barrier_if_distributed``
        + rank0 checkpoint write + ``init_from_checkpoint`` load coordinate the
        hand-off across the stage boundary.
        """
        self.write_resolved_config()
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
                    raise RuntimeError(
                        f"θ auto-wiring broken: stage {stage!r} must inherit weights "
                        f"from {parent!r}, but {parent!r} produced no checkpoint. "
                        f"Refusing to train {stage!r} from scratch."
                    )
            elif stage == "stage1_pretrain":
                # The chain's head may be seeded from an external checkpoint, but
                # normally starts from a fresh init.  resume takes precedence.
                if self.config.resume_checkpoint is None:
                    init_ckpt = self.config.init_from_checkpoint

            result = self._dispatch_stage(stage, init_from_checkpoint=init_ckpt)
            results[stage] = result
            ok = ok and bool(result.get("ok", False))

            checkpoint = None
            metrics = result.get("metrics")
            if isinstance(metrics, dict):
                checkpoint = metrics.get("checkpoint")
            produced[stage] = checkpoint

            if stage in THETA_PRODUCERS:
                lineage.append({
                    "stage": stage,
                    "inherited_from": parent,
                    "init_checkpoint": init_ckpt,
                    "produced_checkpoint": checkpoint,
                })

        summary = {
            "stage": "full_sequence",
            "ok": ok,
            "stages_run": [s for s in STAGE_SEQUENCE if s != "smoke_pipeline"],
            "lineage_chain": lineage,
            "artifact_dir": str(self.artifact_dir),
        }
        (self.artifact_dir / "full_sequence_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return summary
