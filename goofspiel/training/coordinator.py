"""Top-level training coordinator and CLI-callable orchestration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .corpus import generate_random_game_corpus
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

    def run(self) -> dict[str, Any]:
        self.write_resolved_config()
        if self.config.dry_run:
            return {
                "dry_run": True,
                "declared_stages": iter_declared_stages(),
                "artifact_dir": str(self.artifact_dir),
            }

        stage = self.config.stage
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
            )
            return {"stage": stage, "ok": True, "metrics": asdict(metrics)}
        if stage == "stage3_sft":
            metrics = run_stage3_sft(
                steps=self.config.steps,
                batch_size=self.config.batch_size,
                out_dir=self.artifact_dir / "checkpoints",
                device=self.config.device,
                n_cards=self.config.n_cards,
            )
            return {"stage": stage, "ok": True, "metrics": asdict(metrics)}
        if stage == "stage2_semi_supervised":
            metrics = run_stage2_semi_supervised(
                steps=self.config.steps,
                out_dir=self.artifact_dir / "data",
                n_cards=self.config.n_cards,
            )
            return {"stage": stage, "ok": True, "metrics": asdict(metrics)}
        if stage == "stage4_robust_rl":
            metrics = run_stage4_robust_rl(
                steps=self.config.steps,
                batch_size=self.config.batch_size,
                out_dir=self.artifact_dir / "checkpoints",
                device=self.config.device,
                n_cards=self.config.n_cards,
            )
            return {"stage": stage, "ok": True, "metrics": asdict(metrics)}
        if stage == "stage5_adaptive":
            metrics = run_stage5_adaptive(
                steps=self.config.steps,
                out_dir=self.artifact_dir,
                n_cards=self.config.n_cards,
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
