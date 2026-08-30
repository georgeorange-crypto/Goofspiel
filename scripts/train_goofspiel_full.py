"""Stage-gated training entrypoint for the full Goofspiel agent.

This script contains the real training pipeline entrypoint.  It does not run
anything until you invoke it with an explicit stage.

Examples:
    python scripts/train_goofspiel_full.py --stage stage0_verify
    python scripts/train_goofspiel_full.py --stage build_corpus --num-corpus-games 1000
    python scripts/train_goofspiel_full.py --stage stage1_pretrain --steps 1000 --batch-size 64 --device cuda
    python scripts/train_goofspiel_full.py --stage stage3_sft --steps 1000 --batch-size 64 --device cuda
"""

from __future__ import annotations

import argparse
import json

from goofspiel.training import TrainingCoordinator, TrainingRunConfig


def parse_args() -> TrainingRunConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-dir", default="artifacts/runs/manual")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--stage", default="stage0_verify")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--num-corpus-games", type=int, default=32)
    p.add_argument("--n-cards", type=int, default=5)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    return TrainingRunConfig(
        artifact_dir=args.artifact_dir,
        seed=args.seed,
        stage=args.stage,
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
        num_corpus_games=args.num_corpus_games,
        n_cards=args.n_cards,
        dry_run=args.dry_run,
    )


def main() -> None:
    result = TrainingCoordinator(parse_args()).run()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
