"""Stage-gated training entrypoint for the full Goofspiel agent.

This script contains the real training pipeline entrypoint.  It does not run
anything until you invoke it with an explicit stage.

Examples:
    python scripts/train_goofspiel_full.py --stage stage0_verify
    python scripts/train_goofspiel_full.py --stage build_corpus --num-corpus-games 1000
    python scripts/train_goofspiel_full.py --stage stage1_pretrain --steps 1000 --batch-size 64 --device cuda
    python scripts/train_goofspiel_full.py --stage stage3_sft --steps 1000 --batch-size 64 --device cuda
    python scripts/train_goofspiel_full.py --eval-checkpoint artifacts/runs/manual/checkpoints/stage4_robust_rl.pt
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from goofspiel.training import TrainingCoordinator, TrainingRunConfig


def _run_eval_checkpoint(args: argparse.Namespace) -> None:
    """Honest evaluation of a trained checkpoint (Phase 0.1).

    This path never touches the fake benchmark; it loads the real model and
    reports computed win-rate / score-diff / exploitability figures.
    """
    from goofspiel.training.model_eval import evaluate_checkpoint

    report = evaluate_checkpoint(
        args.eval_checkpoint,
        device=args.device,
        n_cards=tuple(args.eval_n_cards),
        num_games=args.eval_games,
        seed=args.seed,
    )
    print(json.dumps(asdict(report), indent=2, ensure_ascii=False))

from goofspiel.training import TrainingCoordinator, TrainingRunConfig


def parse_args() -> argparse.Namespace:
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
    # Phase 0.1: honest evaluation of a trained checkpoint. When supplied, the
    # script evaluates the checkpoint and exits without running any stage.
    p.add_argument("--eval-checkpoint", default=None,
                   help="Path to a .pt checkpoint to honestly evaluate (skips training).")
    p.add_argument("--eval-n-cards", type=int, nargs="+", default=[5, 7],
                   help="Card counts to evaluate the checkpoint at (default: 5 7).")
    p.add_argument("--eval-games", type=int, default=64,
                   help="Games per matchup during checkpoint evaluation.")
    return p.parse_args()


def config_from_args(args: argparse.Namespace) -> TrainingRunConfig:
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
    args = parse_args()
    if args.eval_checkpoint is not None:
        _run_eval_checkpoint(args)
        return
    result = TrainingCoordinator(config_from_args(args)).run()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
