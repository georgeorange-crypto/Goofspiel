"""Stage-gated training entrypoint for the full Goofspiel agent.

This script contains the real training pipeline entrypoint.  It does not run
anything until you invoke it with an explicit stage.

Examples:
    python scripts/train_goofspiel_full.py --stage stage0_verify
    python scripts/train_goofspiel_full.py --stage build_corpus --num-corpus-games 1000
    python scripts/train_goofspiel_full.py --stage stage1_pretrain --steps 1000 --batch-size 64 --device cuda
    python scripts/train_goofspiel_full.py --stage stage3_sft --steps 1000 --batch-size 64 --device cuda
    python scripts/train_goofspiel_full.py --eval-checkpoint artifacts/runs/manual/checkpoints/stage4_robust_rl.pt

    # Full auto-wired pipeline (θ inherited stage-to-stage automatically):
    python scripts/train_goofspiel_full.py --stage all --steps 2000 --batch-size 64 --device cuda

When run per-stage into a SHARED --artifact-dir, each θ-stage auto-discovers and
inherits the previous stage's checkpoint from disk; with --stage all the whole
chain is threaded in one process.  Pass --init-from-checkpoint to override the
seed weights of the first/only stage explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    # Stage transition (Phase 3.1): explicitly seed the first/only stage's θ from
    # a checkpoint. In `--stage all` this seeds the chain head (stage1); for a
    # single θ-stage it overrides the on-disk auto-discovery. Mutually exclusive
    # with --resume-checkpoint.
    p.add_argument("--init-from-checkpoint", default=None,
                   help="Seed θ (weights only) from this checkpoint; overrides auto-discovery.")
    p.add_argument("--resume-checkpoint", default=None,
                   help="Crash resume: restore full training state (model+optimizer+step).")
    p.add_argument("--stage4-resume-checkpoint-interval", type=int, default=250,
                   help="Stage4 periodic crash-resume checkpoint interval in stage steps.")
    # Stage5 control-plane liveness (Phase 1). Stage5 is long and rank0-only;
    # non-rank0 ranks poll rank0's heartbeat file instead of blocking in an NCCL
    # collective. These tune the fail-closed guards — NOT the NCCL timeout.
    p.add_argument("--stage5-heartbeat-timeout", type=float, default=300.0,
                   help="Seconds with no fresh rank0 heartbeat before Stage5 peers "
                        "declare rank0 dead and fail closed. This is a liveness timeout, "
                        "NOT a cap on total Stage5 runtime.")
    p.add_argument("--stage5-hard-timeout", type=float, default=48.0 * 3600.0,
                   help="Absolute last-resort Stage5 wait cap (seconds) guarding a rank0 "
                        "that keeps heartbeating but never terminates (fake-alive).")
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
        init_from_checkpoint=args.init_from_checkpoint,
        resume_checkpoint=args.resume_checkpoint,
        extra={
            "stage4_resume_checkpoint_interval": args.stage4_resume_checkpoint_interval,
            "stage5_heartbeat_timeout": args.stage5_heartbeat_timeout,
            "stage5_hard_timeout": args.stage5_hard_timeout,
        },
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
