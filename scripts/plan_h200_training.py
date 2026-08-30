"""Print H200 production training commands without launching training."""

from __future__ import annotations

import argparse
import json
import shlex

from goofspiel.training.distributed import (
    DistributedTrainingConfig,
    STAGE_SEQUENCE,
    distributed_manifest,
    torchrun_command,
    validate_stage_sequence,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="artifacts/runs/h200_full")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-cards", type=int, default=13)
    parser.add_argument("--stages", nargs="*", default=STAGE_SEQUENCE)
    args = parser.parse_args()
    validate_stage_sequence(args.stages)
    config = DistributedTrainingConfig(
        num_nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
        master_addr=args.master_addr,
        master_port=args.master_port,
        artifact_dir=args.artifact_dir,
    )
    payload = distributed_manifest(config)
    payload["commands"] = [
        shlex.join(
            torchrun_command(config, stage=stage, steps=args.steps, batch_size=args.batch_size, n_cards=args.n_cards)
        )
        for stage in args.stages
    ]
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
