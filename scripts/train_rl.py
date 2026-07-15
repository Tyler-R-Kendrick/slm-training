#!/usr/bin/env python3
"""Online GRPO-lite RL stage for TwoTower (structure-only reward)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.rl import train_grpo_from_paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--train-records",
        type=Path,
        required=True,
        help="JSONL of ExampleRecords used as rollout prompts.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/runs/grpo"))
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--kl-beta", type=float, default=0.05)
    parser.add_argument(
        "--ref-checkpoint",
        type=Path,
        default=None,
        help="Frozen SFT checkpoint for KL penalty (optional).",
    )
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--rl-readiness-report",
        type=Path,
        required=True,
        help="Approved frozen full-suite competence report; RL has no bypass.",
    )
    args = parser.parse_args(argv)

    summary = train_grpo_from_paths(
        args.checkpoint,
        args.train_records,
        out_dir=args.out_dir,
        steps=args.steps,
        group_size=args.group_size,
        device=args.device,
        ref_checkpoint=args.ref_checkpoint,
        limit=args.limit,
        kl_beta=args.kl_beta,
        lr=args.lr,
        readiness_report=args.rl_readiness_report,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
