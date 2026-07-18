#!/usr/bin/env python3
"""Train only the VSS3-02 cost-to-go head from a solver-supervision corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

from slm_training.harnesses.model_build.cost_to_go_train import (
    train_cost_to_go_from_paths,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    summary = train_cost_to_go_from_paths(
        checkpoint=args.checkpoint,
        rows_path=args.rows,
        out_dir=args.out_dir,
        steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
