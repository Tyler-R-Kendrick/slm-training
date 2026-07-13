#!/usr/bin/env python3
"""Train a ModelPlugin (default: StubModel) on train-data artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.model_build import ModelBuildConfig, train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("outputs/train_data/v0"),
    )
    parser.add_argument("--run-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--run-id", default="latest")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-rate", type=float, default=0.0)
    args = parser.parse_args(argv)

    summary = train(
        ModelBuildConfig(
            train_dir=args.train_dir,
            run_root=args.run_root,
            run_id=args.run_id,
            steps=args.steps,
            batch_size=args.batch_size,
            seed=args.seed,
            noise_rate=args.noise_rate,
        )
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
