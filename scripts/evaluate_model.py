#!/usr/bin/env python3
"""Evaluate a ModelPlugin checkpoint on a test suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.model_build import ModelBuildConfig, evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("outputs/test_data/v0"),
    )
    parser.add_argument("--suite", default="smoke")
    parser.add_argument("--run-root", type=Path, default=Path("outputs/runs"))
    parser.add_argument("--run-id", default="latest")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to outputs/runs/<run-id>/checkpoints/last.pt",
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("outputs/train_data/v0"),
        help="Unused for stub eval but kept for config symmetry",
    )
    args = parser.parse_args(argv)

    config = ModelBuildConfig(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        suite=args.suite,
        run_root=args.run_root,
        run_id=args.run_id,
    )
    metrics = evaluate(config, checkpoint=args.checkpoint)
    # Print summary without full details spam
    summary = {k: v for k, v in metrics.items() if k != "details"}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
