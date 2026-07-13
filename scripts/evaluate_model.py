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
        help="Used to rebuild vocab when loading TwoTower without sidecar tokenizer.",
    )
    parser.add_argument(
        "--model",
        choices=("twotower", "stub"),
        default="twotower",
        help="Must match the checkpoint kind.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--fail-under-parse-rate",
        type=float,
        default=None,
        help="Exit non-zero if parse_rate is below this threshold.",
    )
    parser.add_argument(
        "--fail-under-design-lint",
        type=float,
        default=None,
        help="Exit non-zero if mean design_lint_score is below this threshold.",
    )
    args = parser.parse_args(argv)

    config = ModelBuildConfig(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        suite=args.suite,
        run_root=args.run_root,
        run_id=args.run_id,
        model_name=args.model,
        device=args.device,
        context_backend="scratch",
    )
    metrics = evaluate(config, checkpoint=args.checkpoint)
    summary = {k: v for k, v in metrics.items() if k != "details"}
    print(json.dumps(summary, indent=2))
    if args.fail_under_parse_rate is not None:
        if float(metrics.get("parse_rate") or 0) < args.fail_under_parse_rate:
            return 2
    if args.fail_under_design_lint is not None:
        score = metrics.get("design_lint_score")
        if score is None or float(score) < args.fail_under_design_lint:
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
