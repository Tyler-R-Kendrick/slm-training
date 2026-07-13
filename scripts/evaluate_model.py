#!/usr/bin/env python3
"""Evaluate a ModelPlugin checkpoint on a test suite (eval-driven gates)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.model_build import ModelBuildConfig, evaluate
from slm_training.harnesses.model_build.eval_runner import evaluate_suites


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("outputs/test_data/v1"),
    )
    parser.add_argument("--suite", default="smoke")
    parser.add_argument(
        "--suites",
        default=None,
        help="Comma-separated suites for a scoreboard (overrides --suite).",
    )
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
        default=Path("outputs/train_data/v1"),
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
        "--fail-under-placeholder-fidelity",
        type=float,
        default=None,
        help="Exit non-zero if placeholder_fidelity is below this threshold.",
    )
    parser.add_argument(
        "--fail-under-structural-similarity",
        type=float,
        default=None,
        help="Exit non-zero if structural_similarity is below this threshold.",
    )
    parser.add_argument(
        "--fail-under-reward-score",
        type=float,
        default=None,
        help="Exit non-zero if mean composite reward_score is below this threshold.",
    )
    parser.add_argument(
        "--fail-under-design-lint",
        type=float,
        default=None,
        help="Exit non-zero if mean design_lint_score (context) is below this threshold.",
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

    if args.suites:
        suites = [s.strip() for s in args.suites.split(",") if s.strip()]
        scoreboard = evaluate_suites(config, suites, checkpoint=args.checkpoint)
        print(json.dumps({k: v for k, v in scoreboard.items()}, indent=2))
        # Gate on the first / primary suite (smoke preferred).
        primary = "smoke" if "smoke" in scoreboard["suites"] else suites[0]
        metrics = scoreboard["suites"][primary]
    else:
        metrics = evaluate(config, checkpoint=args.checkpoint)
        summary = {k: v for k, v in metrics.items() if k != "details"}
        print(json.dumps(summary, indent=2))

    if args.fail_under_parse_rate is not None:
        if float(metrics.get("parse_rate") or 0) < args.fail_under_parse_rate:
            return 2
    if args.fail_under_placeholder_fidelity is not None:
        if float(metrics.get("placeholder_fidelity") or 0) < args.fail_under_placeholder_fidelity:
            return 4
    if args.fail_under_structural_similarity is not None:
        if float(metrics.get("structural_similarity") or 0) < args.fail_under_structural_similarity:
            return 5
    if args.fail_under_reward_score is not None:
        if float(metrics.get("reward_score") or 0) < args.fail_under_reward_score:
            return 6
    if args.fail_under_design_lint is not None:
        score = metrics.get("design_lint_score")
        if score is None or float(score) < args.fail_under_design_lint:
            return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
