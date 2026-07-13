#!/usr/bin/env python3
"""Evaluate a ModelPlugin checkpoint on a test suite (eval-driven gates)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.model_build import ModelBuildConfig, evaluate
from slm_training.harnesses.model_build.eval_runner import evaluate_suites
from slm_training.harnesses.model_build.ship_gates import (
    DEFAULT_SHIP_GATES,
    evaluate_ship_gates,
    write_ship_gates,
)


def _check_fail_unders(metrics: dict, args: argparse.Namespace) -> int:
    if args.fail_under_parse_rate is not None:
        if float(metrics.get("parse_rate") or 0) < args.fail_under_parse_rate:
            return 2
    if args.fail_under_placeholder_fidelity is not None:
        if float(metrics.get("placeholder_fidelity") or 0) < args.fail_under_placeholder_fidelity:
            return 4
    if args.fail_under_placeholder_validity is not None:
        if float(metrics.get("placeholder_validity") or 0) < args.fail_under_placeholder_validity:
            return 7
    if args.fail_under_structural_similarity is not None:
        if float(metrics.get("structural_similarity") or 0) < args.fail_under_structural_similarity:
            return 5
    if args.fail_under_reward_score is not None:
        if float(metrics.get("reward_score") or 0) < args.fail_under_reward_score:
            return 6
    if args.fail_under_design_lint is not None:
        # Gold-context diagnostic only; prefer --ship-gates for readiness.
        score = metrics.get("gold_design_lint_score")
        if score is None:
            score = metrics.get("design_lint_score")
        if score is None or float(score) < args.fail_under_design_lint:
            return 3
    return 0


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
        "--ship-gates",
        action="store_true",
        help=(
            "Apply honest multi-suite ship gates (see docs/design/adversarial-review.md) "
            "and write gates.json. Implies checking every suite in the policy."
        ),
    )
    parser.add_argument(
        "--fail-under-parse-rate",
        type=float,
        default=None,
        help="Exit non-zero if parse_rate is below this threshold (single/primary suite).",
    )
    parser.add_argument(
        "--fail-under-placeholder-fidelity",
        type=float,
        default=None,
        help="Exit non-zero if placeholder_fidelity is below this threshold.",
    )
    parser.add_argument(
        "--fail-under-placeholder-validity",
        type=float,
        default=None,
        help="Diagnostic soft metric; prefer fidelity + --ship-gates for readiness.",
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
        help="Gold DESIGN.md context lint only — not model skill.",
    )
    parser.add_argument(
        "--grammar-ltr-primary",
        action="store_true",
        help="Override checkpoint: prefer greedy LTR decode.",
    )
    parser.add_argument(
        "--grammar-ltr-repair",
        action="store_true",
        help="Override checkpoint: constrained LTR repair on failed parses.",
    )
    parser.add_argument(
        "--schema-in-context",
        action="store_true",
        help="Override: inject compact schema into context.",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=0,
        help="Override: retrieve K train skeletons into context.",
    )
    parser.add_argument(
        "--best-of-n",
        type=int,
        default=1,
        help="Override: best-of-N decode by composite reward.",
    )
    parser.add_argument(
        "--rico-limit",
        type=int,
        default=None,
        help="Cap rico_held eval size (CPU/matrix).",
    )
    parser.add_argument(
        "--no-design-md-context",
        action="store_true",
        help="Override: do not concatenate DESIGN.md into context.",
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
        grammar_ltr_primary=args.grammar_ltr_primary,
        grammar_ltr_repair=args.grammar_ltr_repair,
        schema_in_context=args.schema_in_context,
        retrieval_k=args.retrieval_k,
        best_of_n=args.best_of_n,
        design_md_in_context=not args.no_design_md_context,
        rico_eval_limit=args.rico_limit,
    )

    if args.ship_gates and not args.suites:
        args.suites = ",".join(DEFAULT_SHIP_GATES.keys())

    if args.suites:
        suites = [s.strip() for s in args.suites.split(",") if s.strip()]
        scoreboard = evaluate_suites(
            config,
            suites,
            checkpoint=args.checkpoint,
            write_gates=args.ship_gates,
        )
        print(json.dumps({k: v for k, v in scoreboard.items()}, indent=2))
        if args.ship_gates:
            gates = scoreboard.get("gates") or write_ship_gates(
                config.run_dir, scoreboard["suites"]
            )
            # Re-read full payload when only summary was embedded.
            if "pass" not in gates or "failures" not in gates:
                gates = evaluate_ship_gates(scoreboard["suites"])
                write_ship_gates(config.run_dir, scoreboard["suites"])
            return 0 if gates.get("pass") else 8
        # Legacy: fail-under applies to every listed suite (not smoke-only).
        for suite_name in suites:
            metrics = scoreboard["suites"][suite_name]
            code = _check_fail_unders(metrics, args)
            if code:
                return code
        return 0

    metrics = evaluate(config, checkpoint=args.checkpoint)
    summary = {k: v for k, v in metrics.items() if k != "details"}
    print(json.dumps(summary, indent=2))
    return _check_fail_unders(metrics, args)


if __name__ == "__main__":
    raise SystemExit(main())
