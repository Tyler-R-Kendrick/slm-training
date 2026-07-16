#!/usr/bin/env python3
"""Metric-ceiling and vocab-coverage diagnostics (F0)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.harnesses.model_build.diagnostic import run_full_diagnostic


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("outputs/data/train/v1"),
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("outputs/data/eval/v1"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/diagnostic_report.json"),
    )
    parser.add_argument(
        "--ltr-max-tokens",
        type=int,
        default=192,
        help="grammar_ltr_max_tokens budget to check against tokenized lengths",
    )
    args = parser.parse_args(argv)

    report = run_full_diagnostic(
        args.train_dir,
        args.test_dir,
        grammar_ltr_max_tokens=args.ltr_max_tokens,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    length = report.get("length_budget") or {}
    report["agentv"] = publish_agentv_evaluation(
        args.out.parent,
        name=f"openui-eval-diagnostic-{args.out.stem}",
        claim="diagnostic_not_ship",
        cases=[
            {
                "id": "eval-diagnostic",
                "criteria": "Complete eval diagnostics without a length-budget failure.",
                "pass": not length or bool(length.get("ok", True)),
                "failures": list(length.get("failures") or []),
                "result": report,
                "metadata": {"honesty": "diagnostic_not_ship"},
            }
        ],
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if length and not length.get("ok", True):
        print(
            "LENGTH_BUDGET_FAIL:",
            "; ".join(length.get("failures") or []),
            flush=True,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
