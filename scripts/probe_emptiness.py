#!/usr/bin/env python3
"""A1 emptiness probe: populated vs minimal-valid program NLL for a checkpoint.

Tests the Grammar-Aligned Decoding / ASAp hypothesis for the valid-but-empty
wall (MODEL_CARD E224-E236): does the model score the empty document cheaper
than the gold populated program, and is that a length-bias (decode-time) or a
content-modeling (training-time) effect? See
``slm_training.evals.emptiness_probe`` and
``docs/design/iter-e248-emptiness-probe-*.md``.

Diagnostic only. Fixture/scratch checkpoints produce wiring evidence, never a
ship claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument(
        "--suites",
        default="smoke,held_out,adversarial,ood",
        help="Comma-separated suites to probe.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--grammar-dsl", default="openui")
    parser.add_argument("--limit", type=int, default=None, help="Cap suite sizes.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Report JSON (default: <checkpoint dir>/emptiness_probe.json)",
    )
    args = parser.parse_args(argv)

    from slm_training.evals.agentv import publish_agentv_evaluation
    from slm_training.evals.emptiness_probe import (
        EmptinessProbeConfig,
        evaluate_emptiness,
    )
    from slm_training.harnesses.model_build.data import load_suite_records
    from slm_training.models.grammar import set_active_dsl
    from slm_training.models.twotower import TwoTowerModel

    set_active_dsl(args.grammar_dsl)
    model = TwoTowerModel.from_checkpoint(args.checkpoint, device=args.device)

    suites = [s.strip() for s in str(args.suites).split(",") if s.strip()]
    by_suite: dict[str, dict] = {}
    for suite in suites:
        try:
            records = load_suite_records(Path(args.test_dir), suite)
        except FileNotFoundError:
            by_suite[suite] = {"n_records": 0, "reason": "suite not found"}
            continue
        if args.limit is not None:
            records = records[: int(args.limit)]
        by_suite[suite] = evaluate_emptiness(
            model, records, config=EmptinessProbeConfig(), dsl=args.grammar_dsl
        )

    report = {
        "checkpoint": str(args.checkpoint),
        "test_dir": str(args.test_dir),
        "grammar_dsl": args.grammar_dsl,
        "by_suite": by_suite,
        "honesty": "diagnostic_not_ship",
    }

    out = args.out or (Path(args.checkpoint).parent / "emptiness_probe.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    report["agentv"] = publish_agentv_evaluation(
        out.parent,
        name="openui-a1-emptiness-probe",
        claim="diagnostic_not_ship",
        cases=[
            {
                "id": f"emptiness-{suite}",
                "criteria": (
                    "Populated program is not scored cheaper than the empty "
                    "program on total NLL (empty_preferred_fraction_total < 0.5)."
                ),
                "pass": (result.get("empty_preferred_fraction_total") or 0.0) < 0.5,
                "result": result,
                "metadata": {"honesty": "diagnostic_not_ship", "suite": suite},
            }
            for suite, result in by_suite.items()
            if result.get("n_records")
        ]
        or [
            {
                "id": "emptiness-no-data",
                "criteria": "At least one suite had scorable records.",
                "pass": False,
                "result": report,
                "metadata": {"honesty": "diagnostic_not_ship"},
            }
        ],
    )
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(out),
                "verdicts": {
                    suite: result.get("verdict")
                    for suite, result in by_suite.items()
                },
                "empty_preferred_fraction_total": {
                    suite: result.get("empty_preferred_fraction_total")
                    for suite, result in by_suite.items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
