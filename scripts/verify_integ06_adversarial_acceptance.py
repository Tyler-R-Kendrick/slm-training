#!/usr/bin/env python3
"""INTEG-06 / SLM-573: release-blocking adversarial end-to-end acceptance.

Runs the content-addressed scenario matrix over existing EVID/KERN/HARN/INTEG
gates, persists a machine-readable report, and fails closed when any
adversarial case misses its gate or a positive control regresses.

    PYTHONPATH=src python -m scripts.verify_integ06_adversarial_acceptance
    PYTHONPATH=src python -m scripts.verify_integ06_adversarial_acceptance --write
    PYTHONPATH=src python -m scripts.verify_integ06_adversarial_acceptance --check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.formal.integ06_acceptance import (
    DEFAULT_RESULTS_RELPATH,
    REPORT_SCHEMA,
    assert_suite_passed,
    run_suite,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]


def _stable_report(report: dict) -> dict:
    """Keep outcome fields only — stamps move with every commit/bump."""

    return {
        "schema": report.get("schema"),
        "integ": report.get("integ"),
        "issue": report.get("issue"),
        "passed": report.get("passed"),
        "adversarial_count": report.get("adversarial_count"),
        "positive_count": report.get("positive_count"),
        "covered_scenarios": report.get("covered_scenarios"),
        "missing_scenarios": report.get("missing_scenarios"),
        "adversarial_cases": [
            {
                "case_id": item["case_id"],
                "scenario": item["scenario"],
                "gate": item["gate"],
                "layer": item["layer"],
                "expect_reject": item["expect_reject"],
                "rejected": item["rejected"],
                "ok": item["ok"],
            }
            for item in report.get("adversarial_cases", [])
        ],
        "positive_controls": [
            {
                "case_id": item["case_id"],
                "scenario": item["scenario"],
                "gate": item["gate"],
                "layer": item["layer"],
                "expect_reject": item["expect_reject"],
                "ok": item["ok"],
            }
            for item in report.get("positive_controls", [])
        ],
        "no_failure_becomes_destructive_authority": report.get(
            "no_failure_becomes_destructive_authority"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / DEFAULT_RESULTS_RELPATH,
        help="machine-readable report path",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the live report to --out",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="require --out to match the live report (stable fields)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the live report as JSON on stdout",
    )
    args = parser.parse_args(argv)

    report = run_suite(root=ROOT)
    assert_suite_passed(report)

    if args.write:
        write_report(report, args.out)
        print(f"wrote {args.out}")

    if args.check:
        if not args.out.is_file():
            raise SystemExit(f"missing committed results: {args.out}")
        committed = json.loads(args.out.read_text(encoding="utf-8"))
        if committed.get("schema") != REPORT_SCHEMA:
            raise SystemExit(f"bad committed schema: {committed.get('schema')!r}")
        live = _stable_report(report)
        frozen = _stable_report(committed)
        if live != frozen:
            raise SystemExit(
                f"INTEG-06 results drift vs {args.out} "
                "(re-run with --write after an intentional matrix change)"
            )
        print(f"checked {args.out}")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"INTEG-06 adversarial acceptance: passed={report['passed']} "
            f"adversarial={report['adversarial_count']} "
            f"positives={report['positive_count']} "
            f"scenarios={len(report['covered_scenarios'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
