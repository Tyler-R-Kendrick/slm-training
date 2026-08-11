#!/usr/bin/env python3
"""EVID-11 / SLM-559: release-blocking formal-evidence mutation suite.

Runs the deterministic red-team matrix (EVID-06/07/09/10 gates), persists a
machine-readable report, and fails closed when any mutation misses its gate
or a positive control regresses.

    PYTHONPATH=src python -m scripts.verify_formal_evidence_mutations
    PYTHONPATH=src python -m scripts.verify_formal_evidence_mutations --write
    PYTHONPATH=src python -m scripts.verify_formal_evidence_mutations --check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.formal.mutation_suite import (
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
        "evid": report.get("evid"),
        "issue": report.get("issue"),
        "passed": report.get("passed"),
        "mutation_count": report.get("mutation_count"),
        "positive_count": report.get("positive_count"),
        "covered_families": report.get("covered_families"),
        "missing_families": report.get("missing_families"),
        "mutations": [
            {
                "case_id": item["case_id"],
                "family": item["family"],
                "gate": item["gate"],
                "capability": item["capability"],
                "expect_reject": item["expect_reject"],
                "rejected": item["rejected"],
                "ok": item["ok"],
            }
            for item in report.get("mutations", [])
        ],
        "positive_controls": [
            {
                "case_id": item["case_id"],
                "family": item["family"],
                "gate": item["gate"],
                "capability": item["capability"],
                "expect_reject": item["expect_reject"],
                "ok": item["ok"],
            }
            for item in report.get("positive_controls", [])
        ],
        "structural_consistency_alone_confers_authority": report.get(
            "structural_consistency_alone_confers_authority"
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
                f"formal evidence mutation results drift vs {args.out} "
                "(re-run with --write after an intentional matrix change)"
            )
        print(f"checked {args.out}")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"formal evidence mutations: passed={report['passed']} "
            f"mutations={report['mutation_count']} "
            f"positives={report['positive_count']} "
            f"families={len(report['covered_families'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
