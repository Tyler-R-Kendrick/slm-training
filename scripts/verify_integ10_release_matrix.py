#!/usr/bin/env python3
"""INTEG-10 / SLM-576: bounded infrastructure release matrix.

Certifies the machine-readable release gate catalog, runs or identity-delegates
existing verify_* / pytest owners under finite process caps, and emits a
content-addressed release-evidence summary. Optional-tool skips are reported
and never counted as success or refutation. Research/default-off pilots are
visible and not required for infrastructure release.

    PYTHONPATH=src uv run python -m scripts.verify_integ10_release_matrix
    PYTHONPATH=src uv run python -m scripts.verify_integ10_release_matrix --check
    PYTHONPATH=src uv run python -m scripts.verify_integ10_release_matrix --write
    PYTHONPATH=src uv run python -m scripts.verify_integ10_release_matrix --execute --write
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.formal.integ10_release_matrix import (
    DEFAULT_RESULTS_RELPATH,
    REPORT_SCHEMA,
    assert_suite_passed,
    run_matrix,
    stable_report,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / DEFAULT_RESULTS_RELPATH,
        help="machine-readable release-evidence summary path",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the live summary to --out",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="require --out to match the live summary (stable fields)",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help=(
            "invoke delegated owners live instead of identity-certifying them; "
            "also runs optional probes when tools are present"
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print the live summary as JSON on stdout",
    )
    args = parser.parse_args(argv)

    report = run_matrix(root=ROOT, execute=args.execute)
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
        live = stable_report(report)
        frozen = stable_report(committed)
        if live != frozen:
            raise SystemExit(
                f"INTEG-10 results drift vs {args.out} "
                "(re-run with --write after an intentional matrix change)"
            )
        print(f"checked {args.out}")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        failed = report.get("required_failed") or []
        opt = report.get("optional_status") or []
        research = report.get("research_default_off") or []
        print(
            f"integ10: {'ok' if report.get('passed') else 'FAILED'} "
            f"required={report.get('required_ok_count')}/{report.get('required_count')} "
            f"failed={failed} optional={len(opt)} research_default_off={len(research)} "
            f"evidence={report.get('evidence_content_address')}"
        )
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
