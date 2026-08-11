#!/usr/bin/env python3
"""INTEG-07 / SLM-561: activation-preflight recall + false-skip gate.

Replays the frozen known-activating / inactive / unknown / adversarial set
through INTEG-02 ``admit_mechanism_treatment`` and fails closed on any false
skip of a known activating case (recall must be 100%).

    PYTHONPATH=src uv run python -m scripts.verify_integ07_activation_preflight
    PYTHONPATH=src uv run python -m scripts.verify_integ07_activation_preflight --write
    PYTHONPATH=src uv run python -m scripts.verify_integ07_activation_preflight --check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.formal.integ07_activation_preflight import (
    DEFAULT_RESULTS_RELPATH,
    REPORT_SCHEMA,
    assert_benchmark_passed,
    run_benchmark,
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
        "metrics": report.get("metrics"),
        "acceptance": {
            "known_activating_recall_min": (report.get("acceptance") or {}).get(
                "known_activating_recall_min"
            ),
            "recall_gate_passed": (report.get("acceptance") or {}).get(
                "recall_gate_passed"
            ),
            "unknown_must_run": (report.get("acceptance") or {}).get(
                "unknown_must_run"
            ),
            "adversarial_kinds_required": (report.get("acceptance") or {}).get(
                "adversarial_kinds_required"
            ),
            "adversarial_kinds_seen": (report.get("acceptance") or {}).get(
                "adversarial_kinds_seen"
            ),
            "adversarial_kinds_missing": (report.get("acceptance") or {}).get(
                "adversarial_kinds_missing"
            ),
        },
        "identities": {
            "revmath_corpus_id": (report.get("identities") or {}).get(
                "revmath_corpus_id"
            ),
            "revmath_corpus_sha256": (report.get("identities") or {}).get(
                "revmath_corpus_sha256"
            ),
            "trigger_theorem_refs": (report.get("identities") or {}).get(
                "trigger_theorem_refs"
            ),
            "locked_corpus_ids": (report.get("identities") or {}).get(
                "locked_corpus_ids"
            ),
            "kernel": (report.get("identities") or {}).get("kernel"),
            "admission": (report.get("identities") or {}).get("admission"),
        },
        "cohort_counts": report.get("cohort_counts"),
        "cases": [
            {
                "case_id": item["case_id"],
                "cohort": item["cohort"],
                "ground_truth": item["ground_truth"],
                "expect_disposition": item["expect_disposition"],
                "raw_disposition": item["raw_disposition"],
                "effective_disposition": item["effective_disposition"],
                "identity_ok": item["identity_ok"],
                "ok": item["ok"],
                "false_skip": item["false_skip"],
                "theorem_ref": item["theorem_ref"],
                "corpus_id": item["corpus_id"],
                "locked_corpus_id": item["locked_corpus_id"],
                "compute_saved": item["compute_saved"],
                "adversarial_kind": item.get("adversarial_kind"),
            }
            for item in report.get("cases", [])
        ],
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
        help="persist the live report to --out",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="require --out to match the live stable report and pass the gate",
    )
    args = parser.parse_args(argv)

    report = run_benchmark(root=ROOT)
    assert_benchmark_passed(report)

    if args.write:
        write_report(report, path=args.out, root=ROOT)
        print(f"wrote {args.out}")

    if args.check:
        if not args.out.is_file():
            raise SystemExit(f"missing committed results at {args.out}")
        committed = json.loads(args.out.read_text(encoding="utf-8"))
        if committed.get("schema") != REPORT_SCHEMA:
            raise SystemExit(
                f"committed schema {committed.get('schema')!r} != {REPORT_SCHEMA!r}"
            )
        live_stable = _stable_report(report)
        committed_stable = _stable_report(committed)
        if live_stable != committed_stable:
            raise SystemExit(
                f"committed results at {args.out} diverge from live benchmark; "
                "re-run with --write"
            )
        print(
            "ok "
            f"recall={report['metrics']['known_activating_recall']} "
            f"specificity={report['metrics']['provably_inactive_specificity']} "
            f"fallback={report['metrics']['unknown_fallback_rate']} "
            f"compute_saved={report['metrics']['compute_saved']}"
        )
        return 0

    print(json.dumps(_stable_report(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
