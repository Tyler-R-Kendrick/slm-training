"""Audit an existing records.jsonl with the synthetic-integrity gate.

Usage:
    python scripts/audit_synthetic_integrity.py \
        --records outputs/data/train/v1/records.jsonl \
        --output outputs/runs/sde2-02-integrity-audit/audit-20260719.json

The script writes a JSON report with per-record integrity reports and aggregate
counts. It does not modify the source manifest.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

from slm_training.dsl.schema import load_jsonl
from slm_training.harnesses.train_data.integrity import evaluate_integrity
from slm_training.versioning import build_version_stamp


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit synthetic-data integrity")
    parser.add_argument("--records", required=True, type=Path, help="records.jsonl path")
    parser.add_argument("--output", required=True, type=Path, help="audit JSON path")
    parser.add_argument(
        "--held-out-fingerprints",
        type=Path,
        default=None,
        help="optional JSON file with held-out fingerprints to check leakage",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="limit records audited (for quick smoke checks)",
    )
    args = parser.parse_args()

    held_out: set[str] = set()
    if args.held_out_fingerprints:
        data = json.loads(args.held_out_fingerprints.read_text(encoding="utf-8"))
        held_out.update(data if isinstance(data, list) else data.get("fingerprints", []))

    records = load_jsonl(args.records)
    if args.max_records:
        records = records[: args.max_records]

    reports: list[dict[str, Any]] = []
    hard_fail_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()

    for record in records:
        report = evaluate_integrity(record, held_out_fingerprints=held_out)
        reports.append(report.to_dict())
        for check in report.checks:
            status_counts[f"{check.name}:{check.status.value}"] += 1
        hard_fail_counts.update(report.hard_fail_reasons)

    total = len(records)
    passed = sum(1 for r in reports if r["passed"])

    payload = {
        "schema_version": "synthetic_integrity_audit/v1",
        "audit_date": date.today().isoformat().replace("-", ""),
        "records_path": str(args.records),
        "n_records": total,
        "n_passed": passed,
        "n_failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
        "hard_fail_counts": dict(hard_fail_counts),
        "status_counts": dict(status_counts),
        "reports": reports,
        "version_stamp": build_version_stamp(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote integrity audit to {args.output}")
    print(f"records: {total} passed: {passed} failed: {total - passed}")
    if hard_fail_counts:
        print("hard fail reasons:", dict(hard_fail_counts))


if __name__ == "__main__":
    main()
