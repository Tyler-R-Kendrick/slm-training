#!/usr/bin/env python3
"""DSH3-32 (SLM-407): run the operator-inference systems benchmark.

Fixture-scale wiring only -- see the module docstring in
``slm_training.evals.operator_systems_benchmark`` and
``docs/design/dsh3-32-operator-systems-benchmark-20260726.md`` for scope,
the explicit unrun ``full_generation`` arm, and honest caveats. Not a ship
claim.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.evals.operator_systems_benchmark import (
    run_operator_systems_benchmark,
)

ROOT = Path(__file__).resolve().parents[1]


def _summarize(report_dict: dict[str, Any]) -> dict[str, Any]:
    by_arm: dict[str, dict[str, Any]] = {}
    for row in report_dict["rows"]:
        bucket = by_arm.setdefault(
            row["arm_id"],
            {"n": 0, "model_forwards": 0, "dry_runs": 0, "wall_ms": 0.0, "failed": 0},
        )
        bucket["n"] += 1
        bucket["model_forwards"] += row["model_forwards"]
        bucket["dry_runs"] += row["dry_runs"]
        bucket["wall_ms"] += row["wall_ms"]
        bucket["failed"] += int(row["failed"])
    return by_arm


def _markdown(report_dict: dict[str, Any], summary: dict[str, Any]) -> str:
    lines = [
        "# DSH3-32 operator-systems-benchmark run",
        "",
        "Status: fixture-scale wiring evidence; not a ship claim.",
        "",
        f"Arms measured: {', '.join(report_dict['arms_measured'])}",
        f"Arms unrun (explicit, not fabricated): {', '.join(report_dict['arms_unrun']) or 'none'}",
        "",
        "| Arm | Rows | Total model_forwards | Total dry_runs | Total wall_ms | Failed |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm_id, bucket in sorted(summary.items()):
        lines.append(
            f"| `{arm_id}` | {bucket['n']} | {bucket['model_forwards']} | "
            f"{bucket['dry_runs']} | {bucket['wall_ms']:.3f} | {bucket['failed']} |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "dsh3-32-operator-systems-benchmark",
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run with the minimum repeats (2) and skip writing files",
    )
    args = parser.parse_args(argv)
    repeats = 2 if args.dry_run else args.repeats
    report = run_operator_systems_benchmark(repeats=repeats)
    report_dict = report.to_dict()
    report_dict["generated_at"] = datetime.now(timezone.utc).isoformat()
    summary = _summarize(report_dict)
    if args.dry_run:
        print(json.dumps({"arms_measured": report.arms_measured, "rows": len(report.rows)}))
        return 0
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report_dict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(_markdown(report_dict, summary), encoding="utf-8")
    print(json.dumps({"report": str(output_dir / "report.json"), "summary": summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
