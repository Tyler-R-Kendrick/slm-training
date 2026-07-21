#!/usr/bin/env python3
"""Run the SLM-233 (DTL0-01) distillation P2 trace-selection eval-holdout
leakage gate stress test.

Exercises the real, unmodified ``build_test_data`` (test-data phase) plus
``DecodeTraceRecorder.finalize`` / ``TraceStore.append`` / ``select_traces``
(distillation P1/P2) against a freshly-built, genuine fixture-scale eval
bundle to check whether self-distillation trace selection enforces the
documented "selection data stays disjoint from frozen evals; never train on
held-out benchmark traces" invariant.

Examples:
  python -m scripts.run_slm233_distill_select_eval_leakage_gate --mode plan-only
  python -m scripts.run_slm233_distill_select_eval_leakage_gate --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm233_distill_select_eval_leakage_gate import (
    EXPERIMENT_ID,
    FIXTURE_SUITES,
    MATRIX_SET,
    MATRIX_VERSION,
    DistillSelectEvalLeakageGateReport,
    render_markdown,
    run_eval_leakage_gate_stress_fixture,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm233-dtl0-01-distill-select-eval-leakage-gate-20260721.json"
_DESIGN_MD = "docs/design/iter-slm233-dtl0-01-distill-select-eval-leakage-gate-20260721.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _build_plan_only_payload() -> dict[str, Any]:
    return {
        "schema": "DistillSelectEvalLeakageGateReportV1",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        "status": "plan_only",
        "claim_class": "wiring",
        "hypothesis": (
            "The real distillation P2 select_traces / filter_traces pipeline "
            "has no discrimination against traces sourced from a frozen eval "
            "suite, unlike the equivalent train-side check test-data already "
            "enforces."
        ),
        "falsifier": (
            "select_traces / filter_traces reject, warn on, drop, or "
            "otherwise select eval-suite-sourced traces differently from "
            "ordinary train-sourced traces, or the distill.select module "
            "already reads meta.source_suite."
        ),
        "fixture_suites": list(FIXTURE_SUITES),
        "results": [],
        "control_selection_rate": 0.0,
        "static_source_suite_audit": {},
        "gate_hash": "",
        "disposition": "inconclusive",
        "disposition_rationale": (
            "Plan-only manifest; run `python -m "
            "scripts.run_slm233_distill_select_eval_leakage_gate --mode fixture` "
            "to execute."
        ),
        "honest_caveats": [
            "Plan-only: no arm was evaluated.",
        ],
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm233_distill_select_eval_leakage_gate",
        ),
        "timestamp": _now(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-233 DTL0-01 distillation trace-selection eval-holdout leakage gate stress test",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={"plan-only", "fixture"},
        default="fixture",
        help="Run mode (default: fixture).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for run artifacts.",
    )
    parser.add_argument(
        "--write-design-docs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write design docs in fixture mode (default: True).",
    )
    parser.add_argument("--design-json", type=Path, default=None)
    parser.add_argument("--design-md", type=Path, default=None)

    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    args.output_dir = args.output_dir or Path(
        f"outputs/runs/slm233-distill-select-eval-leakage-gate-{_today_yyyymmdd()}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "plan-only":
        payload = _build_plan_only_payload()
    else:
        report = run_eval_leakage_gate_stress_fixture(
            run_id=f"{EXPERIMENT_ID}-{_today_yyyymmdd()}"
        )
        payload = report.to_dict()

    run_json = args.output_dir / "slm233_distill_select_eval_leakage_gate_report.json"
    run_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    if args.mode == "fixture" and args.write_design_docs:
        root = Path(__file__).resolve().parents[1]
        json_path = args.design_json or root / _DESIGN_JSON
        md_path = args.design_md or root / _DESIGN_MD
        json_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(
            render_markdown(DistillSelectEvalLeakageGateReport.from_dict(payload)),
            encoding="utf-8",
        )

    print(str(run_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
