#!/usr/bin/env python3
"""Run the SLM-228 (RLRG0-01) RL-readiness reward-variance gate stress test.

Exercises the real, unmodified ``assess_rl_readiness`` function
(:mod:`slm_training.autoresearch.rl_gate`) against a battery of reward-sample
arms -- one healthy diverse arm, several degenerate-but-technically-passing
arms, and two negative controls -- holding every other RL-readiness
requirement at a passing state.

Examples:
  python -m scripts.run_slm228_rl_readiness_variance_gate --mode plan-only
  python -m scripts.run_slm228_rl_readiness_variance_gate --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm228_rl_readiness_variance_gate import (
    CANDIDATE_MIN_SAMPLES,
    CANDIDATE_MIN_SPREAD,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    RlReadinessVarianceGateReport,
    render_markdown,
    run_variance_gate_stress_fixture,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm228-rl-readiness-variance-gate-20260721.json"
_DESIGN_MD = "docs/design/iter-slm228-rl-readiness-variance-gate-20260721.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _build_plan_only_payload() -> dict[str, Any]:
    return {
        "schema": "RlReadinessVarianceGateReportV1",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        "status": "plan_only",
        "claim_class": "wiring",
        "hypothesis": (
            "The real assess_rl_readiness reward-variance requirement "
            "approves reward-sample arms with degenerate diversity whenever "
            "every other RL-readiness requirement is independently "
            "satisfied."
        ),
        "falsifier": (
            "Every degenerate arm is rejected by assess_rl_readiness while "
            "the healthy arm and negative controls behave as expected."
        ),
        "candidate_min_samples": CANDIDATE_MIN_SAMPLES,
        "candidate_min_spread": CANDIDATE_MIN_SPREAD,
        "results": [],
        "gate_hash": "",
        "disposition": "inconclusive",
        "disposition_rationale": (
            "Plan-only manifest; run `python -m "
            "scripts.run_slm228_rl_readiness_variance_gate --mode fixture` "
            "to execute."
        ),
        "honest_caveats": [
            "Plan-only: no arm was evaluated.",
        ],
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm228_rl_readiness_variance_gate",
        ),
        "timestamp": _now(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-228 RLRG0-01 RL-readiness reward-variance gate stress test",
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
        f"outputs/runs/slm228-rl-readiness-variance-gate-{_today_yyyymmdd()}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "plan-only":
        payload = _build_plan_only_payload()
    else:
        report = run_variance_gate_stress_fixture(
            run_id=f"{EXPERIMENT_ID}-{_today_yyyymmdd()}"
        )
        payload = report.to_dict()

    run_json = args.output_dir / "slm228_rl_readiness_variance_gate_report.json"
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
            render_markdown(RlReadinessVarianceGateReport.from_dict(payload)),
            encoding="utf-8",
        )

    print(str(run_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
