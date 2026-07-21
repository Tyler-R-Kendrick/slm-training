#!/usr/bin/env python3
"""Run the SLM-230 (SPV0-04) plan-factor oracle-substitution ceiling matrix.

Exercises the real, unmodified ``PlanOracleSubstitutor`` / ``PlanSeedBuilder``
/ ``OpenUISemanticPlanCompiler`` (:mod:`slm_training.data.semantic_plan`)
against a battery of factor-subset oracle arms on the deterministic SLM-144
fixture corpus, to supply the factor-wise downstream-ceiling evidence
SLM-145's authorization gate and the SLM-160 SPV program disposition both
named as the missing next step.

Examples:
  python -m scripts.run_slm230_plan_factor_ceiling_matrix --mode plan-only
  python -m scripts.run_slm230_plan_factor_ceiling_matrix --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm230_plan_factor_ceiling_matrix import (
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    Slm230Report,
    render_markdown,
    run_fixture_matrix,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm230-spv0-04-plan-factor-ceiling-matrix-20260721.json"
_DESIGN_MD = "docs/design/iter-slm230-spv0-04-plan-factor-ceiling-matrix-20260721.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _build_plan_only_payload() -> dict[str, Any]:
    return {
        "schema": "Slm230PlanFactorCeilingMatrixReportV1",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        "status": "plan_only",
        "claim_class": "wiring",
        "hypothesis": (
            "Oracle-substituting roles+topology (and then bindings) raises "
            "measured downstream ceiling proxies above the no-plan baseline "
            "on the SLM-144 fixture corpus; isolated single-factor arms do "
            "not."
        ),
        "rows": [],
        "gate_hash": "",
        "disposition": "inconclusive",
        "disposition_rationale": (
            "Plan-only manifest; run `python -m "
            "scripts.run_slm230_plan_factor_ceiling_matrix --mode fixture` "
            "to execute."
        ),
        "honest_caveats": ["Plan-only: no arm was evaluated."],
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm230_plan_factor_ceiling_matrix",
        ),
        "timestamp": _now(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-230 SPV0-04 plan-factor oracle-substitution ceiling matrix",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={"plan-only", "fixture"},
        default="fixture",
        help="Run mode (default: fixture).",
    )
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=24,
        help="SLM-144 fixture corpus record count (default: 24).",
    )
    parser.add_argument(
        "--corpus-seed",
        type=int,
        default=0,
        help="SLM-144 fixture corpus deterministic seed (default: 0).",
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
        f"outputs/runs/slm230-spv0-04-plan-factor-ceiling-matrix-{_today_yyyymmdd()}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "plan-only":
        payload = _build_plan_only_payload()
    else:
        report = run_fixture_matrix(
            corpus_size=args.corpus_size,
            corpus_seed=args.corpus_seed,
            run_id=f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        )
        payload = report.to_dict()

    run_json = args.output_dir / "slm230_plan_factor_ceiling_matrix_report.json"
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
            render_markdown(Slm230Report.from_dict(payload)),
            encoding="utf-8",
        )

    print(str(run_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
