#!/usr/bin/env python3
"""Run the SLM-231 (SPV0-05) RoleSlot cardinality dead-field consumption probe.

Exercises the real, unmodified ``PlanOracleSubstitutor`` / ``PlanSeedBuilder``
/ ``OpenUISemanticPlanCompiler`` (:mod:`slm_training.data.semantic_plan`)
twice per fixture record -- once with RoleSlot cardinality left at None
(matching today's production extractor output, per SLM-230), once with a
harness-local candidate cardinality derivation populated -- and diffs the two
arms record-by-record, to check whether the cardinality fields are consumed
anywhere in the current mechanism.

Examples:
  python -m scripts.run_slm231_role_cardinality_dead_field --mode plan-only
  python -m scripts.run_slm231_role_cardinality_dead_field --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm231_role_cardinality_dead_field import (
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    Slm231Report,
    render_markdown,
    run_fixture_matrix,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm231-spv0-05-role-cardinality-dead-field-20260721.json"
_DESIGN_MD = "docs/design/iter-slm231-spv0-05-role-cardinality-dead-field-20260721.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _build_plan_only_payload() -> dict[str, Any]:
    return {
        "schema": "Slm231RoleCardinalityDeadFieldReportV1",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        "status": "plan_only",
        "claim_class": "wiring",
        "hypothesis": (
            "Populating RoleSlot cardinality on the roles+topology oracle "
            "arm produces byte-identical PlanSeedBuilder output to leaving "
            "it None, on every SLM-144 fixture record."
        ),
        "rows": [],
        "mismatches": [],
        "gate_hash": "",
        "disposition": "inconclusive",
        "disposition_rationale": (
            "Plan-only manifest; run `python -m "
            "scripts.run_slm231_role_cardinality_dead_field --mode fixture` "
            "to execute."
        ),
        "honest_caveats": ["Plan-only: no arm was evaluated."],
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm231_role_cardinality_dead_field",
        ),
        "timestamp": _now(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-231 SPV0-05 RoleSlot cardinality dead-field consumption probe",
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
        f"outputs/runs/slm231-spv0-05-role-cardinality-dead-field-{_today_yyyymmdd()}"
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

    run_json = args.output_dir / "slm231_role_cardinality_dead_field_report.json"
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
            render_markdown(Slm231Report.from_dict(payload)),
            encoding="utf-8",
        )

    print(str(run_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
