#!/usr/bin/env python3
"""SLM-223 (NCS0-03): Build a SemanticFloorGateV1 pre-eval gate report from an
existing or synthetic SpectralAtlasV1 (SLM-215).

Examples:
  python -m scripts.build_semantic_floor_gate --describe
  python -m scripts.build_semantic_floor_gate --mode plan-only
  python -m scripts.build_semantic_floor_gate --mode fixture --synthetic-runs 4
  python -m scripts.build_semantic_floor_gate --mode fixture \
      --reports-dir outputs/runs --floor-threshold 0.5
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm223_semantic_floor_gate import (
    DEFAULT_FLOOR_THRESHOLD,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    SemanticFloorGateReport,
    render_markdown,
    run_semantic_floor_gate_fixture,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm223-semantic-floor-gate-20260721.json"
_DESIGN_MD = "docs/design/iter-slm223-semantic-floor-gate-20260721.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _describe_schema() -> str:
    return """\
SLM-223 SemanticFloorGateV1 schema

FloorGateRunRow fields:
  gate_version, run_id, family, n_matrices, mean_alpha_z, weighted_alpha_z,
  parse_rate, floor_label, gate_flag, correct, fold.

SemanticFloorGateReport fields:
  schema, matrix_set, matrix_version, experiment_id, run_id, status,
  claim_class, hypothesis, falsifier, floor_threshold, atlas_hash, rows,
  n_runs, n_families, full_role_weights, real_balanced_accuracy,
  permutation_null, gate_hash, disposition, disposition_rationale,
  honest_caveats, version_stamp, timestamp.

Claim class: wiring / fixture only. Diagnostic pre-screen candidate, not a
promotion or ship gate.
"""


def _build_plan_only_payload() -> dict[str, Any]:
    return {
        "schema": "SemanticFloorGateReportV1",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        "status": "plan_only",
        "claim_class": "wiring",
        "hypothesis": (
            "A role-weighted aggregate of SpectralAtlasV1 alpha_z, calibrated "
            "out-of-fold, can flag floor-risk checkpoints better than a "
            "label-permuted control."
        ),
        "falsifier": (
            "Leave-one-family-out balanced accuracy does not exceed the "
            "permutation-null mean by the required margin."
        ),
        "floor_threshold": DEFAULT_FLOOR_THRESHOLD,
        "atlas_hash": "",
        "rows": [],
        "n_runs": 0,
        "n_families": 0,
        "full_role_weights": {},
        "real_balanced_accuracy": None,
        "permutation_null": {},
        "gate_hash": "",
        "disposition": "inconclusive",
        "disposition_rationale": (
            "Plan-only manifest; run `python -m scripts.build_semantic_floor_gate "
            "--mode fixture` to execute."
        ),
        "honest_caveats": [
            "Plan-only: no gate was calibrated or evaluated.",
            "Real checkpoint provenance resolution is required for production use.",
        ],
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm223_semantic_floor_gate",
            "harness.experiments.slm215_spectral_atlas",
            "harness.experiments.slm214_spectral_snapshot",
        ),
        "timestamp": _now(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-223 NCS0-03 SemanticFloorGateV1 builder",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={"plan-only", "fixture"},
        default="fixture",
        help="Run mode (default: fixture).",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        help="Root directory to scan for slm214_spectral_report.json files (passed through to the SpectralAtlasV1 builder).",
    )
    parser.add_argument(
        "--synthetic-runs",
        type=int,
        default=4,
        help="Number of synthetic fixture runs when no reports are found (default: 4).",
    )
    parser.add_argument(
        "--floor-threshold",
        type=float,
        default=DEFAULT_FLOOR_THRESHOLD,
        help=f"parse_rate below this value counts as floor-risk (default: {DEFAULT_FLOOR_THRESHOLD}).",
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
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print the schema and exit.",
    )

    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    if args.describe:
        print(_describe_schema())
        return 0

    args.output_dir = args.output_dir or Path(f"outputs/runs/slm223-semantic-floor-gate-{_today_yyyymmdd()}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "plan-only":
        payload = _build_plan_only_payload()
    else:
        report = run_semantic_floor_gate_fixture(
            reports_dir=args.reports_dir,
            synthetic_runs=args.synthetic_runs,
            floor_threshold=args.floor_threshold,
            run_id=f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        )
        payload = report.to_dict()

    run_json = args.output_dir / "slm223_semantic_floor_gate_report.json"
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
            render_markdown(SemanticFloorGateReport.from_dict(payload)),
            encoding="utf-8",
        )

    print(str(run_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
