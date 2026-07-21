#!/usr/bin/env python3
"""SLM-224 (NCS0-04): Build a SemanticFloorGateV1 statistical-power sweep
report, testing whether SLM-223's no-signal disposition was a fixture-size
power artifact.

Examples:
  python -m scripts.build_floor_gate_power_sweep --describe
  python -m scripts.build_floor_gate_power_sweep --mode plan-only
  python -m scripts.build_floor_gate_power_sweep --mode fixture
  python -m scripts.build_floor_gate_power_sweep --mode fixture \
      --sweep-grid 4 8 16 32
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm224_floor_gate_power_sweep import (
    DEFAULT_SWEEP_GRID,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    PowerSweepReport,
    render_markdown,
    run_power_sweep_fixture,
)
from slm_training.harnesses.experiments.slm223_semantic_floor_gate import (
    DEFAULT_FLOOR_THRESHOLD,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm224-floor-gate-power-sweep-20260721.json"
_DESIGN_MD = "docs/design/iter-slm224-floor-gate-power-sweep-20260721.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _describe_schema() -> str:
    return """\
SLM-224 SemanticFloorGateV1 power-sweep schema

PowerSweepPoint fields:
  synthetic_runs, n_runs, n_families, real_balanced_accuracy,
  permutation_null_mean, margin, disposition, gate_hash.

PowerSweepReport fields:
  schema, matrix_set, matrix_version, experiment_id, run_id, status,
  claim_class, hypothesis, falsifier, floor_threshold, sweep_grid, points,
  disposition, disposition_rationale, sweep_hash, honest_caveats,
  version_stamp, timestamp.

Claim class: wiring / fixture only. Reruns the unmodified SLM-223 gate
pipeline at larger synthetic fixture sizes; does not itself certify
SemanticFloorGateV1 as a promotion or ship gate.
"""


def _build_plan_only_payload() -> dict[str, Any]:
    return {
        "schema": "PowerSweepReportV1",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        "status": "plan_only",
        "claim_class": "wiring",
        "hypothesis": (
            "SLM-223's no-signal disposition is a statistical-power artifact of "
            "the tiny default fixture; sweeping synthetic_runs larger will "
            "recover signal_predictive at some grid point."
        ),
        "falsifier": (
            "The LOFO-vs-permutation-null margin stays below 0.15 across the "
            "full swept grid."
        ),
        "floor_threshold": DEFAULT_FLOOR_THRESHOLD,
        "sweep_grid": list(DEFAULT_SWEEP_GRID),
        "points": [],
        "disposition": "inconclusive",
        "disposition_rationale": (
            "Plan-only manifest; run `python -m scripts.build_floor_gate_power_sweep "
            "--mode fixture` to execute."
        ),
        "sweep_hash": "",
        "honest_caveats": [
            "Plan-only: no sweep point was evaluated.",
            "Real checkpoint provenance resolution is required for production use.",
        ],
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm224_floor_gate_power_sweep",
            "harness.experiments.slm223_semantic_floor_gate",
            "harness.experiments.slm215_spectral_atlas",
            "harness.experiments.slm214_spectral_snapshot",
        ),
        "timestamp": _now(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-224 NCS0-04 SemanticFloorGateV1 power-sweep builder",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={"plan-only", "fixture"},
        default="fixture",
        help="Run mode (default: fixture).",
    )
    parser.add_argument(
        "--sweep-grid",
        type=int,
        nargs="+",
        default=list(DEFAULT_SWEEP_GRID),
        help=f"synthetic_runs values to sweep (default: {list(DEFAULT_SWEEP_GRID)}).",
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

    args.output_dir = args.output_dir or Path(f"outputs/runs/slm224-floor-gate-power-sweep-{_today_yyyymmdd()}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "plan-only":
        payload = _build_plan_only_payload()
    else:
        report = run_power_sweep_fixture(
            sweep_grid=tuple(args.sweep_grid),
            floor_threshold=args.floor_threshold,
            run_id=f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        )
        payload = report.to_dict()

    run_json = args.output_dir / "slm224_floor_gate_power_sweep_report.json"
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
            render_markdown(PowerSweepReport.from_dict(payload)),
            encoding="utf-8",
        )

    print(str(run_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
