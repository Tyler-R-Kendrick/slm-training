#!/usr/bin/env python3
"""SLM-226 (NCS0-06): Build a SemanticFloorGateV1 permutation-null
seed-stability sweep report, testing whether SLM-225's non-monotonic
n_families=8/16 dip is explained by permutation-null sampling noise (SLM-225's
own honest caveat about its 20-draw, single-seed permutation-null baseline).

Examples:
  python -m scripts.build_floor_gate_seed_stability --describe
  python -m scripts.build_floor_gate_seed_stability --mode plan-only
  python -m scripts.build_floor_gate_seed_stability --mode fixture
  python -m scripts.build_floor_gate_seed_stability --mode fixture \
      --sweep-grid 8 16 --seeds 11 3 7
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm223_semantic_floor_gate import (
    DEFAULT_FLOOR_THRESHOLD,
)
from slm_training.harnesses.experiments.slm226_floor_gate_seed_stability import (
    DEFAULT_RUNS_PER_FAMILY,
    DEFAULT_SEEDS,
    DEFAULT_SWEEP_GRID,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    SeedStabilityReport,
    render_markdown,
    run_seed_stability_fixture,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm226-floor-gate-seed-stability-20260721.json"
_DESIGN_MD = "docs/design/iter-slm226-floor-gate-seed-stability-20260721.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _describe_schema() -> str:
    return """\
SLM-226 SemanticFloorGateV1 permutation-null seed-stability sweep schema

SeedStabilityPoint fields:
  n_families, runs_per_family, synthetic_runs, seeds, margins,
  real_balanced_accuracy, margin_mean, margin_std, margin_min, margin_max,
  seeds_crossing_margin, slm225_disposition, stability, gate_hashes.

SeedStabilityReport fields:
  schema, matrix_set, matrix_version, experiment_id, run_id, status,
  claim_class, hypothesis, falsifier, floor_threshold, sweep_grid,
  runs_per_family, seeds, dip_families, points, disposition,
  disposition_rationale, sweep_hash, honest_caveats, version_stamp, timestamp.

Claim class: wiring / fixture only. Reruns the unmodified SLM-223 gate
pipeline at SLM-225's grid points across multiple permutation-null seeds
(synthetic data held fixed); does not itself certify SemanticFloorGateV1 as a
promotion or ship gate.
"""


def _build_plan_only_payload() -> dict[str, Any]:
    return {
        "schema": "SeedStabilityReportV1",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        "status": "plan_only",
        "claim_class": "wiring",
        "hypothesis": (
            "SLM-225's non-monotonic dip at n_families=8/16 is explained by "
            "permutation-null sampling noise; sweeping the permutation-null "
            "seed at each grid point (synthetic data held fixed) will show the "
            "margin at those points crossing 0.15 for at least one seed."
        ),
        "falsifier": (
            "The margin at n_families=8 and 16 stays below 0.15 for every "
            "swept permutation-null seed."
        ),
        "floor_threshold": DEFAULT_FLOOR_THRESHOLD,
        "sweep_grid": list(DEFAULT_SWEEP_GRID),
        "runs_per_family": DEFAULT_RUNS_PER_FAMILY,
        "seeds": list(DEFAULT_SEEDS),
        "dip_families": [8, 16],
        "points": [],
        "disposition": "inconclusive",
        "disposition_rationale": (
            "Plan-only manifest; run `python -m scripts.build_floor_gate_seed_stability "
            "--mode fixture` to execute."
        ),
        "sweep_hash": "",
        "honest_caveats": [
            "Plan-only: no sweep point was evaluated.",
            "Real checkpoint provenance resolution is required for production use.",
        ],
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm226_floor_gate_seed_stability",
            "harness.experiments.slm225_floor_gate_family_sweep",
            "harness.experiments.slm224_floor_gate_power_sweep",
            "harness.experiments.slm223_semantic_floor_gate",
            "harness.experiments.slm215_spectral_atlas",
            "harness.experiments.slm214_spectral_snapshot",
        ),
        "timestamp": _now(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-226 NCS0-06 SemanticFloorGateV1 permutation-null seed-stability sweep builder",
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
        help=f"n_families values to sweep (default: {list(DEFAULT_SWEEP_GRID)}).",
    )
    parser.add_argument(
        "--runs-per-family",
        type=int,
        default=DEFAULT_RUNS_PER_FAMILY,
        help=f"Runs per family, held fixed across the sweep (default: {DEFAULT_RUNS_PER_FAMILY}).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help=f"Permutation-null seeds to sweep at each grid point (default: {list(DEFAULT_SEEDS)}).",
    )
    parser.add_argument(
        "--dip-families",
        type=int,
        nargs="+",
        default=[8, 16],
        help="n_families values treated as SLM-225's non-monotonic dip points (default: 8 16).",
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

    args.output_dir = args.output_dir or Path(f"outputs/runs/slm226-floor-gate-seed-stability-{_today_yyyymmdd()}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "plan-only":
        payload = _build_plan_only_payload()
    else:
        report = run_seed_stability_fixture(
            sweep_grid=tuple(args.sweep_grid),
            runs_per_family=args.runs_per_family,
            seeds=tuple(args.seeds),
            floor_threshold=args.floor_threshold,
            dip_families=tuple(args.dip_families),
            run_id=f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        )
        payload = report.to_dict()

    run_json = args.output_dir / "slm226_floor_gate_seed_stability_report.json"
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
            render_markdown(SeedStabilityReport.from_dict(payload)),
            encoding="utf-8",
        )

    print(str(run_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
