#!/usr/bin/env python3
"""SLM-215 (NCS0-02): Build a null-calibrated SpectralAtlasV1 from existing
SpectralSnapshotV1 reports and/or synthetic fixtures.

Examples:
  python -m scripts.build_spectral_atlas --describe
  python -m scripts.build_spectral_atlas --mode plan-only
  python -m scripts.build_spectral_atlas --mode fixture --synthetic-runs 4
  python -m scripts.build_spectral_atlas --mode fixture \
      --reports-dir outputs/runs
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm215_spectral_atlas import (
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    SpectralAtlasReport,
    render_markdown,
    run_spectral_atlas_fixture,
)
from slm_training.harnesses.experiments.semantic_floor_gate import (
    DEFAULT_GATE_PATH,
    load_semantic_floor_gate,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm215-spectral-atlas-20260721.json"
_DESIGN_MD = "docs/design/iter-slm215-spectral-atlas-20260721.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _describe_schema() -> str:
    return """\
SLM-215 SpectralAtlasV1 schema

SpectralAtlasV1 row fields:
  atlas_version, run_id, checkpoint_role, checkpoint_sha, matrix_id,
  semantic_role, shape, hill_alpha, alpha_z, randomized_esd_distance,
  stable_rank, effective_rank, spectral_entropy, steps, seen_target_tokens,
  last_loss, weighted_nll, parse_rate, meaningful_rate, fidelity, structure,
  reward, family, claim_scope.

SpectralAtlasReport fields:
  schema, matrix_set, matrix_version, experiment_id, run_id, status,
  claim_class, hypothesis, falsifier, rows, source_reports,
  unresolvable_local_history, n_rows, n_runs, n_families, role_summaries,
  signal, atlas_hash, disposition, disposition_rationale, honest_caveats,
  version_stamp, timestamp.

Claim class: wiring / fixture only. No model-quality or promotion claim.
"""


def _build_plan_only_payload(command: str) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    floor_gate = load_semantic_floor_gate(root / DEFAULT_GATE_PATH)
    return {
        "schema": "SpectralAtlasReportV1",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        "status": "plan_only",
        "claim_class": "wiring",
        "hypothesis": run_spectral_atlas_fixture.__doc__ or "",
        "falsifier": (
            "After holding out complete experiment families, calibrated spectral features "
            "add no descriptive value beyond simple baselines."
        ),
        "rows": [],
        "source_reports": [],
        "unresolvable_local_history": [],
        "n_rows": 0,
        "n_runs": 0,
        "n_families": 0,
        "role_summaries": {},
        "signal": {},
        "atlas_hash": "",
        "floor_gate_ref": DEFAULT_GATE_PATH,
        "floor_gate_hash": floor_gate.gate_hash,
        "floor_gate_verdict": floor_gate.verdict,
        "disposition": "inconclusive",
        "disposition_rationale": "Plan-only manifest; run `python -m scripts.build_spectral_atlas --mode fixture` to execute.",
        "honest_caveats": [
            "Plan-only: no atlas was built.",
            "Real checkpoint provenance resolution is required for production use.",
        ],
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm215_spectral_atlas",
            "harness.experiments.slm214_spectral_snapshot",
            "harness.experiments.semantic_floor_gate",
        ),
        "timestamp": _now(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-215 NCS0-02 SpectralAtlasV1 builder",
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
        help="Root directory to scan for slm214_spectral_report.json files.",
    )
    parser.add_argument(
        "--synthetic-runs",
        type=int,
        default=4,
        help="Number of synthetic fixture runs when no reports are found (default: 4).",
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

    args.output_dir = args.output_dir or Path(f"outputs/runs/slm215-spectral-atlas-{_today_yyyymmdd()}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "plan-only":
        payload = _build_plan_only_payload("python -m scripts.build_spectral_atlas --mode plan-only")
    else:
        report = run_spectral_atlas_fixture(
            reports_dir=args.reports_dir,
            synthetic_runs=args.synthetic_runs,
            run_id=f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        )
        report.to_json(args.output_dir / "slm215_spectral_atlas_report.json")
        payload = report.to_dict()

    run_json = args.output_dir / "slm215_spectral_atlas_report.json"
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
            render_markdown(SpectralAtlasReport.from_dict(payload)),
            encoding="utf-8",
        )

    print(str(run_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
