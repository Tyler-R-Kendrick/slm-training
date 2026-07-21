#!/usr/bin/env python3
"""Run the SLM-234 (CKM0-01) TIES-vs-average merge signal recovery probe.

Builds synthetic parent + sibling checkpoints with a known signal/noise
ground truth (across several deterministic seeds and interference levels),
merges them with the real, unmodified
``slm_training.harness_core.lineage.merge.merge_checkpoints`` under both
``method="average"`` and ``method="ties"``, and scores the merged delta
against the ground truth to test whether TIES-Merging's magnitude-trim +
sign-election + disjoint-merge mechanism recovers a shared consensus update
better than naive parameter averaging as sibling interference rises.

Examples:
  python -m scripts.run_slm234_merge_interference_recovery --mode plan-only
  python -m scripts.run_slm234_merge_interference_recovery --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm234_merge_interference_recovery import (
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    Slm234Report,
    render_markdown,
    run_merge_interference_matrix,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm234-ckm0-01-merge-interference-recovery-20260721.json"
_DESIGN_MD = "docs/design/iter-slm234-ckm0-01-merge-interference-recovery-20260721.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _build_plan_only_payload() -> dict[str, Any]:
    return {
        "schema": "Slm234MergeInterferenceRecoveryReportV1",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        "status": "plan_only",
        "claim_class": "wiring",
        "hypothesis": (
            "merge_checkpoints(method='ties') recovers a synthetic "
            "ground-truth consensus direction/magnitude at least as well as "
            "method='average' as sibling interference (conflict_prob) rises."
        ),
        "seeds": [],
        "n_siblings": 0,
        "density": 0.0,
        "conflict_probs": [],
        "rows": [],
        "metric_summaries": [],
        "disposition": "inconclusive",
        "disposition_rationale": (
            "Plan-only manifest; run `python -m "
            "scripts.run_slm234_merge_interference_recovery --mode fixture` "
            "to execute."
        ),
        "honest_caveats": ["Plan-only: no arm was evaluated."],
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm234_merge_interference_recovery",
        ),
        "timestamp": _now(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-234 CKM0-01 TIES-vs-average merge signal recovery probe",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={"plan-only", "fixture"},
        default="fixture",
        help="Run mode (default: fixture).",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Deterministic ground-truth seeds (default: 0 1 2 3 4).",
    )
    parser.add_argument(
        "--n-siblings",
        type=int,
        default=None,
        help="Number of sibling checkpoints to merge (default: 5).",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=None,
        help="TIES trim density, matches merge_checkpoints default (0.2).",
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
        f"outputs/runs/slm234-ckm0-01-merge-interference-recovery-{_today_yyyymmdd()}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "plan-only":
        payload = _build_plan_only_payload()
    else:
        kwargs: dict[str, Any] = {"run_id": f"{EXPERIMENT_ID}-{_today_yyyymmdd()}"}
        if args.seeds is not None:
            kwargs["seeds"] = tuple(args.seeds)
        if args.n_siblings is not None:
            kwargs["n_siblings"] = args.n_siblings
        if args.density is not None:
            kwargs["density"] = args.density
        report = run_merge_interference_matrix(**kwargs)
        payload = report.to_dict()

    run_json = args.output_dir / "slm234_merge_interference_recovery_report.json"
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
            render_markdown(Slm234Report.from_dict(payload)),
            encoding="utf-8",
        )

    print(str(run_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
