#!/usr/bin/env python3
"""Run the SLM-235 (AEP0-01) annotation export pairing-mechanism consistency
probe.

Exercises the real, unmodified live annotation-persist path
(``FileAnnotationStore.persist`` -> ``persist_annotation`` ->
``maybe_append_preference_pair``, exactly as the FastAPI ``/api/annotate``
endpoint invokes it) followed by the real, unmodified batch export path
(``export_to_preference_pairs``, exactly as the documented
``slm annotations export`` CLI invokes it) against the same default pairs
path, to check whether batch export silently discards incrementally
persisted preference pairs.

Examples:
  python -m scripts.run_slm235_annotation_export_pairing_consistency --mode plan-only
  python -m scripts.run_slm235_annotation_export_pairing_consistency --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm235_annotation_export_pairing_consistency import (
    AnnotationExportPairingReport,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    render_markdown,
    run_pairing_consistency_fixture,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm235-aep0-01-annotation-export-pairing-consistency-20260721.json"
_DESIGN_MD = "docs/design/iter-slm235-aep0-01-annotation-export-pairing-consistency-20260721.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _build_plan_only_payload() -> dict[str, Any]:
    return {
        "schema": "AnnotationExportPairingReportV1",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_id": f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
        "status": "plan_only",
        "claim_class": "wiring",
        "hypothesis": (
            "The live/incremental and batch/export preference-pairing "
            "algorithms over annotation feedback differ, and because "
            "write_pairs replaces rather than appends, running the "
            "documented export CLI against the shared default pairs path "
            "silently discards incrementally persisted preference pairs."
        ),
        "falsifier": (
            "Every multi-flip scenario retains its full incremental pair "
            "set after batch export, or export_to_preference_pairs appends "
            "instead of replacing, or the two default pairs paths differ."
        ),
        "results": [],
        "static_shared_default_path_audit": {},
        "gate_hash": "",
        "disposition": "inconclusive",
        "disposition_rationale": (
            "Plan-only manifest; run `python -m "
            "scripts.run_slm235_annotation_export_pairing_consistency "
            "--mode fixture` to execute."
        ),
        "honest_caveats": [
            "Plan-only: no scenario was evaluated.",
        ],
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm235_annotation_export_pairing_consistency",
        ),
        "timestamp": _now(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-235 AEP0-01 annotation export pairing-mechanism consistency probe",
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
        f"outputs/runs/slm235-annotation-export-pairing-consistency-{_today_yyyymmdd()}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "plan-only":
        payload = _build_plan_only_payload()
    else:
        report = run_pairing_consistency_fixture(
            run_id=f"{EXPERIMENT_ID}-{_today_yyyymmdd()}"
        )
        payload = report.to_dict()

    run_json = args.output_dir / "slm235_annotation_export_pairing_consistency_report.json"
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
            render_markdown(AnnotationExportPairingReport.from_dict(payload)),
            encoding="utf-8",
        )

    print(str(run_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
