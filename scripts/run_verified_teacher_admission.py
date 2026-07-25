#!/usr/bin/env python3
"""Run the SLM-296 (AP-013) verifier-filtered teacher-admission wiring fixture.

Example:
  python -m scripts.run_verified_teacher_admission --mode plan-only
  python -m scripts.run_verified_teacher_admission --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm296_verified_teacher_admission import (
    DEFAULT_CONFIG,
    EXPERIMENT_ID,
    TEACHER_MODEL_FAMILY,
    render_markdown,
    run_verified_teacher_admission,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm296-verified-teacher-admission-20260725.json"
_DESIGN_MD = "docs/design/iter-slm296-verified-teacher-admission-20260725.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def _build_payload(
    mode: str, seeds: tuple[int, ...], locked_holdout_seed: int
) -> tuple[dict[str, Any], str]:
    command = (
        f"python -m scripts.run_verified_teacher_admission --mode {mode} "
        f"--seeds {','.join(str(s) for s in seeds)} "
        f"--locked-holdout-seed {locked_holdout_seed}"
    )
    if mode == "plan-only":
        payload: dict[str, Any] = {
            "schema": "Slm296VerifiedTeacherAdmissionManifestV1",
            "experiment_id": EXPERIMENT_ID,
            "status": "plan_only",
            "claim_class": "wiring",
            "recipe": {
                "seeds": list(seeds),
                "components": list(DEFAULT_CONFIG.components or ()),
                "teacher_model_family": TEACHER_MODEL_FAMILY,
                "locked_holdout_seed": locked_holdout_seed,
            },
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm296_verified_teacher_admission",
            ),
            "timestamp": _now(),
        }
        return payload, command
    payload = run_verified_teacher_admission(
        seeds=seeds, locked_holdout_seed=locked_holdout_seed
    )
    return payload, command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-296 AP-013 verifier-filtered teacher-admission fixture",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={"plan-only", "fixture"},
        default="plan-only",
        help="Run mode: plan-only writes the manifest; fixture runs the CPU campaign.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for run artifacts (default: "
            "outputs/runs/slm296-verified-teacher-admission-<YYYYMMDD>)"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=_parse_int_tuple,
        default="0,1,2",
        help="Comma-separated generator seeds for candidate proposal (default: 0,1,2).",
    )
    parser.add_argument(
        "--locked-holdout-seed",
        type=int,
        default=999,
        help="Seed for the disjoint locked-holdout decontamination set (default: 999).",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    output_dir = args.output_dir or Path(
        f"outputs/runs/slm296-verified-teacher-admission-{_today_yyyymmdd()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, command = _build_payload(args.mode, args.seeds, args.locked_holdout_seed)
    payload["schema"] = payload.get(
        "schema", "Slm296VerifiedTeacherAdmissionReportV1"
    )
    payload["claim_class"] = "wiring"
    payload["status"] = payload.get(
        "status", "yield_limitation_no_real_teacher_available"
    )
    payload["timestamp"] = _now()
    if "version_stamp" not in payload:
        payload["version_stamp"] = build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm296_verified_teacher_admission",
        )

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    run_json = output_dir / "slm296_verified_teacher_admission_report.json"
    run_json.write_text(report_text, encoding="utf-8")

    if args.mode == "fixture":
        root = Path(__file__).resolve().parents[1]
        json_path = root / _DESIGN_JSON
        md_path = root / _DESIGN_MD
        json_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(report_text, encoding="utf-8")
        md_path.write_text(render_markdown(payload), encoding="utf-8")

    print(str(run_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
