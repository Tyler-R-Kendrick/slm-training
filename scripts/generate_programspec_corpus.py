#!/usr/bin/env python3
"""Run the SLM-267 (VSD2-02) ProgramSpec coverage-scaling wiring fixture.

Example:
  python -m scripts.generate_programspec_corpus --mode plan-only
  python -m scripts.generate_programspec_corpus --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm267_programspec_coverage_scaling import (
    DEFAULT_CONFIG,
    EXPERIMENT_ID,
    render_markdown,
    run_coverage_scaling_campaign,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm267-programspec-coverage-scaling-20260725.json"
_DESIGN_MD = "docs/design/iter-slm267-programspec-coverage-scaling-20260725.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _build_payload(
    mode: str, target_count: int, seed: int, shards: int
) -> tuple[dict[str, Any], str]:
    command = (
        f"python -m scripts.generate_programspec_corpus --mode {mode} "
        f"--target-count {target_count} --seed {seed} --shards {shards}"
    )
    if mode == "plan-only":
        payload: dict[str, Any] = {
            "schema": "Slm267ProgramspecCoverageScalingManifestV1",
            "experiment_id": EXPERIMENT_ID,
            "status": "plan_only",
            "claim_class": "wiring",
            "recipe": {
                "target_count": target_count,
                "seed": seed,
                "shards": shards,
                "components": list(DEFAULT_CONFIG.components or ()),
            },
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm267_programspec_coverage_scaling",
            ),
            "timestamp": _now(),
        }
        return payload, command
    payload = run_coverage_scaling_campaign(
        target_count=target_count, seed=seed, shards=shards
    )
    return payload, command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-267 VSD2-02 ProgramSpec coverage-scaling fixture",
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
            "outputs/runs/slm267-programspec-coverage-scaling-<YYYYMMDD>)"
        ),
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=80,
        help="Bounded per-arm program budget (default: 80; fixture scale only).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Global seed (default: 0).")
    parser.add_argument(
        "--shards", type=int, default=2, help="Deterministic shard count (default: 2)."
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    output_dir = args.output_dir or Path(
        f"outputs/runs/slm267-programspec-coverage-scaling-{_today_yyyymmdd()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, command = _build_payload(
        args.mode, args.target_count, args.seed, args.shards
    )
    payload["schema"] = payload.get("schema", "Slm267ProgramspecCoverageScalingReportV1")
    payload["claim_class"] = "wiring"
    payload["status"] = payload.get("status", "inconclusive")
    payload["timestamp"] = _now()
    if "version_stamp" not in payload:
        payload["version_stamp"] = build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm267_programspec_coverage_scaling",
        )

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    run_json = output_dir / "slm267_programspec_coverage_scaling_report.json"
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
