#!/usr/bin/env python3
"""Run the SLM-170 (SDE2-03) exposure-targeted rare-action sampling fixture.

Example:
  python -m scripts.run_slm170_exposure_targeted_rare_action_fixture --mode plan-only
  python -m scripts.run_slm170_exposure_targeted_rare_action_fixture --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm170_exposure_targeted_rare_action import (
    MATRIX_SET,
    RareActionArm,
    RareActionReport,
    build_cells,
    render_markdown,
    run_fixture_campaign,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm170-exposure-targeted-rare-action-20260720.json"
_DESIGN_MD = "docs/design/iter-slm170-exposure-targeted-rare-action-20260720.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def _build_payload(
    mode: str,
    output_dir: Path,
    seeds: tuple[int, ...],
    total_decision_budget: int,
    per_root_cap: int,
    per_template_cap: int,
    max_importance_weight: float,
) -> tuple[dict[str, Any], str]:
    cells = build_cells(
        seeds,
        total_decision_budget=total_decision_budget,
        per_root_cap=per_root_cap,
        per_template_cap=per_template_cap,
        max_importance_weight=max_importance_weight,
    )

    if mode == "plan-only":
        payload: dict[str, Any] = {
            "schema": "Slm170ExposureTargetedRareActionManifestV1",
            "matrix_set": MATRIX_SET,
            "matrix_version": "sde2-03-v1",
            "experiment_id": "slm170-exposure-targeted-rare-action",
            "status": "plan_only",
            "claim_class": "wiring",
            "cells": [cell.to_dict() for cell in cells],
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm170_exposure_targeted_rare_action",
            ),
            "timestamp": _now(),
        }
        command = (
            "python -m scripts.run_slm170_exposure_targeted_rare_action_fixture --mode plan-only"
        )
        return payload, command

    report = run_fixture_campaign(
        cells=cells,
        run_id=f"slm170-exposure-targeted-rare-action-{_today_yyyymmdd()}",
        output_dir=output_dir,
        seeds=seeds,
    )
    payload = report.to_dict()
    command = "python -m scripts.run_slm170_exposure_targeted_rare_action_fixture --mode fixture"
    return payload, command


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    status = payload.get("status", "fixture")
    if status == "plan_only":
        cells = [RareActionArm.from_dict(c) for c in payload["cells"]]
        lines = [
            "# SLM-170 (SDE2-03): exposure-targeted rare-action sampling plan",
            "",
            "**Claim class:** wiring / fixture only",
            "",
            "**Run date:** 2026-07-20",
            "",
            "**Machine-readable result:** ["
            "`iter-slm170-exposure-targeted-rare-action-20260720.json`"
            "](iter-slm170-exposure-targeted-rare-action-20260720.json)",
            "",
            "This is a plan-only manifest. The sampling arms, targets, caps, metrics, "
            "and analysis plumbing are wired; run `--mode fixture` to execute the "
            "CPU-only simulation.",
            "",
            "## Hypothesis",
            "",
            "Exposure-targeted sampling with bounded importance weights and diversity "
            "caps increases rare-action exposure within a fixed total decision budget.",
            "",
            "## Falsifier",
            "",
            "Exposure-targeted sampling fails to increase rare-action exposure or "
            "violates the total budget / diversity caps.",
            "",
            "## Sampling arms",
            "",
            "| arm_id | arm_name | policy | seed | total_decision_budget | per_root_cap | per_template_cap | max_importance_weight |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for cell in cells:
            lines.append(
                f"| {cell.arm_id} | {cell.arm_name} | {cell.policy} | {cell.seed} | "
                f"{cell.total_decision_budget} | {cell.per_root_cap} | "
                f"{cell.per_template_cap} | {cell.max_importance_weight} |"
            )
        lines.extend(
            [
                "",
                "## Exact command",
                "",
                f"```bash\n{command}\n```",
                "",
            ]
        )
        return "\n".join(lines)

    report = RareActionReport.from_dict(payload)
    return render_markdown(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-170 SDE2-03 exposure-targeted rare-action sampling fixture",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={"plan-only", "fixture"},
        default="plan-only",
        help="Run mode: plan-only writes the manifest; fixture runs the CPU simulation.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for run artifacts (default: outputs/runs/slm170-exposure-targeted-rare-action-<YYYYMMDD>)",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_int_tuple,
        default="0,1,2",
        help="Comma-separated random seeds for fixture mode (default: 0,1,2).",
    )
    parser.add_argument(
        "--total-decision-budget",
        type=int,
        default=64,
        help="Total decision budget for each sampling arm (default: 64).",
    )
    parser.add_argument(
        "--per-root-cap",
        type=int,
        default=4,
        help="Max records per root parent for exposure-targeted arms (default: 4).",
    )
    parser.add_argument(
        "--per-template-cap",
        type=int,
        default=4,
        help="Max records per prompt template for exposure-targeted arms (default: 4).",
    )
    parser.add_argument(
        "--max-importance-weight",
        type=float,
        default=10.0,
        help="Cap on per-action importance weights (default: 10.0).",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    output_dir = (
        args.output_dir
        or Path(
            f"outputs/runs/slm170-exposure-targeted-rare-action-{_today_yyyymmdd()}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, command = _build_payload(
        args.mode,
        output_dir,
        args.seeds,
        args.total_decision_budget,
        args.per_root_cap,
        args.per_template_cap,
        args.max_importance_weight,
    )
    payload["schema"] = "Slm170ExposureTargetedRareActionReportV1"
    payload["claim_class"] = "wiring"
    payload["status"] = payload.get("status", "fixture")
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    run_json = output_dir / "slm170_exposure_targeted_rare_action_report.json"
    run_json.write_text(report_text, encoding="utf-8")

    if args.mode == "fixture":
        root = Path(__file__).resolve().parents[1]
        json_path = root / _DESIGN_JSON
        md_path = root / _DESIGN_MD
        json_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(report_text, encoding="utf-8")

        command_line = command
        if args.output_dir is not None:
            command_line += f" --output-dir {output_dir}"
        md_path.write_text(_build_markdown(payload, command_line), encoding="utf-8")

    print(str(run_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
