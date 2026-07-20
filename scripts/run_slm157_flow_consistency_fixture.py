#!/usr/bin/env python3
"""Run the SLM-157 (SPV3-04) flow / consistency / trajectory-imitation fixture.

Example:
  python -m scripts.run_slm157_flow_consistency_fixture --mode plan-only
  python -m scripts.run_slm157_flow_consistency_fixture --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm157_flow_consistency import (
    CommonConfig,
    FlowConsistencyReport,
    FlowManifest,
    build_manifest,
    render_markdown,
    run_fixture_campaign,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm157-flow-consistency-20260720.json"
_DESIGN_MD = "docs/design/iter-slm157-flow-consistency-20260720.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def _build_payload(
    mode: str,
    output_dir: Path,
    n_records: int,
    steps: tuple[int, ...],
    seeds: tuple[int, ...],
) -> tuple[dict[str, Any], str]:
    manifest = build_manifest()

    if mode == "plan-only":
        payload: dict[str, Any] = {
            "schema": "Slm157FlowConsistencyManifestV1",
            "matrix_set": manifest.matrix_set,
            "matrix_version": manifest.matrix_version,
            "experiment_id": manifest.experiment_id,
            "status": "plan_only",
            "claim_class": "wiring",
            "manifest": manifest.to_dict(),
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm157_flow_consistency",
            ),
            "timestamp": _now(),
        }
        command = (
            "python -m scripts.run_slm157_flow_consistency_fixture --mode plan-only"
        )
        return payload, command

    cfg = CommonConfig(
        seeds=seeds,
        n_records=n_records,
        steps_list=steps,
    )
    manifest = FlowManifest(
        matrix_set=manifest.matrix_set,
        matrix_version=manifest.matrix_version,
        experiment_id=manifest.experiment_id,
        hypothesis=manifest.hypothesis,
        falsifier=manifest.falsifier,
        common_config=cfg,
        arms=manifest.arms,
        claim_class=manifest.claim_class,
        status=manifest.status,
    )
    report = run_fixture_campaign(
        manifest=manifest,
        run_id="slm157-flow-consistency-20260720",
        output_dir=output_dir,
    )
    payload = report.to_dict()
    command = (
        "python -m scripts.run_slm157_flow_consistency_fixture --mode fixture"
    )
    return payload, command


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    status = payload.get("status", "fixture")
    if status == "plan_only":
        manifest = FlowManifest.from_dict(payload["manifest"])
        lines = [
            "# SLM-157 (SPV3-04): Flow / consistency / trajectory-imitation plan",
            "",
            "**Claim class:** wiring / fixture only",
            "",
            "**Run date:** 2026-07-20",
            "",
            "**Machine-readable result:** ["
            "`iter-slm157-flow-consistency-20260720.json`"
            "](iter-slm157-flow-consistency-20260720.json)",
            "",
            "This is a plan-only manifest. The flow/consistency arms, path families, "
            "and validation rules are wired; run `--mode fixture` to execute the "
            "CPU-only simulation.",
            "",
            "## Hypothesis",
            "",
            manifest.hypothesis,
            "",
            "## Falsifier",
            "",
            manifest.falsifier,
            "",
            "## Arms",
            "",
            "| Arm | Family | Path family | Promotable | Diagnostic | Description |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for arm in manifest.arms:
            lines.append(
                f"| {arm.arm_id} | {arm.family.value} | {arm.path_family.value} | "
                f"{arm.promotable} | {arm.diagnostic} | {arm.description} |"
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

    report = FlowConsistencyReport.from_dict(payload)
    return render_markdown(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-157 SPV3-04 flow / consistency / trajectory-imitation fixture",
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
        help="Directory for run artifacts (default: outputs/runs/slm157-flow-consistency-<YYYYMMDD>)",
    )
    parser.add_argument(
        "--n-records",
        type=int,
        default=8,
        help="Number of source/target records per seed in fixture mode.",
    )
    parser.add_argument(
        "--steps",
        type=_parse_int_tuple,
        default="4,8",
        help="Comma-separated step budgets for fixture mode (default: 4,8).",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_int_tuple,
        default="0,1,2",
        help="Comma-separated random seeds for fixture mode (default: 0,1,2).",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    output_dir = (
        args.output_dir
        or Path(f"outputs/runs/slm157-flow-consistency-{_today_yyyymmdd()}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, command = _build_payload(
        args.mode,
        output_dir,
        args.n_records,
        args.steps,
        args.seeds,
    )
    payload["schema"] = "Slm157FlowConsistencyReportV1"
    payload["claim_class"] = "wiring"
    payload["status"] = payload.get("status", "fixture")
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    run_json = output_dir / "slm157_flow_consistency_report.json"
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
