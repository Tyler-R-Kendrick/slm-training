#!/usr/bin/env python3
"""Run the SLM-146 SPV1-03 plan-compiler bridge fixture matrix.

Example:
  python -m scripts.run_slm146_semantic_plan_compiler_fixture --mode fixture
  python -m scripts.run_slm146_semantic_plan_compiler_fixture --mode plan-only
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm146_semantic_plan_compiler import (
    MATRIX_SET,
    MATRIX_VERSION,
    Slm146Arm,
    Slm146Manifest,
    Slm146Report,
    Slm146Row,
    build_manifest,
    render_markdown,
    run_fixture_matrix,
)
from slm_training.versioning import build_version_stamp

__all__ = ["main"]

_DESIGN_JSON = "docs/design/iter-slm146-semantic-plan-compiler-20260720.json"
_DESIGN_MD = "docs/design/iter-slm146-semantic-plan-compiler-20260720.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _build_manifest_report(
    mode: str,
    output_dir: Path,
) -> tuple[dict[str, Any], str]:
    manifest = build_manifest()

    if mode == "plan-only":
        payload: dict[str, Any] = {
            "schema": "Slm146PlanCompilerManifestV1",
            "matrix_set": manifest.matrix_set,
            "matrix_version": manifest.matrix_version,
            "status": "plan_only",
            "claim_class": "wiring",
            "manifest": manifest.to_dict(),
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm146_plan_compiler",
            ),
            "timestamp": _now(),
        }
        command = "python -m scripts.run_slm146_semantic_plan_compiler_fixture --mode plan-only"
        return payload, command

    report = run_fixture_matrix(
        run_id="slm146_fixture",
        output_dir=output_dir,
    )
    payload = report.to_dict()
    command = "python -m scripts.run_slm146_semantic_plan_compiler_fixture --mode fixture"
    return payload, command


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    status = payload.get("status", "fixture")
    if status == "plan_only":
        manifest_data = payload["manifest"]
        manifest = Slm146Manifest(
            matrix_set=manifest_data.get("matrix_set", MATRIX_SET),
            matrix_version=manifest_data.get("matrix_version", MATRIX_VERSION),
            hypothesis=manifest_data.get("hypothesis", ""),
            falsifier=manifest_data.get("falsifier", ""),
            arms=[Slm146Arm(**arm) for arm in manifest_data.get("arms", [])],
            claim_class=manifest_data.get("claim_class", "wiring"),
            status=manifest_data.get("status", "not_run"),
        )
        lines = [
            "# SLM-146 / SPV1-03: Plan-compiler bridge plan",
            "",
            "**Claim class:** wiring / fixture only  ",
            "**Run date:** 2026-07-20  ",
            "**Machine-readable result:** [`iter-slm146-semantic-plan-compiler-20260720.json`](iter-slm146-semantic-plan-compiler-20260720.json)",
            "",
            "This is a plan-only manifest. The fixture corpus and arm definitions "
            "are wired; run `--mode fixture` to execute the CPU matrix.",
            "",
            "## Hypothesis",
            "",
            manifest.hypothesis,
            "",
            "## Falsifier",
            "",
            manifest.falsifier,
            "",
            "## Manifest",
            "",
            "| Arm | Seed | Features | Restrictions | Promotable |",
            "| --- | --- | --- | --- | --- |",
        ]
        for arm in manifest.arms:
            lines.append(
                f"| {arm.arm_id} | {arm.seed_mode} | {arm.feature_mode} | "
                f"{arm.restriction_mode} | {arm.promotable} |"
            )
        lines.extend(["", "## Exact command", "", f"```bash\n{command}\n```", ""])
        return "\n".join(lines)

    manifest_data = payload["manifest"]
    report = Slm146Report(
        matrix_set=payload["matrix_set"],
        matrix_version=payload["matrix_version"],
        run_id=payload["run_id"],
        status=payload["status"],
        manifest=Slm146Manifest(
            matrix_set=manifest_data.get("matrix_set", MATRIX_SET),
            matrix_version=manifest_data.get("matrix_version", MATRIX_VERSION),
            hypothesis=manifest_data.get("hypothesis", ""),
            falsifier=manifest_data.get("falsifier", ""),
            arms=[Slm146Arm(**arm) for arm in manifest_data.get("arms", [])],
            claim_class=manifest_data.get("claim_class", "wiring"),
            status=manifest_data.get("status", "not_run"),
        ),
        rows=[Slm146Row(**row) for row in payload["rows"]],
        version_stamp=payload.get("version_stamp", {}),
        claim_class=payload.get("claim_class", "wiring"),
    )
    return render_markdown(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-146 SPV1-03 plan-compiler bridge fixture matrix",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={"plan-only", "fixture"},
        default="plan-only",
        help="Run mode: plan-only writes the manifest; fixture runs the CPU matrix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for run artifacts (default: outputs/runs/slm146-fixture-<YYYYMMDD>)",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    output_dir = (
        args.output_dir
        or Path(f"outputs/runs/slm146-fixture-{_today_yyyymmdd()}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, command = _build_manifest_report(args.mode, output_dir)
    payload["schema"] = "Slm146PlanCompilerReportV1"
    payload["claim_class"] = "wiring"
    payload["status"] = payload.get("status", "fixture")
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    run_json = output_dir / "slm146_semantic_plan_compiler_report.json"
    run_json.write_text(report_text, encoding="utf-8")

    # Design artifacts are durable results; write them only for the fixture run.
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
    raise SystemExit(main())
