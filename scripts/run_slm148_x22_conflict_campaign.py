#!/usr/bin/env python3
"""Run the SLM-148 SPV1-05 plan-conditioned X22 × conflict-slice fixture campaign.

Example:
  python -m scripts.run_slm148_x22_conflict_campaign --mode plan-only
  python -m scripts.run_slm148_x22_conflict_campaign --mode fixture
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm148_x22_conflict_campaign import (
    MATRIX_SET,
    MATRIX_VERSION,
    X22_CONFLICT_CAMPAIGN_ID,
    Slm148Manifest,
    Slm148RecoveryArm,
    Slm148Report,
    Slm148Row,
    Slm148SeedArm,
    build_manifest,
    render_markdown,
    run_fixture_matrix,
)
from slm_training.versioning import build_version_stamp

__all__ = ["main"]

_DESIGN_JSON = "docs/design/iter-slm148-x22-conflict-campaign-20260720.json"
_DESIGN_MD = "docs/design/iter-slm148-x22-conflict-campaign-20260720.md"


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
            "schema": "Slm148X22ConflictCampaignManifestV1",
            "matrix_set": manifest.matrix_set,
            "matrix_version": manifest.matrix_version,
            "experiment_id": manifest.experiment_id,
            "status": "plan_only",
            "claim_class": "wiring",
            "manifest": manifest.to_dict(),
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm148_x22_conflict_campaign",
            ),
            "timestamp": _now(),
        }
        command = "python -m scripts.run_slm148_x22_conflict_campaign --mode plan-only"
        return payload, command

    report = run_fixture_matrix(
        run_id="slm148_fixture",
        output_dir=output_dir,
    )
    payload = report.to_dict()
    command = "python -m scripts.run_slm148_x22_conflict_campaign --mode fixture"
    return payload, command


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    status = payload.get("status", "fixture")
    if status == "plan_only":
        manifest_data = payload["manifest"]
        manifest = Slm148Manifest(
            matrix_set=manifest_data.get("matrix_set", MATRIX_SET),
            matrix_version=manifest_data.get("matrix_version", MATRIX_VERSION),
            experiment_id=manifest_data.get("experiment_id", X22_CONFLICT_CAMPAIGN_ID),
            hypothesis=manifest_data.get("hypothesis", ""),
            falsifier=manifest_data.get("falsifier", ""),
            seed_arms=tuple(
                Slm148SeedArm(**arm) for arm in manifest_data.get("seed_arms", [])
            ),
            recovery_arms=tuple(
                Slm148RecoveryArm(**arm)
                for arm in manifest_data.get("recovery_arms", [])
            ),
            search_config=manifest_data.get("search_config", {}),
            claim_class=manifest_data.get("claim_class", "wiring"),
            status=manifest_data.get("status", "not_run"),
        )
        lines = [
            "# SLM-148 / SPV1-05: plan-conditioned X22 × conflict-slice campaign plan",
            "",
            "**Claim class:** wiring / fixture only",
            "",
            "**Run date:** 2026-07-20",
            "",
            "**Machine-readable result:** ["
            "`iter-slm148-x22-conflict-campaign-20260720.json`](iter-slm148-x22-conflict-campaign-20260720.json)",
            "",
            "This is a plan-only manifest. The staged factorial, plan compiler, "
            "retrieval index, and conflict-slice repair policies are wired; run "
            "`--mode fixture` to execute the CPU matrix.",
            "",
            "## Hypothesis",
            "",
            manifest.hypothesis,
            "",
            "## Falsifier",
            "",
            manifest.falsifier,
            "",
            "## Seed arms",
            "",
            "| Arm | Strategy | Seeds | Promotable | Description |",
            "| --- | --- | --- | --- | --- |",
        ]
        for arm in manifest.seed_arms:
            strategy_value = (
                arm.strategy.value if isinstance(arm.strategy, Enum) else str(arm.strategy)
            )
            lines.append(
                f"| {arm.arm_id} | {strategy_value} | {','.join(map(str, arm.seeds))} | "
                f"{arm.promotable} | {arm.description} |"
            )
        lines.extend(
            [
                "",
                "## Recovery arms",
                "",
                "| Arm | Policy | Diagnostic | Description |",
                "| --- | --- | --- | --- |",
            ]
        )
        for arm in manifest.recovery_arms:
            lines.append(
                f"| {arm.arm_id} | {arm.policy} | {arm.diagnostic} | {arm.description} |"
            )
        lines.extend(["", "## Exact command", "", f"```bash\n{command}\n```", ""])
        return "\n".join(lines)

    manifest_data = payload["manifest"]
    report = Slm148Report(
        matrix_set=payload["matrix_set"],
        matrix_version=payload["matrix_version"],
        experiment_id=payload["experiment_id"],
        run_id=payload["run_id"],
        status=payload["status"],
        manifest=Slm148Manifest(
            matrix_set=manifest_data.get("matrix_set", MATRIX_SET),
            matrix_version=manifest_data.get("matrix_version", MATRIX_VERSION),
            experiment_id=manifest_data.get("experiment_id", X22_CONFLICT_CAMPAIGN_ID),
            hypothesis=manifest_data.get("hypothesis", ""),
            falsifier=manifest_data.get("falsifier", ""),
            seed_arms=tuple(
                Slm148SeedArm(**arm) for arm in manifest_data.get("seed_arms", [])
            ),
            recovery_arms=tuple(
                Slm148RecoveryArm(**arm)
                for arm in manifest_data.get("recovery_arms", [])
            ),
            search_config=manifest_data.get("search_config", {}),
            claim_class=manifest_data.get("claim_class", "wiring"),
            status=manifest_data.get("status", "not_run"),
        ),
        rows=[Slm148Row(**row) for row in payload["rows"]],
        survivors=payload.get("survivors", []),
        version_stamp=payload.get("version_stamp", {}),
        claim_class=payload.get("claim_class", "wiring"),
    )
    return render_markdown(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-148 SPV1-05 plan-conditioned X22 × conflict-slice campaign",
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
        help="Directory for run artifacts (default: outputs/runs/slm148-fixture-<YYYYMMDD>)",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    output_dir = (
        args.output_dir
        or Path(f"outputs/runs/slm148-fixture-{_today_yyyymmdd()}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, command = _build_manifest_report(args.mode, output_dir)
    payload["schema"] = "Slm148X22ConflictCampaignReportV1"
    payload["claim_class"] = "wiring"
    payload["status"] = payload.get("status", "fixture")
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    run_json = output_dir / "slm148_x22_conflict_campaign_report.json"
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
    raise SystemExit(main())
