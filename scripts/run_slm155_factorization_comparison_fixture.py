#!/usr/bin/env python3
"""Run the SLM-155 (SPV3-02) AR vs X22 factorization comparison fixture.

Example:
  python -m scripts.run_slm155_factorization_comparison_fixture --mode plan-only
  python -m scripts.run_slm155_factorization_comparison_fixture --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm155_factorization_comparison import (
    MATRIX_SET,
    MATRIX_VERSION,
    FACTORIZATION_CAMPAIGN_ID,
    CommonConfig,
    FactorizationArm,
    FactorizationManifest,
    FactorizationReport,
    FactorizationRow,
    build_manifest,
    render_markdown,
    run_fixture_campaign,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm155-factorization-comparison-20260720.json"
_DESIGN_MD = "docs/design/iter-slm155-factorization-comparison-20260720.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _build_payload(
    mode: str,
    output_dir: Path,
    n_records: int,
    scorer_steps: int,
) -> tuple[dict[str, Any], str]:
    manifest = build_manifest()

    if mode == "plan-only":
        payload: dict[str, Any] = {
            "schema": "Slm155FactorizationComparisonManifestV1",
            "matrix_set": manifest.matrix_set,
            "matrix_version": manifest.matrix_version,
            "experiment_id": manifest.experiment_id,
            "status": "plan_only",
            "claim_class": "wiring",
            "manifest": manifest.to_dict(),
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm155_factorization_comparison",
            ),
            "timestamp": _now(),
        }
        command = "python -m scripts.run_slm155_factorization_comparison_fixture --mode plan-only"
        return payload, command

    report = run_fixture_campaign(
        run_id="slm155_fixture",
        output_dir=output_dir,
        n_records=n_records,
        scorer_steps=scorer_steps,
    )
    payload = report.to_dict()
    command = "python -m scripts.run_slm155_factorization_comparison_fixture --mode fixture"
    return payload, command


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    status = payload.get("status", "fixture")
    if status == "plan_only":
        manifest_data = payload["manifest"]
        manifest = FactorizationManifest(
            matrix_set=manifest_data.get("matrix_set", MATRIX_SET),
            matrix_version=manifest_data.get("matrix_version", MATRIX_VERSION),
            experiment_id=manifest_data.get("experiment_id", FACTORIZATION_CAMPAIGN_ID),
            hypothesis=manifest_data.get("hypothesis", ""),
            falsifier=manifest_data.get("falsifier", ""),
            common_config=CommonConfig.from_dict(manifest_data.get("common_config", {})),
            arms=tuple(
                FactorizationArm.from_dict(arm)
                for arm in manifest_data.get("arms", [])
            ),
            claim_class=manifest_data.get("claim_class", "wiring"),
            status=manifest_data.get("status", "not_run"),
        )
        lines = [
            "# SLM-155 (SPV3-02): Matched AR vs X22 factorization campaign plan",
            "",
            "**Claim class:** wiring / fixture only",
            "",
            "**Run date:** 2026-07-20",
            "",
            "**Machine-readable result:** ["
            "`iter-slm155-factorization-comparison-20260720.json`](iter-slm155-factorization-comparison-20260720.json)",
            "",
            "This is a plan-only manifest. The staged factorization arms, common "
            "config, and validation rules are wired; run `--mode fixture` to execute "
            "the CPU matrix.",
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
            "| Arm | Family | Promotable | Diagnostic | Description |",
            "| --- | --- | --- | --- | --- |",
        ]
        for arm in manifest.arms:
            lines.append(
                f"| {arm.arm_id} | {arm.family.value} | {arm.promotable} | "
                f"{arm.diagnostic} | {arm.description} |"
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

    report = FactorizationReport(
        matrix_set=payload.get("matrix_set", MATRIX_SET),
        matrix_version=payload.get("matrix_version", MATRIX_VERSION),
        experiment_id=payload.get("experiment_id", FACTORIZATION_CAMPAIGN_ID),
        run_id=payload.get("run_id", "slm155_fixture"),
        status=payload.get("status", "fixture"),
        manifest=FactorizationManifest(
            matrix_set=payload["manifest"].get("matrix_set", MATRIX_SET),
            matrix_version=payload["manifest"].get("matrix_version", MATRIX_VERSION),
            experiment_id=payload["manifest"].get(
                "experiment_id", FACTORIZATION_CAMPAIGN_ID
            ),
            hypothesis=payload["manifest"].get("hypothesis", ""),
            falsifier=payload["manifest"].get("falsifier", ""),
            common_config=CommonConfig.from_dict(
                payload["manifest"].get("common_config", {})
            ),
            arms=tuple(
                FactorizationArm.from_dict(arm)
                for arm in payload["manifest"].get("arms", [])
            ),
            claim_class=payload["manifest"].get("claim_class", "wiring"),
            status=payload["manifest"].get("status", "not_run"),
        ),
        rows=[FactorizationRow(**row) for row in payload.get("rows", [])],
        version_stamp=payload.get("version_stamp", {}),
        claim_class=payload.get("claim_class", "wiring"),
    )
    return render_markdown(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-155 SPV3-02 AR vs X22 factorization comparison fixture",
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
        help="Directory for run artifacts (default: outputs/runs/slm155-fixture-<YYYYMMDD>)",
    )
    parser.add_argument(
        "--n-records",
        type=int,
        default=16,
        help="Number of records per X22/hybrid arm in fixture mode.",
    )
    parser.add_argument(
        "--scorer-steps",
        type=int,
        default=20,
        help="Training steps for the shared fixture scorer.",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    output_dir = (
        args.output_dir or Path(f"outputs/runs/slm155-fixture-{_today_yyyymmdd()}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, command = _build_payload(
        args.mode, output_dir, args.n_records, args.scorer_steps
    )
    payload["schema"] = "Slm155FactorizationComparisonReportV1"
    payload["claim_class"] = "wiring"
    payload["status"] = payload.get("status", "fixture")
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    run_json = output_dir / "slm155_factorization_comparison_report.json"
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
