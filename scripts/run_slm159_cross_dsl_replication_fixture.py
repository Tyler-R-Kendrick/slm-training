#!/usr/bin/env python3
"""Run the SLM-159 (SPV4-01) cross-DSL semantic-plan replication fixture.

Example:
  python -m scripts.run_slm159_cross_dsl_replication_fixture --mode plan-only
  python -m scripts.run_slm159_cross_dsl_replication_fixture --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm159_cross_dsl_replication import (
    CommonConfig,
    CrossDslManifest,
    CrossDslReplicationReport,
    build_manifest,
    render_markdown,
    run_fixture_campaign,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm159-cross-dsl-replication-20260720.json"
_DESIGN_MD = "docs/design/iter-slm159-cross-dsl-replication-20260720.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def _build_payload(
    mode: str,
    output_dir: Path,
    n_graphql_records: int,
    graphql_depth: int,
    seeds: tuple[int, ...],
) -> tuple[dict[str, Any], str]:
    manifest = build_manifest()

    if mode == "plan-only":
        payload: dict[str, Any] = {
            "schema": "Slm159CrossDslReplicationManifestV1",
            "matrix_set": manifest.matrix_set,
            "matrix_version": manifest.matrix_version,
            "experiment_id": manifest.experiment_id,
            "status": "plan_only",
            "claim_class": "wiring",
            "manifest": manifest.to_dict(),
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm159_cross_dsl_replication",
            ),
            "timestamp": _now(),
        }
        command = (
            "python -m scripts.run_slm159_cross_dsl_replication_fixture --mode plan-only"
        )
        return payload, command

    cfg = CommonConfig(
        n_graphql_records=n_graphql_records,
        graphql_depth=graphql_depth,
        seeds=seeds,
    )
    manifest = CrossDslManifest(
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
        run_id="slm159_fixture",
        output_dir=output_dir,
    )
    payload = report.to_dict()
    command = (
        "python -m scripts.run_slm159_cross_dsl_replication_fixture --mode fixture"
    )
    return payload, command


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    status = payload.get("status", "fixture")
    if status == "plan_only":
        manifest = CrossDslManifest.from_dict(payload["manifest"])
        lines = [
            "# SLM-159 (SPV4-01): Cross-DSL semantic-plan replication plan",
            "",
            "**Claim class:** wiring / fixture only",
            "",
            "**Run date:** 2026-07-20",
            "",
            "**Machine-readable result:** ["
            "`iter-slm159-cross-dsl-replication-20260720.json`"
            "](iter-slm159-cross-dsl-replication-20260720.json)",
            "",
            "This is a plan-only manifest. The GraphQL adapter, second-pack readiness "
            "rubric, and blocked-arm disposition are wired; run `--mode fixture` to "
            "exercise the GraphQL pack adapter.",
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
            "| Arm | Pack | Family | Promotable | Blocked | Description |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for arm in manifest.arms:
            lines.append(
                f"| {arm.arm_id} | {arm.pack_id} | {arm.family.value} | "
                f"{arm.promotable} | {arm.blocked} | {arm.description} |"
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

    report = CrossDslReplicationReport.from_dict(payload)
    return render_markdown(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-159 SPV4-01 cross-DSL semantic-plan replication fixture",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={"plan-only", "fixture"},
        default="plan-only",
        help="Run mode: plan-only writes the manifest; fixture runs the pack adapter.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for run artifacts (default: outputs/runs/slm159-fixture-<YYYYMMDD>)",
    )
    parser.add_argument(
        "--n-graphql-records",
        type=int,
        default=8,
        help="Number of GraphQL records to exercise per seed in fixture mode.",
    )
    parser.add_argument(
        "--graphql-depth",
        type=int,
        default=1,
        help="GraphQL corpus generator selection depth.",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default="0,1",
        help="Comma-separated random seeds for fixture mode (default: 0,1).",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    output_dir = (
        args.output_dir or Path(f"outputs/runs/slm159-fixture-{_today_yyyymmdd()}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, command = _build_payload(
        args.mode,
        output_dir,
        args.n_graphql_records,
        args.graphql_depth,
        args.seeds,
    )
    payload["schema"] = "Slm159CrossDslReplicationReportV1"
    payload["claim_class"] = "wiring"
    payload["status"] = payload.get("status", "fixture")
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    run_json = output_dir / "slm159_cross_dsl_replication_report.json"
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
