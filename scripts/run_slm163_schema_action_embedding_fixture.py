#!/usr/bin/env python3
"""Run the SLM-163 (SDE1-01) schema-description action-embedding fixture.

Example:
  python -m scripts.run_slm163_schema_action_embedding_fixture --mode plan-only
  python -m scripts.run_slm163_schema_action_embedding_fixture --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm163_schema_action_embedding import (
    InitArm,
    SchemaActionEmbeddingReport,
    build_manifest,
    render_markdown,
    run_fixture_campaign,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm163-schema-action-embedding-20260720.json"
_DESIGN_MD = "docs/design/iter-slm163-schema-action-embedding-20260720.md"


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
    d_model: int,
) -> tuple[dict[str, Any], str]:
    arms = build_manifest()

    if mode == "plan-only":
        payload: dict[str, Any] = {
            "schema": "Slm163SchemaActionEmbeddingManifestV1",
            "matrix_set": arms[0].source,  # placeholder, overwritten below
            "matrix_version": "sde1-01-v1",
            "experiment_id": "slm163-schema-action-embedding",
            "status": "plan_only",
            "claim_class": "wiring",
            "d_model": d_model,
            "arms": [arm.to_dict() for arm in arms],
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm163_schema_action_embedding",
            ),
            "timestamp": _now(),
        }
        command = (
            "python -m scripts.run_slm163_schema_action_embedding_fixture --mode plan-only"
        )
        return payload, command

    report = run_fixture_campaign(
        arms=arms,
        run_id=f"slm163-schema-action-embedding-{_today_yyyymmdd()}",
        output_dir=output_dir,
        seeds=seeds,
        d_model=d_model,
    )
    payload = report.to_dict()
    command = "python -m scripts.run_slm163_schema_action_embedding_fixture --mode fixture"
    return payload, command


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    status = payload.get("status", "fixture")
    if status == "plan_only":
        arms = [InitArm.from_dict(a) for a in payload["arms"]]
        lines = [
            "# SLM-163 (SDE1-01): Schema-description action-embedding plan",
            "",
            "**Claim class:** wiring / fixture only",
            "",
            "**Run date:** 2026-07-20",
            "",
            "**Machine-readable result:** ["
            "`iter-slm163-schema-action-embedding-20260720.json`"
            "](iter-slm163-schema-action-embedding-20260720.json)",
            "",
            "This is a plan-only manifest. The schema-description arms, metrics, "
            "and validation rules are wired; run `--mode fixture` to execute the "
            "CPU-only simulation.",
            "",
            "## Hypothesis",
            "",
            "Schema-derived action descriptions produce action embeddings that are more "
            "structured than random or stub initializations, as measured by coverage, "
            "nearest-neighbor cosine separation, sibling-family separation, and "
            "rare-vs-common centroid distance.",
            "",
            "## Falsifier",
            "",
            "Schema descriptions do not improve any of the above metrics over the "
            "current_stub baseline, or the shuffled control arm performs as well as "
            "the schema-driven arms.",
            "",
            "## Arms",
            "",
            "| Arm | Source | Promotable | Description |",
            "| --- | --- | --- | --- |",
        ]
        for arm in arms:
            lines.append(
                f"| {arm.arm_id} | {arm.source} | {arm.promotable} | {arm.description} |"
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

    report = SchemaActionEmbeddingReport.from_dict(payload)
    return render_markdown(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-163 SDE1-01 schema-description action-embedding fixture",
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
        help="Directory for run artifacts (default: outputs/runs/slm163-schema-action-embedding-<YYYYMMDD>)",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_int_tuple,
        default="0,1,2",
        help="Comma-separated random seeds for fixture mode (default: 0,1,2).",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=64,
        help="Embedding dimension for the fixture encoder (default: 64).",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    output_dir = (
        args.output_dir
        or Path(f"outputs/runs/slm163-schema-action-embedding-{_today_yyyymmdd()}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, command = _build_payload(
        args.mode,
        output_dir,
        args.seeds,
        args.d_model,
    )
    payload["schema"] = "Slm163SchemaActionEmbeddingReportV1"
    payload["claim_class"] = "wiring"
    payload["status"] = payload.get("status", "fixture")
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    run_json = output_dir / "slm163_schema_action_embedding_report.json"
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
