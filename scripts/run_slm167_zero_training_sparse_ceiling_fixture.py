#!/usr/bin/env python3
"""Run the SLM-167 (SDE1-05) zero-training sparse-action ceiling fixture.

Example:
  python -m scripts.run_slm167_zero_training_sparse_ceiling_fixture --mode plan-only
  python -m scripts.run_slm167_zero_training_sparse_ceiling_fixture --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm167_zero_training_sparse_ceiling import (
    FrozenActionArm,
    FrozenActionReport,
    build_cells,
    render_markdown,
    run_fixture_campaign,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm167-zero-training-sparse-ceiling-20260720.json"
_DESIGN_MD = "docs/design/iter-slm167-zero-training-sparse-ceiling-20260720.md"


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
    k_retrieve: int,
    use_expanded_descriptions: bool,
) -> tuple[dict[str, Any], str]:
    cells = build_cells(
        seeds,
        d_model=d_model,
        k_retrieve=k_retrieve,
        use_expanded_descriptions=use_expanded_descriptions,
    )

    if mode == "plan-only":
        payload: dict[str, Any] = {
            "schema": "Slm167ZeroTrainingSparseCeilingManifestV1",
            "matrix_set": cells[0].arm_name,  # placeholder, overwritten below
            "matrix_version": "sde1-05-v1",
            "experiment_id": "slm167-zero-training-sparse-ceiling",
            "status": "plan_only",
            "claim_class": "wiring",
            "cells": [cell.to_dict() for cell in cells],
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm167_zero_training_sparse_ceiling",
            ),
            "timestamp": _now(),
        }
        command = (
            "python -m scripts.run_slm167_zero_training_sparse_ceiling_fixture --mode plan-only"
        )
        return payload, command

    report = run_fixture_campaign(
        cells=cells,
        run_id=f"slm167-zero-training-sparse-ceiling-{_today_yyyymmdd()}",
        output_dir=output_dir,
        seeds=seeds,
    )
    payload = report.to_dict()
    command = "python -m scripts.run_slm167_zero_training_sparse_ceiling_fixture --mode fixture"
    return payload, command


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    status = payload.get("status", "fixture")
    if status == "plan_only":
        cells = [FrozenActionArm.from_dict(c) for c in payload["cells"]]
        lines = [
            "# SLM-167 (SDE1-05): zero-training sparse-action ceiling plan",
            "",
            "**Claim class:** wiring / fixture only",
            "",
            "**Run date:** 2026-07-20",
            "",
            "**Machine-readable result:** ["
            "`iter-slm167-zero-training-sparse-ceiling-20260720.json`"
            "](iter-slm167-zero-training-sparse-ceiling-20260720.json)",
            "",
            "This is a plan-only manifest. The scoring arms, baselines, metrics, "
            "and analysis plumbing are wired; run `--mode fixture` to execute the "
            "CPU-only simulation.",
            "",
            "## Hypothesis",
            "",
            "A frozen semantic scorer performs materially above random and frequency "
            "baselines on grammar-action ranking and produces some nontrivial "
            "end-to-end programs.",
            "",
            "## Falsifier",
            "",
            "The frozen scorer is statistically indistinguishable from strong "
            "nonsemantic baselines and produces no meaningful programs.",
            "",
            "## Scoring arms",
            "",
            "| arm_id | arm_name | decode_setting | seed |",
            "| --- | --- | --- | --- |",
        ]
        for cell in cells:
            lines.append(
                f"| {cell.arm_id} | {cell.arm_name} | {cell.decode_setting} | {cell.seed} |"
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

    report = FrozenActionReport.from_dict(payload)
    return render_markdown(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-167 SDE1-05 zero-training sparse-action ceiling fixture",
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
        help="Directory for run artifacts (default: outputs/runs/slm167-zero-training-sparse-ceiling-<YYYYMMDD>)",
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
    parser.add_argument(
        "--k-retrieve",
        type=int,
        default=8,
        help="Top-k retrieved candidates for the hybrid arm (default: 8).",
    )
    parser.add_argument(
        "--use-expanded-descriptions",
        action="store_true",
        help="Use the expanded description overrides from SLM-163.",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    output_dir = (
        args.output_dir
        or Path(f"outputs/runs/slm167-zero-training-sparse-ceiling-{_today_yyyymmdd()}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, command = _build_payload(
        args.mode,
        output_dir,
        args.seeds,
        args.d_model,
        args.k_retrieve,
        args.use_expanded_descriptions,
    )
    payload["schema"] = "Slm167ZeroTrainingSparseCeilingReportV1"
    payload["claim_class"] = "wiring"
    payload["status"] = payload.get("status", "fixture")
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    run_json = output_dir / "slm167_zero_training_sparse_ceiling_report.json"
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
