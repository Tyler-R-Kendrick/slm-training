#!/usr/bin/env python3
"""Run the SLM-165 (SDE1-03) 2×2×2 interaction factorial fixture.

Example:
  python -m scripts.run_slm165_interaction_factorial_fixture --mode plan-only
  python -m scripts.run_slm165_interaction_factorial_fixture --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm165_interaction_factorial import (
    InteractionCell,
    InteractionFactorialReport,
    build_cells,
    render_markdown,
    run_fixture_campaign,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm165-interaction-factorial-20260720.json"
_DESIGN_MD = "docs/design/iter-slm165-interaction-factorial-20260720.md"


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
    strict_winners: bool,
) -> tuple[dict[str, Any], str]:
    cells = build_cells(seeds)

    if mode == "plan-only":
        payload: dict[str, Any] = {
            "schema": "Slm165InteractionFactorialManifestV1",
            "matrix_set": cells[0].action_init,  # placeholder, overwritten below
            "matrix_version": "sde1-03-v1",
            "experiment_id": "slm165-interaction-factorial",
            "status": "plan_only",
            "claim_class": "wiring",
            "cells": [cell.to_dict() for cell in cells],
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm165_interaction_factorial",
            ),
            "timestamp": _now(),
        }
        command = (
            "python -m scripts.run_slm165_interaction_factorial_fixture --mode plan-only"
        )
        return payload, command

    report = run_fixture_campaign(
        cells=cells,
        run_id=f"slm165-interaction-factorial-{_today_yyyymmdd()}",
        output_dir=output_dir,
        seeds=seeds,
        strict_winners=strict_winners,
    )
    payload = report.to_dict()
    command = "python -m scripts.run_slm165_interaction_factorial_fixture --mode fixture"
    return payload, command


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    status = payload.get("status", "fixture")
    if status == "plan_only":
        cells = [InteractionCell.from_dict(c) for c in payload["cells"]]
        lines = [
            "# SLM-165 (SDE1-03): 2×2×2 interaction factorial plan",
            "",
            "**Claim class:** wiring / fixture only",
            "",
            "**Run date:** 2026-07-20",
            "",
            "**Machine-readable result:** ["
            "`iter-slm165-interaction-factorial-20260720.json`"
            "](iter-slm165-interaction-factorial-20260720.json)",
            "",
            "This is a plan-only manifest. The factorial cells, metrics, "
            "and analysis plumbing are wired; run `--mode fixture` to execute the "
            "CPU-only simulation.",
            "",
            "## Hypothesis",
            "",
            "The combination of schema-description action initialization (SLM-163), "
            "E396 type balancing, and targeted legal-sibling margin (SLM-164) produces "
            "a positive three-way interaction on rare-component recall, exceeding any "
            "two-lever combination.",
            "",
            "## Falsifier",
            "",
            "The full three-lever cell does not outperform the best two-lever cell by "
            "at least 0.05 rare-component recall, or the three-way interaction is "
            "non-positive.",
            "",
            "## Factorial cells",
            "",
            "| Cell | action_init | type_balance | sibling_margin | seed |",
            "| --- | --- | --- | --- | --- |",
        ]
        for cell in cells:
            lines.append(
                f"| {cell.cell_id} | {cell.action_init} | {cell.type_balance} | "
                f"{cell.sibling_margin} | {cell.seed} |"
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

    report = InteractionFactorialReport.from_dict(payload)
    return render_markdown(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-165 SDE1-03 2×2×2 interaction factorial fixture",
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
        help="Directory for run artifacts (default: outputs/runs/slm165-interaction-factorial-<YYYYMMDD>)",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_int_tuple,
        default="0,1,2",
        help="Comma-separated random seeds for fixture mode (default: 0,1,2).",
    )
    parser.add_argument(
        "--strict-winners",
        action="store_true",
        help="Fail closed if predecessor result JSONs do not declare the expected winners.",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    output_dir = (
        args.output_dir
        or Path(f"outputs/runs/slm165-interaction-factorial-{_today_yyyymmdd()}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, command = _build_payload(
        args.mode,
        output_dir,
        args.seeds,
        args.strict_winners,
    )
    payload["schema"] = "Slm165InteractionFactorialReportV1"
    payload["claim_class"] = "wiring"
    payload["status"] = payload.get("status", "fixture")
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    run_json = output_dir / "slm165_interaction_factorial_report.json"
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
