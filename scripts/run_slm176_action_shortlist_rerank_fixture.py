#!/usr/bin/env python3
"""Run the SLM-176 (P14) action-shortlist retrieve-then-rerank fixture.

Example:
  python -m scripts.run_slm176_action_shortlist_rerank_fixture --mode plan-only
  python -m scripts.run_slm176_action_shortlist_rerank_fixture --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm176_action_shortlist_rerank import (
    MATRIX_SET,
    ShortlistReport,
    ShortlistScenario,
    build_cells,
    render_markdown,
    run_fixture_campaign,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm176-action-shortlist-rerank-20260720.json"
_DESIGN_MD = "docs/design/iter-slm176-action-shortlist-rerank-20260720.md"


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
    cells = build_cells(seeds)

    if mode == "plan-only":
        payload: dict[str, Any] = {
            "schema": "Slm176ActionShortlistRerankManifestV1",
            "matrix_set": MATRIX_SET,
            "matrix_version": "p14-v1",
            "experiment_id": "slm176-action-shortlist-rerank",
            "status": "plan_only",
            "claim_class": "wiring",
            "d_model": d_model,
            "cells": [cell.to_dict() for cell in cells],
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm176_action_shortlist_rerank",
                "dsl.action_shortlist",
                "dsl.action_descriptions",
            ),
            "timestamp": _now(),
        }
        command = "python -m scripts.run_slm176_action_shortlist_rerank_fixture --mode plan-only"
        return payload, command

    report = run_fixture_campaign(
        cells=cells,
        run_id=f"slm176-action-shortlist-rerank-{_today_yyyymmdd()}",
        output_dir=output_dir,
        seeds=seeds,
        d_model=d_model,
    )
    payload = report.to_dict()
    command = (
        "python -m scripts.run_slm176_action_shortlist_rerank_fixture --mode fixture"
    )
    return payload, command


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    status = payload.get("status", "fixture")
    if status == "plan_only":
        cells = [ShortlistScenario.from_dict(c) for c in payload["cells"]]
        lines = [
            "# SLM-176 (P14): action-shortlist retrieve-then-rerank fixture plan",
            "",
            "**Claim class:** wiring / fixture only",
            "",
            "**Run date:** 2026-07-20",
            "",
            "**Machine-readable result:** ["
            "`iter-slm176-action-shortlist-rerank-20260720.json`"
            "](iter-slm176-action-shortlist-rerank-20260720.json)",
            "",
            "This is a plan-only manifest. The description-retrieval shortlist, "
            "rerank plumbing, and recall metrics are wired; run "
            "`--mode fixture` to execute the CPU-only simulation.",
            "",
            "## Hypothesis",
            "",
            "A deterministic description-retrieval shortlist preserves the "
            "full-set top candidate for synthetic legal action sets.",
            "",
            "## Falsifier",
            "",
            "The retrieval shortlist drops the full-set top-1 candidate or "
            "collapses to fallback for every non-trivial legal set.",
            "",
            "## Scenarios",
            "",
            "| scenario_id | legal_set_size | k | seed | query_hint |",
            "| --- | --- | --- | --- | --- |",
        ]
        for cell in cells:
            lines.append(
                f"| {cell.scenario_id} | {cell.legal_set_size} | {cell.k} | "
                f"{cell.seed} | {cell.query_hint} |"
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

    report = ShortlistReport.from_dict(payload)
    return render_markdown(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-176 P14 action-shortlist retrieve-then-rerank fixture",
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
        help="Directory for run artifacts (default: outputs/runs/slm176-action-shortlist-rerank-<YYYYMMDD>)",
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
        help="Dimension of the deterministic fixture description encoder.",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    output_dir = args.output_dir or Path(
        f"outputs/runs/slm176-action-shortlist-rerank-{_today_yyyymmdd()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, command = _build_payload(
        args.mode,
        output_dir,
        args.seeds,
        args.d_model,
    )
    payload["schema"] = "Slm176ActionShortlistRerankReportV1"
    payload["claim_class"] = "wiring"
    payload["status"] = payload.get("status", "fixture")
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    run_json = output_dir / "slm176_action_shortlist_rerank_report.json"
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
