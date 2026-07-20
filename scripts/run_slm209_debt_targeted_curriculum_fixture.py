#!/usr/bin/env python3
"""Run the SLM-209 (SDE5-02) debt-targeted curriculum fixture.

Example:
  python -m scripts.run_slm209_debt_targeted_curriculum_fixture --mode plan-only
  python -m scripts.run_slm209_debt_targeted_curriculum_fixture --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm209_debt_targeted_curriculum import (
    MATRIX_SET,
    DebtCurriculumCellV1,
    DebtCurriculumManifestV1,
    build_cells,
    render_markdown,
    run_fixture_campaign,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm209-debt-targeted-curriculum-20260720.json"
_DESIGN_MD = "docs/design/iter-slm209-debt-targeted-curriculum-20260720.md"


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
    n_states: int,
    total_decision_budget: int,
    per_group_cap: int,
    seed: int,
) -> tuple[dict[str, Any], str]:
    cells = build_cells(
        seeds,
        total_decision_budget=total_decision_budget,
        per_group_cap=per_group_cap,
    )

    if mode == "plan-only":
        payload: dict[str, Any] = {
            "schema": "DebtCurriculumManifestV1",
            "matrix_set": MATRIX_SET,
            "matrix_version": "sde5-02-v1",
            "experiment_id": "slm209-debt-targeted-curriculum",
            "status": "plan_only",
            "claim_class": "wiring",
            "cells": [cell.to_dict() for cell in cells],
            "total_decision_budget": total_decision_budget,
            "per_group_cap": per_group_cap,
            "lineage": {
                "synthetic_state_count": n_states,
                "plan_seed": seed,
            },
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm209_debt_targeted_curriculum",
                "harness.preference.constraint_debt",
                "harness.train_data",
            ),
            "timestamp": _now(),
        }
        command = (
            "python -m scripts.run_slm209_debt_targeted_curriculum_fixture --mode plan-only"
        )
        return payload, command

    manifest = run_fixture_campaign(
        output_dir=output_dir,
        seeds=seeds,
        n_states=n_states,
        total_decision_budget=total_decision_budget,
        per_group_cap=per_group_cap,
        seed=seed,
        write_design_docs=True,
    )
    payload = manifest.to_dict()
    command = "python -m scripts.run_slm209_debt_targeted_curriculum_fixture --mode fixture"
    return payload, command


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    status = payload.get("status", "fixture")
    if status == "plan_only":
        cells = [DebtCurriculumCellV1.from_dict(c) for c in payload["cells"]]
        lines = [
            "# SLM-209 (SDE5-02): debt-targeted semantic exposure curriculum plan",
            "",
            "**Claim class:** wiring / fixture only",
            "",
            "**Run date:** 2026-07-20",
            "",
            "**Machine-readable result:** ["
            "`iter-slm209-debt-targeted-curriculum-20260720.json`"
            "](iter-slm209-debt-targeted-curriculum-20260720.json)",
            "",
            "This is a plan-only manifest. The selection policies, score components, "
            "caps, and audit plumbing are wired; run `--mode fixture` to execute the "
            "CPU-only simulation.",
            "",
            "## Hypothesis",
            "",
            "A fixed-budget curriculum that selects exact states by grammar-mask debt, "
            "inverse decision-kind frequency, and legal-support entropy increases the "
            "exposure of high-debt exact states while respecting per-group caps and "
            "train/held-out isolation.",
            "",
            "## Falsifier",
            "",
            "Debt-targeted selection fails to increase high-debt state exposure, violates "
            "the total decision budget, exceeds per-group caps, or leaks a group across "
            "train/held-out splits.",
            "",
            "## Cells",
            "",
            "| policy_name | seed | decision_budget | per_group_cap |",
            "| --- | --- | --- | --- |",
        ]
        for cell in cells:
            lines.append(
                f"| {cell.policy_name} | {cell.seed} | "
                f"{cell.decision_budget} | {cell.per_group_cap} |"
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

    manifest = DebtCurriculumManifestV1.from_dict(payload)
    return render_markdown(manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-209 SDE5-02 debt-targeted semantic exposure curriculum fixture",
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
        help="Directory for run artifacts (default: outputs/runs/slm209-debt-targeted-curriculum-<YYYYMMDD>)",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_int_tuple,
        default="0,1,2",
        help="Comma-separated random seeds for fixture mode (default: 0,1,2).",
    )
    parser.add_argument(
        "--n-states",
        type=int,
        default=200,
        help="Number of synthetic states to generate (default: 200).",
    )
    parser.add_argument(
        "--total-decision-budget",
        type=int,
        default=120,
        help="Total decision budget for each policy cell (default: 120).",
    )
    parser.add_argument(
        "--per-group-cap",
        type=int,
        default=6,
        help="Max selected states per group for each policy cell (default: 6).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for synthetic state generation (default: 0).",
    )
    parser.add_argument(
        "--write-design-docs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write design docs in fixture mode (default: True).",
    )
    parser.add_argument(
        "--design-json",
        type=Path,
        default=None,
        help="Override path for the design JSON (default: docs/design/iter-slm209-debt-targeted-curriculum-<YYYYMMDD>.json).",
    )
    parser.add_argument(
        "--design-md",
        type=Path,
        default=None,
        help="Override path for the design markdown (default: docs/design/iter-slm209-debt-targeted-curriculum-<YYYYMMDD>.md).",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    output_dir = (
        args.output_dir
        or Path(
            f"outputs/runs/slm209-debt-targeted-curriculum-{_today_yyyymmdd()}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, command = _build_payload(
        args.mode,
        output_dir,
        args.seeds,
        args.n_states,
        args.total_decision_budget,
        args.per_group_cap,
        args.seed,
    )
    payload["schema"] = "DebtCurriculumManifestV1"
    payload["claim_class"] = "wiring"
    payload["status"] = payload.get("status", "fixture")
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    run_json = output_dir / "slm209_debt_targeted_curriculum_report.json"
    run_json.write_text(report_text, encoding="utf-8")

    if args.mode == "fixture" and args.write_design_docs:
        root = Path(__file__).resolve().parents[1]
        json_path = args.design_json or root / _DESIGN_JSON
        md_path = args.design_md or root / _DESIGN_MD
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
