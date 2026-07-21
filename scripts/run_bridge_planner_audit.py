#!/usr/bin/env python3
"""Run the SLM-189 (FFE2-01) bridge planner audit/fixture.

Example:
  python -m scripts.run_bridge_planner_audit --mode plan-only
  python -m scripts.run_bridge_planner_audit --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm189_bridge_planner import (
    ARM_NAMES,
    MATRIX_SET,
    MATRIX_VERSION,
    BridgePlannerArmSummary,
    BridgePlannerManifest,
    run_bridge_planner_fixture,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm189-bridge-planner-20260721.json"
_DESIGN_MD = "docs/design/iter-slm189-bridge-planner-20260721.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _build_plan_only_payload(
    arms: tuple[str, ...],
    exact_budget: int,
) -> dict[str, Any]:
    arm_summaries = tuple(
        BridgePlannerArmSummary(
            arm_name=name,
            n_cases=0,
            n_reached=0,
            n_unknown_budget=0,
            n_unreachable=0,
            n_certificate_failure=0,
            mean_path_length=0.0,
            p95_path_length=0.0,
            mean_wall_seconds=0.0,
            mean_nodes_expanded=0.0,
            source_bias_index=0.0,
            path_entropy_bits=0.0,
            excess_cost_ratio=0.0,
        )
        for name in arms
    )
    return {
        "schema": "BridgePlannerManifest",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": "slm189-bridge-planner",
        "status": "plan_only",
        "claim_class": "wiring",
        "arms": [arm.to_dict() for arm in arm_summaries],
        "cases": [],
        "n_cases": 0,
        "n_reached": 0,
        "source_policies": ["minimal"],
        "version_stamp": build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm189_bridge_planner",
            "harness.model_build.eval",
            "matrix.slm189_bridge_planner",
            "data.flow.bridge_planner",
        ),
        "timestamp": _now(),
    }


def _build_training_manifest(
    manifest: BridgePlannerManifest,
    output_dir: Path,
) -> Path:
    """Write a tiny training manifest selecting canonical_greedy + minimal source."""
    permitted_arms = tuple(
        arm.arm_name for arm in manifest.arms if arm.arm_name == "canonical_greedy"
    ) or ("canonical_greedy",)
    payload: dict[str, Any] = {
        "schema": "BridgePlannerManifestV1",
        "matrix_set": manifest.matrix_set,
        "matrix_version": manifest.matrix_version,
        "experiment_id": manifest.experiment_id,
        "run_id": manifest.run_id,
        "status": "training_selection",
        "claim_class": "wiring",
        "permitted_arms": list(permitted_arms),
        "source_policy": "minimal",
        "timestamp": _now(),
        "version_stamp": manifest.version_stamp,
    }
    path = output_dir / "slm189_bridge_planner_training_manifest.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return path


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    status = payload.get("status", "fixture")
    if status == "plan_only":
        lines = [
            "# SLM-189 (FFE2-01): bridge planner plan",
            "",
            "**Claim class:** wiring / fixture only",
            "",
            f"**Run date:** {_today_yyyymmdd()}",
            "",
            "**Machine-readable result:** ["
            "`iter-slm189-bridge-planner-20260721.json`"
            "](iter-slm189-bridge-planner-20260721.json)",
            "",
            "This is a plan-only manifest. The bridge planner arms, dependency DAG, "
            "transition certificates, and source policies are wired; run "
            "`--mode fixture` to execute the CPU-only simulation.",
            "",
            "## Arms",
            "",
            "| arm_name |",
            "| --- |",
        ]
        for arm in payload.get("arms", []):
            lines.append(f"| {arm['arm_name']} |")
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

    manifest = BridgePlannerManifest.from_dict(payload)
    from slm_training.harnesses.experiments.slm189_bridge_planner import render_markdown

    return render_markdown(manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-189 FFE2-01 bridge planner audit/fixture",
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
        help="Directory for run artifacts (default: outputs/runs/slm189-bridge-planner-<YYYYMMDD>)",
    )
    parser.add_argument(
        "--arms",
        type=str,
        default=",".join(ARM_NAMES),
        help="Comma-separated list of arms to run.",
    )
    parser.add_argument(
        "--exact-budget",
        type=int,
        default=8,
        help="Maximum edit count for the exact-shortest arm (default: 8).",
    )
    parser.add_argument(
        "--scale-grid",
        action="store_true",
        help="Include synthetic scale targets in fixture mode.",
    )
    parser.add_argument(
        "--compare-sources",
        action="store_true",
        help="Run all source policies (minimal, template, gold, retrieved).",
    )
    parser.add_argument(
        "--emit-training-manifest",
        action="store_true",
        help="Write a tiny training manifest selecting canonical_greedy + minimal.",
    )
    parser.add_argument(
        "--design-json",
        type=Path,
        default=None,
        help="Override path for the design JSON.",
    )
    parser.add_argument(
        "--design-md",
        type=Path,
        default=None,
        help="Override path for the design markdown.",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    requested_arms = tuple(a.strip() for a in args.arms.split(",") if a.strip())
    invalid = set(requested_arms) - set(ARM_NAMES)
    if invalid:
        print(f"unknown arms: {sorted(invalid)}", file=sys.stderr)
        return 2

    output_dir = (
        args.output_dir
        or Path(f"outputs/runs/slm189-bridge-planner-{_today_yyyymmdd()}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_policies: tuple[str, ...]
    if args.compare_sources:
        source_policies = ("minimal", "template", "gold", "retrieved")
    else:
        source_policies = ("minimal",)

    if args.mode == "plan-only":
        payload = _build_plan_only_payload(requested_arms, args.exact_budget)
        command = "python -m scripts.run_bridge_planner_audit --mode plan-only"
    else:
        manifest = run_bridge_planner_fixture(
            output_dir=output_dir,
            arms=requested_arms,
            source_policies=source_policies,
            exact_budget=args.exact_budget,
            include_scale_grid=args.scale_grid,
            write_design_docs=True,
            design_json=args.design_json,
            design_md=args.design_md,
        )
        payload = manifest.to_dict()
        command = "python -m scripts.run_bridge_planner_audit --mode fixture"

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    run_json = output_dir / "slm189_bridge_planner_report.json"
    run_json.write_text(report_text, encoding="utf-8")

    if args.mode == "fixture":
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

        if args.emit_training_manifest:
            manifest_obj = BridgePlannerManifest.from_dict(payload)
            _build_training_manifest(manifest_obj, output_dir)

    print(str(run_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
