#!/usr/bin/env python3
"""Run the SLM-192 (FFE3-01) stage-accurate flow-pipeline cost-profile fixture.

Example:
  python -m scripts.profile_flow_pipeline --describe
  python -m scripts.profile_flow_pipeline --fixture
  python -m scripts.profile_flow_pipeline --fixture --n-repeats 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm192_profile_flow_pipeline import (
    ARM_NAMES,
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    FlowPipelineManifestV1,
    run_profile_flow_pipeline,
)
from slm_training.versioning import build_version_stamp


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


_DESIGN_JSON = f"docs/design/iter-slm192-profile-flow-pipeline-{_today_yyyymmdd()}.json"
_DESIGN_MD = f"docs/design/iter-slm192-profile-flow-pipeline-{_today_yyyymmdd()}.md"


def _describe_schema() -> str:
    return f"""\
SLM-192 stage-accurate flow-pipeline cost-profile fixture schema

Matrix set: {MATRIX_SET}
Matrix version: {MATRIX_VERSION}
Experiment ID: {EXPERIMENT_ID}

Reference arms:
{chr(10).join(f'  - {name}' for name in ARM_NAMES)}

FlowPipelineManifestV1 fields:
  schema, matrix_set, matrix_version, experiment_id, run_id, status,
  claim_class, hypothesis, falsifier, disposition, disposition_rationale,
  arms (FlowCostProfileV1), cases (FlowPipelineProfileCase),
  cost_gate (CostGateManifestV1), on_policy (OnPolicyFeasibilityV1),
  n_cases, n_arms, honest_caveats, version_stamp, timestamp.

FlowCostProfileV1 fields:
  arm_name, condition, total_ms, span_records (CostSpanRecord), work_units, n_repeats.

Claim class: wiring / fixture only.  No model, GPU, or checkpoint involvement.
"""


_HYPOTHESIS = (
    "Valid-edit bridge/training/decode/closure/verification stages have separable, "
    "measurable CPU cost profiles; the dominant bottleneck on toy fixtures is either "
    "candidate enumeration, exact closure, or verifier replay; and the combined "
    "per-target bridge+enumeration cost extrapolates to an on-policy epoch budget."
)

_FALSIFIER = (
    "The cold and warm profiles are identical (no caching benefit), or the top "
    "bottleneck is not enumeration/closure/verification, or the per-target cost "
    "extrapolates beyond the 30-minute on-policy epoch bound despite the tiny "
    "fixture domain."
)


def _build_payload(
    mode: str,
    output_dir: Path,
    argv_flags: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if mode == "plan-only":
        payload: dict[str, Any] = {
            "schema": "FlowPipelineManifestV1",
            "matrix_set": MATRIX_SET,
            "matrix_version": MATRIX_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "run_id": f"{EXPERIMENT_ID}-{_today_yyyymmdd()}",
            "status": "plan_only",
            "claim_class": "wiring",
            "hypothesis": _HYPOTHESIS,
            "falsifier": _FALSIFIER,
            "disposition": "cost_profile_wired",
            "disposition_rationale": "Plan-only manifest; run --fixture to execute.",
            "arms": [],
            "cases": [],
            "cost_gate": {
                "max_on_policy_epoch_seconds": 1800.0,
                "allowed_strategy": "unknown",
                "enumeration_bound": False,
                "bottlenecks": [],
            },
            "on_policy": {
                "strategy": "unknown",
                "projected_seconds_per_target": 0.0,
                "projected_seconds_for_108_targets": 0.0,
                "extrapolated_dagger_round_seconds": 0.0,
                "extrapolated_five_seeds_seconds": 0.0,
                "extrapolated_confirmation_suite_seconds": 0.0,
                "rationale": "Plan-only; no warm measurements available.",
            },
            "n_cases": 0,
            "n_arms": len(ARM_NAMES),
            "honest_caveats": [
                "Plan-only: no cost spans were measured.",
                "Real on-policy timing requires a trained model and decode path.",
            ],
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm192_profile_flow_pipeline",
            ),
            "timestamp": _now(),
        }
        command = "python -m scripts.profile_flow_pipeline --plan-only"
        return payload, command

    manifest = run_profile_flow_pipeline(
        output_dir=output_dir,
        n_repeats=argv_flags.get("n_repeats", 3),
        seed=argv_flags.get("seed", 0),
        write_design_docs=argv_flags.get("write_design_docs", True),
    )
    payload = manifest.to_dict()
    command = "python -m scripts.profile_flow_pipeline --fixture"
    return payload, command


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    status = payload.get("status", "fixture")
    if status == "plan_only":
        lines = [
            "# SLM-192 (FFE3-01): flow-pipeline cost-profile plan",
            "",
            "**Claim class:** wiring / fixture only",
            "",
            f"**Run date:** {_today_yyyymmdd()}",
            "",
            f"**Machine-readable result:** [`iter-slm192-profile-flow-pipeline-{_today_yyyymmdd()}.json`](iter-slm192-profile-flow-pipeline-{_today_yyyymmdd()}.json)",
            "",
            "This is a plan-only manifest. The cost-profile arms, telemetry spans, and "
            "on-policy extrapolation are wired; run `--fixture` to execute the CPU-only "
            "fixture matrix.",
            "",
            "## Hypothesis",
            "",
            _HYPOTHESIS,
            "",
            "## Falsifier",
            "",
            _FALSIFIER,
            "",
            "## Exact command",
            "",
            f"```bash\n{command}\n```",
            "",
        ]
        return "\n".join(lines)

    manifest = FlowPipelineManifestV1.from_dict(payload)
    from slm_training.harnesses.experiments.slm192_profile_flow_pipeline import render_markdown

    return render_markdown(manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-192 FFE3-01 flow-pipeline cost-profile fixture",
        exit_on_error=False,
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print the schema and arms, then exit.",
    )
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="Run the CPU-only cost-profile fixture.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write a plan-only manifest without profiling.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for run artifacts (default: outputs/runs/slm192-profile-flow-pipeline-<YYYYMMDD>)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Deterministic seed (default: 0).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="Repeats per arm; first is cold, remainder are warm (default: 3).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional model checkpoint path (fixture mode ignores this; reserved for future model runs).",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Acknowledge fixture-only wiring evidence before running.",
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

    if args.describe:
        print(_describe_schema())
        return 0

    mode = "plan-only" if args.plan_only else "fixture"
    if args.confirm and mode == "fixture":
        print("Confirmed: fixture-only wiring evidence, no model/GPU/ship claim.")

    output_dir = args.output_dir or Path(f"outputs/runs/slm192-profile-flow-pipeline-{_today_yyyymmdd()}")
    output_dir.mkdir(parents=True, exist_ok=True)

    flags = {
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "write_design_docs": args.write_design_docs,
    }
    payload, command = _build_payload(mode, output_dir, flags)
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    run_json = output_dir / "slm192_profile_flow_pipeline_report.json"
    run_json.write_text(report_text, encoding="utf-8")

    if mode == "fixture" and args.write_design_docs:
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
