#!/usr/bin/env python3
"""Run the SLM-212 (SDE5-05) constraint-debt routing fixture.

Example:
  python -m scripts.run_slm212_debt_routing_fixture --mode plan-only
  python -m scripts.run_slm212_debt_routing_fixture --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm212_debt_routing import (
    ARM_NAMES,
    MATRIX_SET,
    DebtRoutingArmResult,
    DebtRoutingMatrixManifest,
    render_markdown,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm212-debt-routing-20260721.json"
_DESIGN_MD = "docs/design/iter-slm212-debt-routing-20260721.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _build_payload(
    mode: str,
    output_dir: Path,
    n_examples: int,
    signal_name: str,
    threshold_high: float,
    threshold_low: float | None,
    hysteresis: int,
    budget_mode: str,
    seed: int,
) -> tuple[dict[str, Any], str]:
    if mode == "plan-only":
        arms = tuple(
            DebtRoutingArmResult(
                arm_name=name,
                route_counts={},
                route_by_kind={},
                accuracy=0.0,
                mean_outcome=0.0,
                mean_regret=0.0,
                total_verifier_cost=0.0,
                budget_mode=budget_mode,
            )
            for name in ARM_NAMES
        )
        payload: dict[str, Any] = {
            "schema": "DebtRoutingMatrixManifest",
            "matrix_set": MATRIX_SET,
            "matrix_version": "sde5-05-v1",
            "experiment_id": "slm212-debt-routing",
            "status": "plan_only",
            "claim_class": "wiring",
            "arms": [arm.to_dict() for arm in arms],
            "n_examples": n_examples,
            "signal_name": signal_name,
            "threshold_high": threshold_high,
            "threshold_low": threshold_low,
            "hysteresis": hysteresis,
            "budget_mode": budget_mode,
            "lineage": {"plan_seed": seed},
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm212_debt_routing",
                "harness.model_build.eval",
                "matrix.slm212_debt_routing",
            ),
            "timestamp": _now(),
        }
        command = "python -m scripts.run_slm212_debt_routing_fixture --mode plan-only"
        return payload, command

    manifest = run_fixture(
        output_dir=output_dir,
        n_examples=n_examples,
        signal_name=signal_name,
        threshold_high=threshold_high,
        threshold_low=threshold_low,
        hysteresis=hysteresis,
        budget_mode=budget_mode,
        seed=seed,
    )
    payload = manifest.to_dict()
    command = "python -m scripts.run_slm212_debt_routing_fixture --mode fixture"
    return payload, command


def run_fixture(
    output_dir: Path,
    *,
    n_examples: int,
    signal_name: str,
    threshold_high: float,
    threshold_low: float | None,
    hysteresis: int,
    budget_mode: str,
    seed: int,
) -> DebtRoutingMatrixManifest:
    """Thin wrapper that writes run artifacts and design docs."""
    from slm_training.harnesses.experiments.slm212_debt_routing import run_fixture_matrix

    return run_fixture_matrix(
        output_dir=output_dir,
        n_examples=n_examples,
        signal_name=signal_name,
        threshold_high=threshold_high,
        threshold_low=threshold_low,
        hysteresis=hysteresis,
        budget_mode=budget_mode,
        seed=seed,
        write_design_docs=True,
    )


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    status = payload.get("status", "fixture")
    if status == "plan_only":
        arms = [DebtRoutingArmResult(**a) for a in payload["arms"]]
        lines = [
            "# SLM-212 (SDE5-05): constraint-debt routing plan",
            "",
            "**Claim class:** wiring / fixture only",
            "",
            f"**Run date:** {_today_yyyymmdd()}",
            "",
            "**Machine-readable result:** ["
            "`iter-slm212-debt-routing-20260721.json`"
            "](iter-slm212-debt-routing-20260721.json)",
            "",
            "This is a plan-only manifest. The routing policy, calibrator fallback, "
            "hysteresis, budget accounting, and arm structure are wired; run "
            "`--mode fixture` to execute the CPU-only simulation.",
            "",
            "## Arms",
            "",
            "| arm_name |",
            "| --- |",
        ]
        for arm in arms:
            lines.append(f"| {arm.arm_name} |")
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

    manifest = DebtRoutingMatrixManifest.from_dict(payload)
    return render_markdown(manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-212 SDE5-05 constraint-debt routing fixture",
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
        help="Directory for run artifacts (default: outputs/runs/slm212-debt-routing-<YYYYMMDD>)",
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=200,
        help="Number of synthetic decode states (default: 200).",
    )
    parser.add_argument(
        "--signal-name",
        choices=("D_legal", "D_good_proxy", "legal_mass_deficit", "pre_post_mask_kl"),
        default="D_legal",
        help="Signal used by the debt router (default: D_legal).",
    )
    parser.add_argument(
        "--threshold-high",
        type=float,
        default=2.0,
        help="High-debt threshold that triggers the strict decode path.",
    )
    parser.add_argument(
        "--threshold-low",
        type=float,
        default=None,
        help="Low-debt threshold that reverts to the cheap path (default: high).",
    )
    parser.add_argument(
        "--hysteresis",
        type=int,
        default=1,
        help="Minimum consecutive steps before a route switch.",
    )
    parser.add_argument(
        "--budget-mode",
        choices=("equal_verifier_budget", "equal_forward_budget", "equal_wall_budget"),
        default="equal_verifier_budget",
        help="Budget-accounting mode for matched-matrix reporting.",
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

    output_dir = (
        args.output_dir
        or Path(f"outputs/runs/slm212-debt-routing-{_today_yyyymmdd()}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, command = _build_payload(
        args.mode,
        output_dir,
        args.n_examples,
        args.signal_name,
        args.threshold_high,
        args.threshold_low,
        args.hysteresis,
        args.budget_mode,
        args.seed,
    )
    payload["schema"] = "DebtRoutingMatrixManifest"
    payload["claim_class"] = "wiring"
    payload["status"] = payload.get("status", "fixture")
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    run_json = output_dir / "slm212_debt_routing_report.json"
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
