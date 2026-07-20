#!/usr/bin/env python3
"""SLM-139 activation-gate closeout.

The issue's activation gate requires SLM-138 to return a positive recursive-core
result (recursive_core_positive, weight_sharing_only with a usable checkpoint, or
another explicit result supporting the shared-recursive base). SLM-138 was merged
as wiring-only evidence on CPU fixtures; no GPU-backed matched-block-evaluation
verdict exists. This script records the gate failure, lists the prerequisite
statuses, and emits a fail-closed closeout report.

Example:
  python -m scripts.run_slm139_gate_closeout --mode plan-only
  python -m scripts.run_slm139_gate_closeout --mode closeout
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.versioning import build_version_stamp


def _today_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _gate_status() -> dict[str, Any]:
    """Return the static gate assessment based on committed evidence.

    Values mirror the issue's activation-gate language.  SLM-138 is treated as
    ``no_recursive_gain`` from the campaign-evaluation standpoint because only
    the wiring/fixture slice was completed; no frontier semantic/recovery
    verdict was produced.
    """
    return {
        "gate_1_recursive_base": {
            "issue": "SLM-138",
            "required_outcome": "recursive_core_positive | weight_sharing_only with usable checkpoint | other explicit supporting result",
            "observed_outcome": "wiring_only fixture; no GPU matched-block evaluation or recursive_core_positive verdict",
            "passed": False,
        },
        "gate_2_multimodal_regime": {
            "issue": "SLM-130",
            "required_outcome": "frozen ambiguity set with >=30 prompts having >=2 hard-valid canonical AST modes",
            "observed_outcome": "merged; fixture evidence exists, but the recursive-base gate already failed",
            "passed": "deferred_to_recursive_base",
        },
        "gate_3_selector": {
            "issue": "SLM-127",
            "required_outcome": "selector selected-pass@K above simple likelihood with calibrated risk/coverage",
            "observed_outcome": "merged; selector available, but the recursive-base gate already failed",
            "passed": "deferred_to_recursive_base",
        },
    }


def _closeout_report() -> dict[str, Any]:
    gates = _gate_status()
    failed = [k for k, v in gates.items() if v.get("passed") is False]
    return {
        "matrix_set": "slm139-stochastic-recursive-width",
        "matrix_version": "slm139-v1",
        "run_id": "slm139_gate_closeout",
        "status": "closeout",
        "claim_class": "wiring",
        "decision": "no_supported_probabilistic_regime",
        "reason": "SLM-138 activation gate not satisfied: only wiring-only fixture evidence exists; no recursive_core_positive or weight_sharing_only verdict from matched-block evaluation.",
        "gates": gates,
        "failed_gates": failed,
        "blocked_arms": ["high_trained", "low_trained", "high_plus_low"],
        "allowed_arms": ["none", "low_inference_only"],
        "note": (
            "No stochastic production code added. If SLM-138 later produces a "
            "positive recursive verdict, this issue can be reopened or superseded."
        ),
        "version_stamp": build_version_stamp("harness.experiments"),
    }


def _plan_only_report() -> dict[str, Any]:
    return {
        "matrix_set": "slm139-stochastic-recursive-width",
        "matrix_version": "slm139-v1",
        "run_id": "slm139_gate_plan",
        "status": "plan_only",
        "claim_class": "wiring",
        "gates": list(_gate_status().keys()),
        "note": "plan-only: gate assessment not executed",
        "version_stamp": build_version_stamp("harness.experiments"),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# SLM-139 / EFS4-03: Stochastic recursive width gate closeout ({report.get('run_id')})",
        "",
        f"Matrix set: `{report.get('matrix_set')}`  ",
        f"Version: `{report.get('matrix_version')}`  ",
        f"Status: **{report.get('status')}**  ",
        f"Decision: **{report.get('decision', 'pending')}**",
        "",
        "## Activation gate assessment",
        "",
        report.get("reason", ""),
        "",
    ]
    gates = report.get("gates", {})
    if gates:
        if isinstance(gates, dict):
            lines.extend(["| Gate | Issue | Required | Observed | Passed |", "| --- | --- | --- | --- | --- |"])
            for key, gate in gates.items():
                lines.append(
                    f"| {key} | {gate.get('issue')} | {gate.get('required_outcome')} | "
                    f"{gate.get('observed_outcome')} | {gate.get('passed')} |"
                )
        else:
            lines.extend(["", "Gates to be assessed:", ""] + [f"- `{g}`" for g in gates])
    blocked = report.get("blocked_arms")
    if blocked:
        lines.extend(["", "## Blocked arms", ""] + [f"- `{a}`" for a in blocked])
    allowed = report.get("allowed_arms")
    if allowed:
        lines.extend(["", "## Allowed control arms", ""] + [f"- `{a}`" for a in allowed])
    note = report.get("note")
    if note:
        lines.extend(["", "## Note", "", note])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SLM-139 activation-gate closeout")
    parser.add_argument(
        "--mode",
        choices=("plan-only", "closeout"),
        default="plan-only",
        help="plan-only emits the skeleton; closeout records the gate failure",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"outputs/runs/slm139-stochastic-recursive-width-{_today_slug()}"),
    )
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    design_json = Path(f"docs/design/iter-slm139-stochastic-recursive-width-{_today_slug()}.json")
    design_md = Path(f"docs/design/iter-slm139-stochastic-recursive-width-{_today_slug()}.md")

    report = _plan_only_report() if args.mode == "plan-only" else _closeout_report()

    report_text = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    report_path = output_dir / "slm139_gate_closeout_report.json"
    report_path.write_text(report_text, encoding="utf-8")
    markdown = _render_markdown(report)
    (output_dir / "slm139_gate_closeout_report.md").write_text(markdown, encoding="utf-8")

    design_json.parent.mkdir(parents=True, exist_ok=True)
    design_json.write_text(report_text, encoding="utf-8")
    design_md.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"\nReport JSON: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
