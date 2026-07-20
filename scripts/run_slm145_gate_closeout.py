#!/usr/bin/env python3
"""SLM-145 authorization-gate closeout.

The issue's authorization gate requires SPV0-02 (SLM-142) gold substitution to
show a material downstream ceiling for topology, cardinality, and/or bindings.
SLM-142 wired extraction, canonicalization, oracle substitution, and seed
construction, but did not run factor-wise oracle-substitution experiments on
real completions or models.  Without that evidence, no factor is justified and
no learned topology/cardinality/pointer head is implemented.

Example:
  python -m scripts.run_slm145_gate_closeout --mode plan-only
  python -m scripts.run_slm145_gate_closeout --mode closeout
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


def _factor_status() -> dict[str, Any]:
    return {
        "archetype": {
            "spv0_02_evidence": "SLM-144 fixture defines gold_archetype arm; toy corpus only",
            "ceiling_observed": False,
            "note": "covered by SLM-144 SPV1-01; not a new SLM-145 target",
        },
        "role_set": {
            "spv0_02_evidence": "SLM-144 fixture defines gold_role_set arm; plan_only status",
            "ceiling_observed": False,
            "note": "covered by SLM-144 SPV1-01; not a new SLM-145 target",
        },
        "topology": {
            "spv0_02_evidence": "no oracle-substitution experiment found",
            "ceiling_observed": False,
            "note": "PlanOracleSubstitutor supports topology factor, but no downstream ceiling measured",
        },
        "cardinality": {
            "spv0_02_evidence": "RoleSlot.min/max_cardinality not populated by extractor; no oracle arm",
            "ceiling_observed": False,
            "note": "schema field exists but extraction and oracle substitution are incomplete",
        },
        "bindings_pointers": {
            "spv0_02_evidence": "binding extraction exists; no oracle-substitution experiment found",
            "ceiling_observed": False,
            "note": "PlanOracleSubstitutor supports bindings factor, but no downstream ceiling measured",
        },
    }


def _closeout_report() -> dict[str, Any]:
    factors = _factor_status()
    justified = [k for k, v in factors.items() if v["ceiling_observed"]]
    not_justified = [k for k, v in factors.items() if not v["ceiling_observed"]]
    return {
        "matrix_set": "slm145-plan-predictor-factors",
        "matrix_version": "slm145-v1",
        "run_id": "slm145_gate_closeout",
        "status": "closeout",
        "claim_class": "wiring",
        "decision": "blocked_pending_spv0_02_ceiling_evidence",
        "reason": (
            "SLM-145 authorization gate not satisfied: SPV0-02/SLM-142 did not "
            "run factor-wise gold-substitution experiments measuring downstream "
            "semantic ceilings for topology, cardinality, or bindings/pointers. "
            "No learned head is justified."
        ),
        "factors": factors,
        "justified_factors": justified,
        "not_justified_factors": not_justified,
        "blocked_heads": [
            "topology_head",
            "cardinality_head",
            "live_symbol_pointer_head",
        ],
        "recommended_next_step": (
            "Run a factor-wise oracle-substitution matrix on a real or fixture "
            "completion corpus and re-open SLM-145 only for factors that show a "
            "preregistered downstream gain."
        ),
        "version_stamp": build_version_stamp("harness.experiments"),
    }


def _plan_only_report() -> dict[str, Any]:
    return {
        "matrix_set": "slm145-plan-predictor-factors",
        "matrix_version": "slm145-v1",
        "run_id": "slm145_gate_plan",
        "status": "plan_only",
        "claim_class": "wiring",
        "factors": list(_factor_status().keys()),
        "note": "plan-only: gate assessment not executed",
        "version_stamp": build_version_stamp("harness.experiments"),
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# SLM-145 / SPV1-02: Plan-predictor factor gate closeout ({report.get('run_id')})",
        "",
        f"Matrix set: `{report.get('matrix_set')}`  ",
        f"Version: `{report.get('matrix_version')}`  ",
        f"Status: **{report.get('status')}**  ",
        f"Decision: **{report.get('decision', 'pending')}**",
        "",
        "## Authorization gate assessment",
        "",
        report.get("reason", ""),
        "",
    ]
    factors = report.get("factors", {})
    if factors and isinstance(factors, dict):
        lines.extend(
            [
                "| Factor | SPV0-02 evidence | Ceiling observed | Note |",
                "| --- | --- | --- | --- |",
            ]
        )
        for key, fac in factors.items():
            lines.append(
                f"| {key} | {fac.get('spv0_02_evidence')} | "
                f"{fac.get('ceiling_observed')} | {fac.get('note')} |"
            )
    blocked = report.get("blocked_heads")
    if blocked:
        lines.extend(["", "## Blocked heads", ""] + [f"- `{h}`" for h in blocked])
    next_step = report.get("recommended_next_step")
    if next_step:
        lines.extend(["", "## Recommended next step", "", next_step])
    note = report.get("note")
    if note:
        lines.extend(["", "## Note", "", note])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SLM-145 authorization-gate closeout")
    parser.add_argument(
        "--mode",
        choices=("plan-only", "closeout"),
        default="plan-only",
        help="plan-only emits the skeleton; closeout records the gate failure",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"outputs/runs/slm145-plan-predictor-factors-{_today_slug()}"),
    )
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    design_json = Path(f"docs/design/iter-slm145-plan-predictor-factors-{_today_slug()}.json")
    design_md = Path(f"docs/design/iter-slm145-plan-predictor-factors-{_today_slug()}.md")

    report = _plan_only_report() if args.mode == "plan-only" else _closeout_report()

    report_text = json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    report_path = output_dir / "slm145_gate_closeout_report.json"
    report_path.write_text(report_text, encoding="utf-8")
    markdown = _render_markdown(report)
    (output_dir / "slm145_gate_closeout_report.md").write_text(markdown, encoding="utf-8")

    design_json.parent.mkdir(parents=True, exist_ok=True)
    design_json.write_text(report_text, encoding="utf-8")
    design_md.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"\nReport JSON: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
