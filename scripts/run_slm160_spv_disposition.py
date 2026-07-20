#!/usr/bin/env python3
"""Run the SLM-160 (SPV4-02) causal architecture disposition audit.

Example:
  python -m scripts.run_slm160_spv_disposition --mode plan-only
  python -m scripts.run_slm160_spv_disposition --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm160_spv_disposition import (
    SPVDispositionReport,
    build_default_dispositions,
    render_markdown,
    run_disposition_audit,
)
from slm_training.versioning import UNKNOWN, build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm160-spv-disposition-20260720.json"
_DESIGN_MD = "docs/design/iter-slm160-spv-disposition-20260720.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _build_version_stamp() -> dict[str, Any]:
    """Build a version stamp, degrading if the slm160 component is not yet registered."""
    try:
        return build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm160_spv_disposition",
        )
    except KeyError:
        base = build_version_stamp("harness.experiments")
        base["components"]["harness.experiments.slm160_spv_disposition"] = UNKNOWN
        return base


def _build_payload(mode: str, output_dir: Path) -> tuple[dict[str, Any], str]:
    if mode == "plan-only":
        report = SPVDispositionReport(
            schema="SPVDispositionV1",
            matrix_set="slm160_spv_disposition",
            matrix_version="spv4-02-v1",
            experiment_id="slm160-spv-disposition",
            run_id="slm160_disposition_plan",
            status="plan_only",
            claim_class="wiring",
            evidence_cutoff_commit="not_applicable",
            generated_at=_now(),
            mechanism_dispositions=build_default_dispositions(),
            cross_pack_summary=(
                "Preregistered SPV4-02 disposition manifest. Run --mode fixture "
                "to validate evidence docs and produce the audited report."
            ),
            canonical_architecture_recommendation=(
                "No audited recommendation in plan-only mode; run fixture mode."
            ),
            rejected_or_blocked_ids=[],
            version_stamp=_build_version_stamp(),
        )
        command = "python -m scripts.run_slm160_spv_disposition --mode plan-only"
        return report.to_dict(), command

    report = run_disposition_audit(
        run_id="slm160_disposition_fixture",
        status="fixture",
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm160_spv_disposition_report.json")
    command = "python -m scripts.run_slm160_spv_disposition --mode fixture"
    return report.to_dict(), command


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    if payload.get("status") == "plan_only":
        lines = [
            "# SLM-160 (SPV4-02): Causal architecture disposition plan",
            "",
            "**Claim class:** wiring / disposition audit only",
            "",
            "**Run date:** 2026-07-20",
            "",
            "**Machine-readable result:** ["
            "`iter-slm160-spv-disposition-20260720.json`"
            "](iter-slm160-spv-disposition-20260720.json)",
            "",
            "This is a plan-only manifest. The preregistered dispositions are "
            "wired; run `--mode fixture` to validate evidence docs and produce "
            "the audited report.",
            "",
            "## Exact command",
            "",
            f"```bash\n{command}\n```",
            "",
        ]
        return "\n".join(lines)

    report = SPVDispositionReport.from_dict(payload)
    return render_markdown(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-160 SPV4-02 causal architecture disposition audit",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={"plan-only", "fixture"},
        default="plan-only",
        help="Run mode: plan-only writes the manifest; fixture runs the audit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for run artifacts "
            "(default: outputs/runs/slm160-disposition-<YYYYMMDD>)"
        ),
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    output_dir = (
        args.output_dir or Path(f"outputs/runs/slm160-disposition-{_today_yyyymmdd()}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, command = _build_payload(args.mode, output_dir)
    payload["schema"] = "SPVDispositionV1"
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    run_json = output_dir / "slm160_spv_disposition_report.json"
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
