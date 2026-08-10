#!/usr/bin/env python3
"""Run the SLM-474 (SRP-010) symbolic-pack portability fixture.

Example:
  python -m scripts.run_srp010_pack_portability_fixture --mode plan-only
  python -m scripts.run_srp010_pack_portability_fixture --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.srp010_pack_portability import (
    CLAIM_CLASS,
    MATRIX_SET,
    MATRIX_VERSION,
    PORTABILITY_CAMPAIGN_ID,
    RELATED_CATALOGUE_ID,
    VERSION_STAMP_COMPONENTS,
    PackPortabilityReport,
    render_markdown,
    run_fixture_campaign,
)
from slm_training.versioning import build_version_stamp

_DESIGN_STEM = "docs/design/iter-slm474-srp-010-portability-20260810"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _plan_only_payload() -> dict[str, Any]:
    return {
        "schema": "Slm474Srp010PackPortabilityReportV1",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "experiment_id": PORTABILITY_CAMPAIGN_ID,
        "related_catalogue_id": RELATED_CATALOGUE_ID,
        "status": "plan_only",
        "claim_class": CLAIM_CLASS,
        "hypothesis": (
            "The symbolic-regression pack exercises the same durable "
            "pack/synthesis/evidence contracts as OpenUI without core forks "
            "or changing OpenUI defaults."
        ),
        "falsifier": (
            "Default isolation breaks, the pack cannot run "
            "parse/canonicalize/oracle/evaluate-fit/evidence on a tiny "
            "problem, OpenUI imports leak into pack modules, or forks/"
            "unsupported hooks are silent."
        ),
        "steps": [
            "Confirm get_pack()/auto/default remain openui with env unset",
            "Load get_pack('symbolic_regression')",
            "Build SymbolicRegressionProblemV1 + VerifiedSynthesisProblemV1",
            "Parse/canonicalize/oracle/evaluate-fit/evidence",
            "build_sygus_capability_report on the VSP envelope",
            "AST-import-audit pack modules for openui imports",
            "Honestly inventory forks + exercise optional SRP-008/009 corpus/enumerate APIs",
        ],
        "version_stamp": build_version_stamp(*VERSION_STAMP_COMPONENTS),
        "timestamp": _now(),
    }


def _build_payload(mode: str, output_dir: Path) -> tuple[dict[str, Any], str]:
    if mode == "plan-only":
        return _plan_only_payload(), (
            "python -m scripts.run_srp010_pack_portability_fixture --mode plan-only"
        )
    report = run_fixture_campaign(run_id="srp010_fixture", output_dir=output_dir)
    payload = report.to_dict()
    payload["schema"] = "Slm474Srp010PackPortabilityReportV1"
    return payload, (
        "python -m scripts.run_srp010_pack_portability_fixture --mode fixture"
    )


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    if payload.get("status") == "plan_only":
        lines = [
            "# SLM-474 (SRP-010): Symbolic-pack portability plan",
            "",
            f"**Claim class:** `{CLAIM_CLASS}` only",
            "",
            "**Run date:** 2026-08-10",
            "",
            "**Machine-readable result:** ["
            "`iter-slm474-srp-010-portability-20260810.json`"
            "](iter-slm474-srp-010-portability-20260810.json)",
            "",
            "Plan-only manifest. Related catalogue identity "
            f"`{RELATED_CATALOGUE_ID}` is cited, not locked. Run "
            "`--mode fixture` to exercise the pack path.",
            "",
            "## Hypothesis",
            "",
            str(payload.get("hypothesis", "")),
            "",
            "## Falsifier",
            "",
            str(payload.get("falsifier", "")),
            "",
            "## Steps",
            "",
        ]
        for step in payload.get("steps", []):
            lines.append(f"- {step}")
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
    return render_markdown(PackPortabilityReport.from_dict(payload))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-474 SRP-010 symbolic-pack portability fixture",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={"plan-only", "fixture"},
        default="plan-only",
        help="plan-only writes the manifest; fixture runs the pack path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for run artifacts "
            "(default: outputs/runs/srp010-fixture-<YYYYMMDD>)"
        ),
    )
    parser.add_argument(
        "--write-design",
        action="store_true",
        help=(
            "Also write docs/design/iter-slm474-srp-010-portability-20260810."
            "{json,md} (opt-in so fixture tests do not dirty the tree)"
        ),
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    output_dir = args.output_dir or Path(
        f"outputs/runs/srp010-fixture-{_today_yyyymmdd()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, command = _build_payload(args.mode, output_dir)
    payload["schema"] = "Slm474Srp010PackPortabilityReportV1"
    payload["claim_class"] = CLAIM_CLASS
    payload["status"] = payload.get("status", "fixture")
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    run_json = output_dir / "srp010_pack_portability_report.json"
    run_json.write_text(report_text, encoding="utf-8")

    if args.mode == "fixture" and args.write_design:
        root = Path(__file__).resolve().parents[1]
        json_path = root / f"{_DESIGN_STEM}.json"
        md_path = root / f"{_DESIGN_STEM}.md"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        command_line = command
        if args.output_dir is not None:
            command_line += f" --output-dir {output_dir}"
        json_path.write_text(report_text, encoding="utf-8")
        md_path.write_text(_build_markdown(payload, command_line), encoding="utf-8")

    print(str(run_json))
    return 0


if __name__ == "__main__":
    sys.exit(main())
