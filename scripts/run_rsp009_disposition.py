#!/usr/bin/env python3
"""Run SLM-490 (RSP-009) cross-experiment EXP-SR disposition audit.

Example:
  python -m scripts.run_rsp009_disposition --mode plan-only
  python -m scripts.run_rsp009_disposition --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.rsp009_cross_experiment_disposition import (
    EXPERIMENT_ID,
    MATRIX_SET,
    MATRIX_VERSION,
    build_exp_sr_disposition_records,
    render_markdown,
    run_rsp009_disposition_audit,
)
from slm_training.versioning import UNKNOWN, build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm490-rsp-009-disposition-20260810.json"
_DESIGN_MD = "docs/design/iter-slm490-rsp-009-disposition-20260810.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _build_version_stamp() -> dict[str, Any]:
    try:
        return build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm490_rsp009_disposition",
        )
    except KeyError:
        base = build_version_stamp("harness.experiments")
        base["components"]["harness.experiments.slm490_rsp009_disposition"] = UNKNOWN
        return base


def _build_payload(mode: str) -> tuple[dict[str, Any], str]:
    if mode == "plan-only":
        records = build_exp_sr_disposition_records()
        payload: dict[str, Any] = {
            "schema": "rsp009_cross_experiment_disposition/v1",
            "matrix_set": MATRIX_SET,
            "matrix_version": MATRIX_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "linear_issue": "SLM-490",
            "run_id": "slm490_rsp009_plan",
            "status": "plan_only",
            "claim_class": "wiring",
            "evidence_cutoff_commit": "not_applicable",
            "generated_at": _now(),
            "catalogue_families": [f"exp-sr-{i}" for i in range(1, 13)],
            "record_count": len(records),
            "cross_pack_summary": (
                "Preregistered RSP-009 manifest over EXP-SR-1..12. Run --mode fixture "
                "to validate evidence docs and publish the audited disposition report."
            ),
            "version_stamp": _build_version_stamp(),
        }
        return payload, "python -m scripts.run_rsp009_disposition --mode plan-only"

    report = run_rsp009_disposition_audit(
        run_id="slm490_rsp009_fixture",
        status="fixture",
    )
    return report.to_dict(), "python -m scripts.run_rsp009_disposition --mode fixture"


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    if payload.get("status") == "plan_only":
        return "\n".join(
            [
                "# SLM-490 (RSP-009): Cross-experiment EXP-SR disposition plan",
                "",
                "**Claim class:** wiring / disposition audit only",
                "",
                "**Machine-readable result:** "
                "[`iter-slm490-rsp-009-disposition-20260810.json`]"
                "(iter-slm490-rsp-009-disposition-20260810.json)",
                "",
                "Plan-only manifest. Run `--mode fixture` to validate evidence and "
                "publish the audited report.",
                "",
                "## Exact command",
                "",
                f"```bash\n{command}\n```",
                "",
            ]
        )

    from slm_training.harnesses.experiments.rsp009_cross_experiment_disposition import (
        Rsp009CrossExperimentDispositionV1,
        run_rsp009_disposition_audit,
    )

    # Rebuild typed report for markdown rendering (payload may have been serialized).
    report = run_rsp009_disposition_audit(
        run_id=payload.get("run_id", "slm490_rsp009_fixture"),
        status=payload.get("status", "fixture"),
    )
    return render_markdown(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-490 RSP-009 cross-experiment EXP-SR disposition audit",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={"plan-only", "fixture"},
        default="plan-only",
        help="plan-only writes the manifest; fixture runs the evidence audit.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for run artifacts "
            "(default: outputs/runs/slm490-rsp009-disposition-<YYYYMMDD>)"
        ),
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    output_dir = (
        args.output_dir
        or Path(f"outputs/runs/slm490-rsp009-disposition-{_today_yyyymmdd()}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, command = _build_payload(args.mode)
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    run_json = output_dir / "slm490_rsp009_disposition_report.json"
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
