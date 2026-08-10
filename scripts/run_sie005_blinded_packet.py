#!/usr/bin/env python3
"""Run the SLM-478 (SIE-005) blinded construct-validity packet fixture.

Example:
  python -m scripts.run_sie005_blinded_packet --mode plan-only
  python -m scripts.run_sie005_blinded_packet --mode fixture
  python -m scripts.run_sie005_blinded_packet --mode fixture --write-design
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.sie005_blinded_packet import (
    AUDIT_ID,
    CLAIM_CLASS,
    MATRIX_SET,
    MATRIX_VERSION,
    RELATED_CATALOGUE_ID,
    SIE005_STRATA,
    VERSION_STAMP_COMPONENTS,
    BlindedPacketReport,
    render_markdown,
    run_fixture_packet,
)
from slm_training.versioning import build_version_stamp

_DESIGN_STEM = "docs/design/iter-slm478-sie-005-blinded-packet-20260810"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _plan_only_payload() -> dict[str, Any]:
    return {
        "schema": "Slm478Sie005BlindedPacketReportV1",
        "matrix_set": MATRIX_SET,
        "matrix_version": MATRIX_VERSION,
        "audit_id": AUDIT_ID,
        "related_catalogue_id": RELATED_CATALOGUE_ID,
        "status": "plan_only",
        "claim_class": CLAIM_CLASS,
        "hypothesis": (
            "A stratified, blinded annotation + external-judge packet can be "
            "frozen and round-tripped without credentials, without leaking "
            "checkpoint identity or deterministic verdicts, and without "
            "admitting the audit sample into training."
        ),
        "falsifier": (
            "Public packet leaks identity/verdicts/secrets, import mishandles "
            "duplicates/conflicts/adjudication, or the harness requires a real "
            "provider credential to exercise."
        ),
        "strata": list(SIE005_STRATA),
        "steps": [
            "Build fixture stratified audit rows (hard-valid, metamorphic, model-output, ambiguity)",
            "freeze_blinded_pairs with opaque rec_* IDs; private key outside package",
            "Emit randomized annotation_order + two-rater/adjudication label schema (no auto scores)",
            "Pin ExternalJudgeRequestV1/ResponseV1 + independence declaration + mock transports",
            "Write credential runbook (env-only secrets)",
            "Import round-trip: identical duplicates OK, conflicts rejected, adjudication complete",
        ],
        "version_stamp": build_version_stamp(*VERSION_STAMP_COMPONENTS),
        "timestamp": _now(),
    }


def _build_payload(mode: str, output_dir: Path, seed: int) -> tuple[dict[str, Any], str]:
    if mode == "plan-only":
        return _plan_only_payload(), (
            "python -m scripts.run_sie005_blinded_packet --mode plan-only"
        )
    report = run_fixture_packet(output_dir=output_dir, seed=seed)
    payload = report.to_dict()
    return payload, (
        "python -m scripts.run_sie005_blinded_packet --mode fixture"
    )


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    if payload.get("status") == "plan_only":
        return "\n".join(
            [
                "# SLM-478 (SIE-005): Blinded packet plan",
                "",
                f"**Claim class:** `{CLAIM_CLASS}` only",
                "",
                "**Run date:** 2026-08-10",
                "",
                "**Machine-readable result:** "
                "[`iter-slm478-sie-005-blinded-packet-20260810.json`]"
                "(iter-slm478-sie-005-blinded-packet-20260810.json)",
                "",
                f"**Related catalogue:** `{RELATED_CATALOGUE_ID}` (SIE-006 executes the campaign)",
                "",
                "## Steps",
                "",
                *[f"{i}. {s}" for i, s in enumerate(payload.get("steps") or [], 1)],
                "",
                f"Command: `{command}`",
                "",
            ]
        )
    return render_markdown(payload) + f"\nCommand: `{command}`\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan-only", "fixture"), default="plan-only")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/sie005_blinded_packet"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--write-design",
        action="store_true",
        help="Write durable docs/design/iter-slm478-sie-005-*.{json,md}",
    )
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload, command = _build_payload(args.mode, args.output_dir, args.seed)
    payload.setdefault("timestamp", _now())
    payload.setdefault("version_stamp", build_version_stamp(*VERSION_STAMP_COMPONENTS))

    run_json = args.output_dir / "sie005_blinded_packet_report.json"
    run_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = _build_markdown(payload, command)
    (args.output_dir / "sie005_blinded_packet_report.md").write_text(md, encoding="utf-8")

    if args.write_design:
        design_json = Path(f"{_DESIGN_STEM}.json")
        design_md = Path(f"{_DESIGN_STEM}.md")
        design_json.parent.mkdir(parents=True, exist_ok=True)
        design_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        design_md.write_text(md, encoding="utf-8")
        print(f"wrote {design_json}")
        print(f"wrote {design_md}")

    print(f"wrote {run_json}")
    status = payload.get("status")
    if isinstance(payload, dict) and status == "fixture":
        # Smoke-check fields for operators.
        report = BlindedPacketReport(
            status=str(status),
            pair_n=int(payload.get("pair_n") or 0),
            blinding_ok=bool(payload.get("blinding_ok")),
            training_excluded=bool(payload.get("training_excluded")),
            independence_ok=bool(payload.get("independence_ok")),
        )
        if not (report.blinding_ok and report.training_excluded and report.independence_ok):
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
