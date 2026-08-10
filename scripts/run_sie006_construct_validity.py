#!/usr/bin/env python3
"""Run SLM-483 (SIE-006) EXP-SR-3 construct-validity + calibration campaign.

Example:
  python -m scripts.run_sie006_construct_validity --mode plan-only
  python -m scripts.run_sie006_construct_validity --mode fixture
  python -m scripts.run_sie006_construct_validity --mode external-blocked
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.sie006_construct_validity import (
    CATALOGUE_ID,
    PRIMARY_METRIC,
    SIE005_FROZEN_DOC,
    Sie006CampaignV1,
    plan_only_preview,
    render_markdown,
    run_external_blocked_campaign,
    run_fixture_campaign,
)
from slm_training.versioning import build_version_stamp

_DESIGN_STEM = "docs/design/iter-slm483-sie-006-construct-validity-20260810"
_VERSION_COMPONENT = "harness.experiments.slm483_sie006_construct_validity"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_markdown(docs_json: Path, payload: dict[str, Any], command: str) -> Path:
    md_path = docs_json.with_suffix(".md")
    md_path.write_text(render_markdown(payload) + f"\nCommand: `{command}`\n", encoding="utf-8")
    return md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("plan-only", "fixture", "external-blocked"),
        default="plan-only",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/sie006_construct_validity"),
    )
    parser.add_argument(
        "--sie005-doc",
        type=Path,
        default=SIE005_FROZEN_DOC,
        help="Frozen SIE-005 packet pin (default: durable design doc)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--write-design",
        action="store_true",
        help=f"Write durable {_DESIGN_STEM}.{{json,md}}",
    )
    parser.add_argument(
        "--docs-out",
        type=Path,
        default=Path(f"{_DESIGN_STEM}.json"),
    )
    args = parser.parse_args(argv)

    command = " ".join(
        [
            "python",
            "-m",
            "scripts.run_sie006_construct_validity",
            "--mode",
            args.mode,
            "--seed",
            str(args.seed),
        ]
    )

    if args.mode == "plan-only":
        payload = plan_only_preview()
        payload["timestamp"] = _now()
        payload["version_stamp"] = build_version_stamp(_VERSION_COMPONENT)
        print(json.dumps(payload, indent=2, sort_keys=True))
        if args.write_design:
            args.docs_out.parent.mkdir(parents=True, exist_ok=True)
            args.docs_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            _write_markdown(args.docs_out, payload, command)
            print(f"wrote {args.docs_out}")
            print(f"wrote {args.docs_out.with_suffix('.md')}")
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    campaign = Sie006CampaignV1(seed=args.seed, sie005_frozen_doc=str(args.sie005_doc))
    if args.mode == "fixture":
        result = run_fixture_campaign(campaign, root=args.output_dir)
    else:
        result = run_external_blocked_campaign(campaign, root=args.output_dir)

    payload = {
        "kind": result["schema"],
        "matrix_set": "slm483_sie006_construct_validity",
        "matrix_version": "sie006-v1",
        "catalogue_id": CATALOGUE_ID,
        "primary_metric": PRIMARY_METRIC,
        "recipe": {
            "command": command.split(),
            "output_dir": str(args.output_dir),
            "docs_path": str(args.docs_out),
            "device": "cpu",
            "honesty_mode": "fixture_diagnostic"
            if args.mode == "fixture"
            else "external_blocked",
        },
        "timestamp": _now(),
        **result,
    }
    payload.setdefault("version_stamp", build_version_stamp(_VERSION_COMPONENT))

    run_json = args.output_dir / "sie006_construct_validity_report.json"
    run_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.output_dir / "sie006_construct_validity_report.md").write_text(
        render_markdown(payload) + f"\nCommand: `{command}`\n",
        encoding="utf-8",
    )

    args.docs_out.parent.mkdir(parents=True, exist_ok=True)
    args.docs_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path = _write_markdown(args.docs_out, payload, command)

    print(f"wrote {run_json}")
    print(f"wrote {args.docs_out}")
    print(f"wrote {md_path}")
    print(f"status={payload.get('status')} disposition={payload.get('disposition')}")
    print(f"external_blocked={payload.get('external_blocked')}")
    if payload.get("evaluator_human_agreement_kappa") is not None:
        print(f"evaluator_human_agreement_kappa={payload['evaluator_human_agreement_kappa']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
