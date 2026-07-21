#!/usr/bin/env python3
"""Run the SLM-229 (RSC0-01) looped-latent differentiation audit.

Zero-compute, docs/schema-only research audit. No model/data implementation,
no training run, no checkpoint.

Example:
  python -m scripts.run_slm229_looped_latent_differentiation --mode plan-only
  python -m scripts.run_slm229_looped_latent_differentiation --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm229_looped_latent_differentiation import (
    LoopedLatentDifferentiationReport,
    LoopedLatentVerdict,
    build_differentiators,
    build_mechanism_comparison,
    build_oracle_intervention_ceiling,
    build_prior_art_audit,
    build_scale_regime_audit,
    build_target_support_audit,
    render_markdown,
    run_differentiation_audit,
)
from slm_training.versioning import UNKNOWN, build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm229-looped-latent-differentiation-20260721.json"
_DESIGN_MD = "docs/design/iter-slm229-looped-latent-differentiation-20260721.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _build_version_stamp() -> dict[str, Any]:
    """Build a version stamp, degrading if the slm229 component is not yet registered."""
    try:
        return build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm229_looped_latent_differentiation",
        )
    except KeyError:
        base = build_version_stamp("harness.experiments")
        base["components"]["harness.experiments.slm229_looped_latent_differentiation"] = UNKNOWN
        return base


def _build_payload(mode: str, output_dir: Path) -> tuple[dict[str, Any], str]:
    if mode == "plan-only":
        report = LoopedLatentDifferentiationReport(
            schema="LoopedLatentDifferentiationV1",
            matrix_set="slm229_looped_latent_differentiation",
            matrix_version="rsc0-01-v1",
            experiment_id="slm229-looped-latent-differentiation",
            run_id="slm229_differentiation_plan",
            status="plan_only",
            claim_class="wiring",
            source_commit="not_applicable",
            evidence_cutoff="not_applicable",
            reviewed_refs=(),
            generated_at=_now(),
            mechanism_comparison=tuple(build_mechanism_comparison()),
            target_support_audit=tuple(build_target_support_audit()),
            oracle_intervention_ceiling=build_oracle_intervention_ceiling(),
            scale_regime_audit=build_scale_regime_audit(),
            prior_art_audit=tuple(build_prior_art_audit()),
            differentiators=tuple(build_differentiators()),
            verdict=LoopedLatentVerdict.INCONCLUSIVE,
            allowed_implementation_scope=(
                "No audited allowed scope in plan-only mode; run --mode fixture."
            ),
            forbidden_duplicate_scope=(
                "No audited forbidden scope in plan-only mode; run --mode fixture."
            ),
            resolving_evidence="Preregistered manifest only; run --mode fixture to validate evidence docs.",
            contract_hash=None,
            minimal_contract=None,
            version_stamp=_build_version_stamp(),
        )
        command = "python -m scripts.run_slm229_looped_latent_differentiation --mode plan-only"
        return report.to_dict(), command

    report = run_differentiation_audit(
        run_id="slm229_differentiation_fixture",
        status="fixture",
    )
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        report.to_json(output_dir / "slm229_looped_latent_differentiation_report.json")
    command = "python -m scripts.run_slm229_looped_latent_differentiation --mode fixture"
    return report.to_dict(), command


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    if payload.get("status") == "plan_only":
        lines = [
            "# SLM-229 (RSC0-01): Looped-latent differentiation plan",
            "",
            "**Claim class:** wiring / zero-compute differentiation and "
            "authorization audit only",
            "",
            "**Run date:** 2026-07-21",
            "",
            "**Machine-readable result:** ["
            "`iter-slm229-looped-latent-differentiation-20260721.json`"
            "](iter-slm229-looped-latent-differentiation-20260721.json)",
            "",
            "This is a plan-only manifest. The preregistered mechanism "
            "comparison, target-support audit, oracle-ceiling, scale audit, "
            "prior-art audit, and 7 differentiators are wired; run `--mode "
            "fixture` to validate evidence docs, evaluate the decision rule, "
            "and produce the audited report.",
            "",
            "## Exact command",
            "",
            f"```bash\n{command}\n```",
            "",
        ]
        return "\n".join(lines)

    report = LoopedLatentDifferentiationReport.from_dict(payload)
    return render_markdown(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-229 (RSC0-01) looped-latent differentiation audit",
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
            "(default: outputs/runs/slm229-differentiation-<YYYYMMDD>)"
        ),
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    output_dir = (
        args.output_dir
        or Path(f"outputs/runs/slm229-differentiation-{_today_yyyymmdd()}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, command = _build_payload(args.mode, output_dir)
    payload["schema"] = "LoopedLatentDifferentiationV1"
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    run_json = output_dir / "slm229_looped_latent_differentiation_report.json"
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
