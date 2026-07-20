#!/usr/bin/env python3
"""Run the SLM-183 powered cluster-aware confirmation protocol fixture.

Example:
  python -m scripts.run_flow_power_protocol --mode plan-only
  python -m scripts.run_flow_power_protocol --mode fixture
  python -m scripts.run_flow_power_protocol --mode analyze-existing \
      --iter-json outputs/runs/some-run/iter.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm183_power_protocol import (
    MATRIX_SET,
    EXPERIMENT_ID,
    MATRIX_VERSION,
    ConfirmationSuiteManifest,
    PowerProtocolReport,
    analyze_existing_iter,
    build_default_manifest,
    render_markdown,
    run_variance_fixture,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm183-power-protocol-20260720.json"
_DESIGN_MD = "docs/design/iter-slm183-power-protocol-20260720.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def _build_payload(
    mode: str,
    output_dir: Path,
    seeds: tuple[int, ...],
    n_targets: int,
    paths_per_target: int,
    n_seeds: int,
    iter_json: Path | None,
) -> tuple[dict[str, Any], str]:
    manifest = build_default_manifest(seeds=seeds)

    if mode == "plan-only":
        payload: dict[str, Any] = {
            "schema": "Slm183PowerProtocolManifestV1",
            "matrix_set": MATRIX_SET,
            "matrix_version": MATRIX_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "status": "plan_only",
            "claim_class": "wiring",
            "manifest": manifest.to_dict(),
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm183_power_protocol",
                "evals.power_protocol",
            ),
            "timestamp": _now(),
        }
        command = "python -m scripts.run_flow_power_protocol --mode plan-only"
        return payload, command

    if mode == "analyze-existing":
        if iter_json is None:
            raise ValueError("--mode analyze-existing requires --iter-json")
        analysis = analyze_existing_iter(iter_json)
        payload = {
            "schema": "Slm183PowerProtocolAnalysisV1",
            "matrix_set": MATRIX_SET,
            "matrix_version": MATRIX_VERSION,
            "experiment_id": EXPERIMENT_ID,
            "status": "analysis",
            "claim_class": "wiring",
            "analysis": analysis,
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm183_power_protocol",
                "evals.power_protocol",
            ),
            "timestamp": _now(),
        }
        command = (
            f"python -m scripts.run_flow_power_protocol --mode analyze-existing "
            f"--iter-json {iter_json}"
        )
        return payload, command

    report = run_variance_fixture(
        n_targets=n_targets,
        paths_per_target=paths_per_target,
        n_seeds=n_seeds,
        run_id=f"slm183-power-protocol-{_today_yyyymmdd()}",
        output_dir=output_dir,
        seed=seeds[0] if seeds else 0,
    )
    payload = report.to_dict()
    command = "python -m scripts.run_flow_power_protocol --mode fixture"
    return payload, command


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    status = payload.get("status", "fixture")
    if status == "plan_only":
        manifest = ConfirmationSuiteManifest.from_dict(payload["manifest"])
        lines = [
            "# SLM-183 (PQR): powered cluster-aware confirmation protocol plan",
            "",
            "**Claim class:** wiring / fixture only",
            "",
            "**Run date:** 2026-07-20",
            "",
            "**Machine-readable result:** ["
            "`iter-slm183-power-protocol-20260720.json`"
            "](iter-slm183-power-protocol-20260720.json)",
            "",
            "This is a plan-only manifest. The power-protocol statistical "
            "utilities, cluster bootstrap, ICC, and MDE simulation are wired; "
            "run `--mode fixture` to execute the CPU-only simulation.",
            "",
            "## Hypothesis",
            "",
            "A powered confirmation protocol can separate seed variance from "
            "target variance and estimate the minimum detectable effect size.",
            "",
            "## Falsifier",
            "",
            "The protocol collapses seed and target variance into a single "
            "pooled estimate and cannot produce a calibrated MDE curve.",
            "",
            "## Manifest",
            "",
            f"- suite_role: `{manifest.suite_role}`",
            f"- generator_version: `{manifest.generator_version}`",
            f"- target_cluster_id: `{manifest.target_cluster_id}`",
            f"- primary_endpoint: `{manifest.primary_endpoint}`",
            f"- primary_contrast: `{manifest.primary_contrast}`",
            f"- mde: `{manifest.mde}`",
            f"- alpha: `{manifest.alpha}`",
            f"- power: `{manifest.power}`",
            f"- multiplicity_family: `{manifest.multiplicity_family}`",
            "",
            "## Exact command",
            "",
            f"```bash\n{command}\n```",
            "",
        ]
        return "\n".join(lines)

    if status == "analysis":
        analysis = payload.get("analysis", {})
        lines = [
            "# SLM-183 (PQR): existing-iter power-protocol analysis",
            "",
            f"Source: `{analysis.get('source', 'unknown')}`",
            "",
            f"- n_records: **{analysis.get('n_records', 0)}**",
            f"- n_successes: **{analysis.get('n_successes', 0)}**",
            f"- success_rate: **{analysis.get('success_rate', 0.0):.3f}**",
            f"- seed_variance: **{analysis.get('seed_variance', 0.0):.4f}**",
            "",
            "## Exact command",
            "",
            f"```bash\n{command}\n```",
            "",
        ]
        return "\n".join(lines)

    report = PowerProtocolReport.from_dict(payload)
    return render_markdown(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-183 PQR powered cluster-aware confirmation protocol fixture",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={"plan-only", "fixture", "analyze-existing"},
        default="plan-only",
        help="Run mode: plan-only writes the manifest; fixture runs the CPU simulation; "
        "analyze-existing reads an iter JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for run artifacts (default: outputs/runs/slm183-power-protocol-<YYYYMMDD>)",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_int_tuple,
        default="0,1,2,3,4",
        help="Comma-separated random seeds for fixture mode (default: 0,1,2,3,4).",
    )
    parser.add_argument(
        "--n-targets",
        type=int,
        default=50,
        help="Number of synthetic targets (default: 50).",
    )
    parser.add_argument(
        "--paths-per-target",
        type=int,
        default=3,
        help="Independent paths per target (default: 3).",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of seeds per target (default: 5).",
    )
    parser.add_argument(
        "--iter-json",
        type=Path,
        help="Path to an existing iter JSON for --mode analyze-existing.",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    output_dir = args.output_dir or Path(
        f"outputs/runs/slm183-power-protocol-{_today_yyyymmdd()}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        payload, command = _build_payload(
            args.mode,
            output_dir,
            args.seeds,
            args.n_targets,
            args.paths_per_target,
            args.n_seeds,
            args.iter_json,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload["schema"] = "Slm183PowerProtocolReportV1"
    payload["claim_class"] = "wiring"
    payload["status"] = payload.get("status", "fixture")
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    run_json = output_dir / "slm183_power_protocol_report.json"
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
