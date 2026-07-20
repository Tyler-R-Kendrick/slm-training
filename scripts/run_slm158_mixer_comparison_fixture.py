#!/usr/bin/env python3
"""Run the SLM-158 (SPV3-05) sequence-mixer comparison fixture.

Example:
  python -m scripts.run_slm158_mixer_comparison_fixture --mode plan-only
  python -m scripts.run_slm158_mixer_comparison_fixture --mode fixture
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from slm_training.harnesses.experiments.slm158_mixer_comparison import (
    CommonConfig,
    MixerManifest,
    MixerReport,
    build_manifest,
    render_markdown,
    run_fixture_campaign,
)
from slm_training.versioning import build_version_stamp

_DESIGN_JSON = "docs/design/iter-slm158-mixer-comparison-20260720.json"
_DESIGN_MD = "docs/design/iter-slm158-mixer-comparison-20260720.md"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _today_yyyymmdd() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _parse_seeds(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def _build_payload(
    mode: str,
    output_dir: Path,
    n_train: int,
    n_eval: int,
    epochs: int,
    seeds: tuple[int, ...],
) -> tuple[dict[str, Any], str]:
    manifest = build_manifest()

    if mode == "plan-only":
        payload: dict[str, Any] = {
            "schema": "Slm158MixerComparisonManifestV1",
            "matrix_set": manifest.matrix_set,
            "matrix_version": manifest.matrix_version,
            "experiment_id": manifest.experiment_id,
            "status": "plan_only",
            "claim_class": "wiring",
            "manifest": manifest.to_dict(),
            "version_stamp": build_version_stamp(
                "harness.experiments",
                "harness.experiments.slm158_mixer_comparison",
            ),
            "timestamp": _now(),
        }
        command = "python -m scripts.run_slm158_mixer_comparison_fixture --mode plan-only"
        return payload, command

    cfg = CommonConfig(
        seq_len=manifest.common_config.seq_len,
        vocab_size=manifest.common_config.vocab_size,
        d_model=manifest.common_config.d_model,
        n_layers=manifest.common_config.n_layers,
        n_classes=manifest.common_config.n_classes,
        n_train=n_train,
        n_eval=n_eval,
        seeds=seeds,
        lr=manifest.common_config.lr,
        epochs=epochs,
        batch_size=manifest.common_config.batch_size,
        mixer_state_dim=manifest.common_config.mixer_state_dim,
        metric_versions=manifest.common_config.metric_versions,
    )
    manifest = MixerManifest(
        matrix_set=manifest.matrix_set,
        matrix_version=manifest.matrix_version,
        experiment_id=manifest.experiment_id,
        hypothesis=manifest.hypothesis,
        falsifier=manifest.falsifier,
        common_config=cfg,
        arms=manifest.arms,
        claim_class=manifest.claim_class,
        status=manifest.status,
    )
    report = run_fixture_campaign(
        manifest=manifest,
        run_id="slm158_fixture",
        output_dir=output_dir,
    )
    payload = report.to_dict()
    command = "python -m scripts.run_slm158_mixer_comparison_fixture --mode fixture"
    return payload, command


def _build_markdown(payload: dict[str, Any], command: str) -> str:
    status = payload.get("status", "fixture")
    if status == "plan_only":
        manifest = MixerManifest.from_dict(payload["manifest"])
        lines = [
            "# SLM-158 (SPV3-05): Sequence-mixer comparison plan",
            "",
            "**Claim class:** wiring / fixture only",
            "",
            "**Run date:** 2026-07-20",
            "",
            "**Machine-readable result:** ["
            "`iter-slm158-mixer-comparison-20260720.json`](iter-slm158-mixer-comparison-20260720.json)",
            "",
            "This is a plan-only manifest. The staged mixer arms, common "
            "config, and validation rules are wired; run `--mode fixture` to execute "
            "the CPU matrix.",
            "",
            "## Hypothesis",
            "",
            manifest.hypothesis,
            "",
            "## Falsifier",
            "",
            manifest.falsifier,
            "",
            "## Arms",
            "",
            "| Arm | Family | Promotable | Reference | Description |",
            "| --- | --- | --- | --- | --- |",
        ]
        for arm in manifest.arms:
            lines.append(
                f"| {arm.arm_id} | {arm.family.value} | {arm.promotable} | "
                f"{arm.reference} | {arm.description} |"
            )
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

    report = MixerReport.from_dict(payload)
    return render_markdown(report)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-158 SPV3-05 sequence-mixer comparison fixture",
        exit_on_error=False,
    )
    parser.add_argument(
        "--mode",
        choices={"plan-only", "fixture"},
        default="plan-only",
        help="Run mode: plan-only writes the manifest; fixture runs the CPU matrix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for run artifacts (default: outputs/runs/slm158-fixture-<YYYYMMDD>)",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=128,
        help="Number of synthetic training sequences per seed in fixture mode.",
    )
    parser.add_argument(
        "--n-eval",
        type=int,
        default=32,
        help="Number of synthetic evaluation sequences per seed in fixture mode.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Training epochs for the shared fixture classifier.",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_seeds,
        default="0,1",
        help="Comma-separated random seeds for fixture mode (default: 0,1).",
    )
    try:
        args = parser.parse_args(argv)
    except (argparse.ArgumentError, SystemExit):
        return 2

    output_dir = (
        args.output_dir or Path(f"outputs/runs/slm158-fixture-{_today_yyyymmdd()}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    payload, command = _build_payload(
        args.mode,
        output_dir,
        args.n_train,
        args.n_eval,
        args.epochs,
        args.seeds,
    )
    payload["schema"] = "Slm158MixerComparisonReportV1"
    payload["claim_class"] = "wiring"
    payload["status"] = payload.get("status", "fixture")
    payload["timestamp"] = _now()

    report_text = json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"

    run_json = output_dir / "slm158_mixer_comparison_report.json"
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
