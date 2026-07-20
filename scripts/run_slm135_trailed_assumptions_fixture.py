#!/usr/bin/env python3
"""Run the SLM-135 EFS4-01 trailed-assumptions ablation fixture.

Example (plan only):
  python -m scripts.run_slm135_trailed_assumptions_fixture --mode plan-only

Example (fixture wiring check):
  python -m scripts.run_slm135_trailed_assumptions_fixture --mode fixture
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from slm_training.harnesses.experiments.slm135_trailed_assumptions_ablation import (
    AblationResult,
    Slm135Manifest,
    Slm135Report,
    Slm135Row,
    build_manifest,
    render_markdown,
    run_fixture_matrix,
)
from slm_training.versioning import build_version_stamp


def _today_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _plan_only_report(manifest: Slm135Manifest) -> Slm135Report:
    """Return a torch-free plan-only report skeleton."""
    rows: list[Slm135Row] = []
    for arm in manifest.arms:
        for seed in manifest.seeds:
            rows.append(
                Slm135Row(
                    arm_id=arm.arm_id,
                    policy=arm.policy,
                    seed=seed,
                    status="plan_only",
                    result=AblationResult(
                        policy=arm.policy,
                        status="plan_only",
                        terminal=(),
                        decisions=(),
                        deductions=(),
                        nogoods=(),
                        backtracks=0,
                        nodes=0,
                        false_prune=False,
                        unknown_violation=False,
                        leaked_deductions=(),
                        restored_fingerprint=None,
                        stop_reason=None,
                    ),
                    decision_count=0,
                    backtrack_count=0,
                    false_prune=False,
                    leaked_deduction_count=0,
                    notes=["plan-only: no search executed"],
                )
            )
    return Slm135Report(
        matrix_set=manifest.matrix_set,
        matrix_version=manifest.matrix_version,
        run_id="slm135_plan",
        status="plan_only",
        manifest=replace(manifest, status="plan_only"),
        rows=rows,
        verdict="not_run",
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.slm135_trailed_assumptions_ablation",
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-135 EFS4-01 trailed-assumptions ablation fixture"
    )
    parser.add_argument(
        "--mode",
        choices=("plan-only", "fixture"),
        default="plan-only",
        help=(
            "plan-only emits the manifest and report skeleton; "
            "fixture runs the closed CSP ablation and emits a full report"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"outputs/runs/slm135-trailed-assumptions-{_today_slug()}"),
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated seeds for the rows",
    )
    args = parser.parse_args(argv)

    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    manifest = build_manifest(seeds=seeds)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    design_json = Path(
        f"docs/design/iter-slm135-trailed-assumptions-{_today_slug()}.json"
    )
    design_md = Path(
        f"docs/design/iter-slm135-trailed-assumptions-{_today_slug()}.md"
    )

    if args.mode == "plan-only":
        manifest = replace(manifest, status="plan_only")
        report = _plan_only_report(manifest)
        manifest.to_json(output_dir / "slm135_trailed_assumptions_manifest.json")
    else:
        report = run_fixture_matrix(
            manifest,
            run_id="slm135_fixture",
            output_dir=output_dir,
        )

    markdown = render_markdown(report)

    report_path = output_dir / "slm135_trailed_assumptions_report.json"
    report.to_json(report_path)
    (output_dir / "slm135_trailed_assumptions_report.md").write_text(
        markdown, encoding="utf-8"
    )

    design_json.parent.mkdir(parents=True, exist_ok=True)
    report.to_json(design_json)
    design_md.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"\nReport JSON: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
