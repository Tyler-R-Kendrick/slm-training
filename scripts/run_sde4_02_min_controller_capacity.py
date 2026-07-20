#!/usr/bin/env python3
"""Run the SLM-180 SDE4-02 minimum-controller-capacity plan/fixture.

Example (plan only, no model load):
  python -m scripts.run_sde4_02_min_controller_capacity --mode plan-only

Example (fixture wiring check):
  python -m scripts.run_sde4_02_min_controller_capacity --mode fixture
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from slm_training.harnesses.experiments.sde4_02_min_controller_capacity import (
    ControllerCapacityManifest,
    ControllerCapacityReport,
    ControllerCapacityRow,
    build_manifest,
    render_markdown,
    run_fixture_ladder,
)
from slm_training.versioning import build_version_stamp


def _today_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def _plan_only_report(manifest: ControllerCapacityManifest) -> ControllerCapacityReport:
    """Return a torch-free plan-only report skeleton."""
    rows: list[ControllerCapacityRow] = []
    for rung in manifest.rungs:
        for seed in manifest.seeds:
            rows.append(
                ControllerCapacityRow(
                    rung_id=rung.rung_id,
                    hidden_dim=rung.hidden_dim,
                    seed=seed,
                    train_accuracy=0.0,
                    val_accuracy=0.0,
                    trainable_parameters=0,
                    active_parameters=0,
                    meets_competence_target=False,
                    status="plan_only",
                    notes=["plan-only: no model trained"],
                )
            )
    return ControllerCapacityReport(
        matrix_set=manifest.matrix_set,
        matrix_version=manifest.matrix_version,
        run_id="sde4_02_plan",
        status="plan_only",
        manifest=replace(manifest, status="plan_only"),
        rows=rows,
        selected_rung_id=None,
        capacity_threshold_not_identifiable=True,
        version_stamp=build_version_stamp(
            "harness.experiments",
            "harness.experiments.sde4_02_min_controller_capacity",
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-180 SDE4-02 minimum controller capacity fixture"
    )
    parser.add_argument(
        "--mode",
        choices=("plan-only", "fixture"),
        default="plan-only",
        help=(
            "plan-only emits the manifest and report skeleton; "
            "fixture trains tiny CPU MLPs and emits a full report"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"outputs/runs/sde4-02-min-controller-{_today_slug()}"),
    )
    parser.add_argument(
        "--rungs",
        type=int,
        default=5,
        help="Number of capacity rungs (default 5)",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated seeds for the rows",
    )
    parser.add_argument(
        "--hidden-dims",
        default=",".join(map(str, [8, 16, 32, 64, 128])),
        help="Comma-separated hidden dimensions, one per rung",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=200,
        help="Fixture training steps per (rung, seed)",
    )
    args = parser.parse_args(argv)

    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    hidden_dims = tuple(
        int(h.strip()) for h in args.hidden_dims.split(",") if h.strip()
    )

    manifest = build_manifest(
        rungs=args.rungs,
        seeds=seeds,
        hidden_dims=hidden_dims,
    )

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    design_json = Path(
        f"docs/design/iter-sde4-02-min-controller-capacity-{_today_slug()}.json"
    )
    design_md = Path(
        f"docs/design/iter-sde4-02-min-controller-capacity-{_today_slug()}.md"
    )

    if args.mode == "plan-only":
        manifest = replace(manifest, status="plan_only")
        report = _plan_only_report(manifest)
        manifest.to_json(output_dir / "sde4_02_min_controller_capacity_manifest.json")
    else:
        report = run_fixture_ladder(
            manifest,
            run_id="sde4_02_fixture",
            output_dir=output_dir,
            train_steps=args.train_steps,
        )

    markdown = render_markdown(report)

    report_path = output_dir / "sde4_02_min_controller_capacity_report.json"
    report.to_json(report_path)
    (output_dir / "sde4_02_min_controller_capacity_report.md").write_text(
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
