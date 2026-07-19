#!/usr/bin/env python3
"""Run the SLM-124 EFS3-03 B3 surface-vs-choice capacity ladder v2 plan/fixture.

Example (plan only, no model load):
  python -m scripts.run_b3_capacity_v2 --mode plan-only \
      --output-dir outputs/runs/slm124_b3_capacity_v2

Example (fixture wiring check):
  python -m scripts.run_b3_capacity_v2 --mode fixture \
      --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/e228-candidate-margin-matched/ref.json \
      --output-dir outputs/runs/slm124_b3_capacity_fixture

Frontier execution (GPU + durable checkpoints required):
  python -m scripts.run_b3_capacity_v2 --mode frontier \
      --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/e228-candidate-margin-matched/ref.json \
      --checkpoint-bucket hf://buckets/TKendrick/OpenUI \
      --output-dir outputs/runs/slm124_b3_capacity_frontier
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from slm_training.harnesses.experiments.b3_capacity_v2 import (
    CAPACITY_ARMS,
    CAPACITY_WIDTHS,
    build_b3_capacity_v2_manifest,
    render_markdown,
    run_fixture_ladder,
    validate_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="SLM-124 B3 capacity ladder v2")
    parser.add_argument(
        "--mode",
        choices=("plan-only", "fixture", "frontier"),
        default="plan-only",
        help=(
            "plan-only emits the manifest; fixture runs a torch-free wiring check; "
            "frontier dispatches the real GPU ladder (not implemented in this wiring slice)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/slm124_b3_capacity_v2"),
    )
    parser.add_argument(
        "--parent-checkpoint-uri",
        default=None,
        help="Durable URI of the EFS1-decided base checkpoint",
    )
    parser.add_argument(
        "--checkpoint-bucket",
        default="hf://buckets/TKendrick/OpenUI",
        help="HF bucket for durable ladder checkpoints",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated seeds for the rows",
    )
    parser.add_argument(
        "--widths",
        default=",".join(map(str, CAPACITY_WIDTHS)),
        help="Comma-separated d_model widths",
    )
    parser.add_argument(
        "--representations",
        default=",".join(CAPACITY_ARMS),
        help="Comma-separated representation arms",
    )
    args = parser.parse_args(argv)

    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    widths = tuple(int(w.strip()) for w in args.widths.split(",") if w.strip())
    representations = tuple(r.strip() for r in args.representations.split(",") if r.strip())

    manifest = build_b3_capacity_v2_manifest(
        parent_checkpoint_uri=args.parent_checkpoint_uri,
        checkpoint_bucket=args.checkpoint_bucket,
        widths=widths,
        seeds=seeds,
        representations=representations,
    )
    errors = validate_manifest(manifest)
    if errors:
        for error in errors:
            print(f"manifest error: {error}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_json(args.output_dir / "b3_capacity_v2_manifest.json")

    if args.mode == "plan-only":
        report = run_fixture_ladder(
            manifest, run_id="slm124_plan", output_dir=args.output_dir
        )
    elif args.mode == "fixture":
        report = run_fixture_ladder(
            manifest, run_id="slm124_fixture", output_dir=args.output_dir
        )
    else:
        print(
            "frontier mode requires GPU host and durable checkpoints; "
            "emitting fixture plan only",
            file=sys.stderr,
        )
        report = run_fixture_ladder(
            manifest, run_id="slm124_frontier_partial", output_dir=args.output_dir
        )

    markdown = render_markdown(report)
    (args.output_dir / "b3_capacity_v2_report.md").write_text(
        markdown, encoding="utf-8"
    )
    report.to_json(args.output_dir / "b3_capacity_v2_report.json")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
