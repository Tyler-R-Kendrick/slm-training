#!/usr/bin/env python3
"""Run the SLM-109 E228 ≥100× training-exposure checkpoint ladder.

Example (plan only, no model load):
  python -m scripts.run_e228_exposure_ladder --plan-only \
      --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/e228-candidate-margin-matched/ref.json \
      --output-dir outputs/runs/slm109_e228_ladder

Example (fixture wiring check):
  python -m scripts.run_e228_exposure_ladder --fixture \
      --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/e228-candidate-margin-matched/ref.json \
      --output-dir outputs/runs/slm109_e228_fixture

Frontier execution (GPU + durable checkpoint required):
  python -m scripts.run_e228_exposure_ladder --frontier \
      --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/e228-candidate-margin-matched/ref.json \
      --checkpoint-bucket hf://buckets/TKendrick/OpenUI \
      --output-dir outputs/runs/slm109_e228_frontier
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from slm_training.harnesses.experiments.e228_exposure_ladder import (
    build_e228_exposure_ladder,
    render_markdown,
    run_fixture_ladder,
    validate_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-109 E228 ≥100× training-exposure checkpoint ladder"
    )
    parser.add_argument(
        "--mode",
        choices=("plan-only", "fixture", "frontier"),
        default="plan-only",
        help=(
            "plan-only emits the manifest without loading models; fixture runs a "
            "torch-free wiring check; frontier dispatches the real GPU ladder (not "
            "implemented in this wiring slice)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/slm109_e228_ladder"),
    )
    parser.add_argument(
        "--parent-checkpoint-uri",
        default=None,
        help="Durable URI of the E228 1× parent checkpoint",
    )
    parser.add_argument(
        "--checkpoint-bucket",
        default="hf://buckets/TKendrick/OpenUI",
        help="HF bucket for durable ladder checkpoints",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated seeds for the ladder",
    )
    parser.add_argument(
        "--multipliers",
        default="1,4,16,64,128",
        help="Comma-separated exposure multipliers",
    )
    args = parser.parse_args(argv)

    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    multipliers = tuple(int(m.strip()) for m in args.multipliers.split(",") if m.strip())

    manifest = build_e228_exposure_ladder(
        parent_checkpoint_uri=args.parent_checkpoint_uri,
        checkpoint_bucket=args.checkpoint_bucket,
        seeds=seeds,
        multipliers=multipliers,
    )
    errors = validate_manifest(manifest)
    if errors:
        for error in errors:
            print(f"manifest error: {error}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_json(args.output_dir / "e228_exposure_manifest.json")

    if args.mode == "plan-only":
        report = run_fixture_ladder(manifest, run_id="slm109_plan", output_dir=args.output_dir)
    elif args.mode == "fixture":
        report = run_fixture_ladder(manifest, run_id="slm109_fixture", output_dir=args.output_dir)
    else:
        print(
            "frontier mode requires GPU host and durable checkpoints; "
            "emitting fixture plan only",
            file=sys.stderr,
        )
        report = run_fixture_ladder(manifest, run_id="slm109_frontier_partial", output_dir=args.output_dir)

    markdown = render_markdown(report)
    (args.output_dir / "e228_exposure_report.md").write_text(markdown, encoding="utf-8")
    report.to_json(args.output_dir / "e228_exposure_report.json")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
