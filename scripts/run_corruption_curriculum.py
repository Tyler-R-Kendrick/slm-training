#!/usr/bin/env python3
"""Run the SLM-120 EFS3-02 near-solved semantic corruption curriculum.

Example (plan only, no model load):
  python -m scripts.run_corruption_curriculum --mode plan-only \
      --output-dir outputs/runs/slm120_corruption_curriculum

Example (fixture wiring check):
  python -m scripts.run_corruption_curriculum --mode fixture \
      --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/e228-candidate-margin-matched/ref.json \
      --output-dir outputs/runs/slm120_corruption_fixture

Frontier execution (GPU + durable checkpoint required):
  python -m scripts.run_corruption_curriculum --mode frontier \
      --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/e228-candidate-margin-matched/ref.json \
      --checkpoint-bucket hf://buckets/TKendrick/OpenUI \
      --output-dir outputs/runs/slm120_corruption_frontier
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from slm_training.harnesses.experiments.corruption_curriculum import (
    build_corruption_curriculum_manifest,
    render_markdown,
    run_fixture_curriculum,
    validate_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-120 near-solved semantic corruption curriculum"
    )
    parser.add_argument(
        "--mode",
        choices=("plan-only", "fixture", "frontier"),
        default="plan-only",
        help=(
            "plan-only emits the manifest without loading models; fixture runs a "
            "torch-free wiring check; frontier dispatches the real GPU curriculum (not "
            "implemented in this wiring slice)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/slm120_corruption_curriculum"),
    )
    parser.add_argument(
        "--parent-checkpoint-uri",
        default=None,
        help="Durable URI of the EFS1-decided parent checkpoint",
    )
    parser.add_argument(
        "--checkpoint-bucket",
        default="hf://buckets/TKendrick/OpenUI",
        help="HF bucket for durable curriculum checkpoints",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated seeds for the curriculum arms",
    )
    parser.add_argument(
        "--near-solved-shares",
        default="0.0,0.05,0.10,0.15,0.30",
        help="Comma-separated near-solved (S1+S2) shares for arms A–E",
    )
    args = parser.parse_args(argv)

    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
    shares = tuple(
        float(x.strip()) for x in args.near_solved_shares.split(",") if x.strip()
    )

    manifest = build_corruption_curriculum_manifest(
        parent_checkpoint_uri=args.parent_checkpoint_uri,
        checkpoint_bucket=args.checkpoint_bucket,
        seeds=seeds,
        near_solved_shares=shares,
    )
    errors = validate_manifest(manifest)
    if errors:
        for error in errors:
            print(f"manifest error: {error}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_json(args.output_dir / "corruption_curriculum_manifest.json")

    if args.mode == "plan-only":
        report = run_fixture_curriculum(
            manifest, run_id="slm120_plan", output_dir=args.output_dir
        )
    elif args.mode == "fixture":
        report = run_fixture_curriculum(
            manifest, run_id="slm120_fixture", output_dir=args.output_dir
        )
    else:
        print(
            "frontier mode requires GPU host and durable checkpoints; "
            "emitting fixture plan only",
            file=sys.stderr,
        )
        report = run_fixture_curriculum(
            manifest, run_id="slm120_frontier_partial", output_dir=args.output_dir
        )

    markdown = render_markdown(report)
    (args.output_dir / "corruption_curriculum_report.md").write_text(
        markdown, encoding="utf-8"
    )
    report.to_json(args.output_dir / "corruption_curriculum_report.json")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
