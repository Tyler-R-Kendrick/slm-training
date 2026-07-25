#!/usr/bin/env python3
"""Run the SLM-300 AP-015 self-context exposure-bias curriculum.

Example (plan only, no model load):
  python -m scripts.run_self_context_curriculum --mode plan-only \
      --output-dir outputs/runs/slm300_self_context_curriculum

Example (fixture wiring check):
  python -m scripts.run_self_context_curriculum --mode fixture \
      --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/certified-baseline/ref.json \
      --output-dir outputs/runs/slm300_self_context_fixture

Frontier execution (GPU + durable checkpoint required):
  python -m scripts.run_self_context_curriculum --mode frontier \
      --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/certified-baseline/ref.json \
      --checkpoint-bucket hf://buckets/TKendrick/OpenUI \
      --output-dir outputs/runs/slm300_self_context_frontier
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from slm_training.harnesses.experiments.self_context_curriculum import (
    build_self_context_manifest,
    render_markdown,
    run_fixture_self_context_curriculum,
    validate_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-300 AP-015 self-context exposure-bias curriculum"
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
        default=Path("outputs/runs/slm300_self_context_curriculum"),
    )
    parser.add_argument(
        "--parent-checkpoint-uri",
        default=None,
        help="Durable URI of the certified-baseline parent checkpoint",
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
        "--self-context-rates",
        default="0.0,0.10,0.25,0.50",
        help="Comma-separated self-context mixture rates",
    )
    parser.add_argument(
        "--lagged-policy-share",
        type=float,
        default=0.5,
        help="Share of self-context mass drawn from the lagged/EMA policy vs current policy",
    )
    args = parser.parse_args(argv)

    try:
        seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())
        rates = tuple(
            float(x.strip()) for x in args.self_context_rates.split(",") if x.strip()
        )
    except ValueError as error:
        print(f"argument error: {error}", file=sys.stderr)
        return 2

    manifest = build_self_context_manifest(
        parent_checkpoint_uri=args.parent_checkpoint_uri,
        checkpoint_bucket=args.checkpoint_bucket,
        seeds=seeds,
        self_context_rates=rates,
        lagged_policy_share=args.lagged_policy_share,
    )
    errors = validate_manifest(manifest)
    if errors:
        for error in errors:
            print(f"manifest error: {error}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_json(args.output_dir / "self_context_curriculum_manifest.json")

    exit_code = 0
    if args.mode == "plan-only":
        report = run_fixture_self_context_curriculum(
            manifest, run_id="slm300_plan", output_dir=args.output_dir
        )
    elif args.mode == "fixture":
        report = run_fixture_self_context_curriculum(
            manifest, run_id="slm300_fixture", output_dir=args.output_dir
        )
    else:
        print(
            "frontier mode requires GPU host and durable checkpoints; "
            "emitting fixture plan only (not a completed frontier run)",
            file=sys.stderr,
        )
        report = run_fixture_self_context_curriculum(
            manifest, run_id="slm300_frontier_partial", output_dir=args.output_dir
        )
        # A fixture fallback must never be mistaken for completed frontier
        # evidence by a scheduler that only checks the exit code.
        exit_code = 2

    markdown = render_markdown(report)
    (args.output_dir / "self_context_curriculum_report.md").write_text(
        markdown, encoding="utf-8"
    )
    report.to_json(args.output_dir / "self_context_curriculum_report.json")
    print(markdown)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
