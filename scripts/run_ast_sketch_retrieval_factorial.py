#!/usr/bin/env python3
"""Run the SLM-133 EFS3-06 AST-sketch × choice-native retrieval factorial plan/fixture.

Examples:
  # Plan only (CPU, no model load)
  python -m scripts.run_ast_sketch_retrieval_factorial --mode plan-only \
      --output-dir outputs/runs/slm133_ast_sketch_retrieval

  # Fixture wiring check
  python -m scripts.run_ast_sketch_retrieval_factorial --mode fixture \
      --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/e228-candidate-margin-matched/ref.json \
      --output-dir outputs/runs/slm133_ast_sketch_retrieval_fixture

Frontier execution (GPU + durable checkpoints required):
  python -m scripts.run_ast_sketch_retrieval_factorial --mode frontier \
      --parent-checkpoint-uri hf://buckets/TKendrick/OpenUI/checkpoints/e228-candidate-margin-matched/ref.json \
      --checkpoint-bucket hf://buckets/TKendrick/OpenUI \
      --output-dir outputs/runs/slm133_ast_sketch_retrieval_frontier
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from slm_training.harnesses.experiments.ast_sketch_retrieval_factorial import (
    build_ast_sketch_retrieval_manifest,
    render_markdown,
    run_fixture_matrix,
    validate_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-133 AST-sketch × choice-native retrieval factorial"
    )
    parser.add_argument(
        "--mode",
        choices=("plan-only", "fixture", "frontier"),
        default="plan-only",
        help=(
            "plan-only emits the manifest; fixture runs a torch-free wiring check; "
            "frontier dispatches the real GPU factorial (not implemented in this wiring slice)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/slm133_ast_sketch_retrieval"),
    )
    parser.add_argument(
        "--parent-checkpoint-uri",
        default=None,
        help="Durable URI of the EFS1-decided base checkpoint",
    )
    parser.add_argument(
        "--checkpoint-bucket",
        default="hf://buckets/TKendrick/OpenUI",
        help="HF bucket for durable factorial checkpoints",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated seeds for the rows",
    )
    parser.add_argument(
        "--include-controls",
        action="store_true",
        help="Include random_choice and surface_skeleton control arms",
    )
    args = parser.parse_args(argv)

    seeds = tuple(int(s.strip()) for s in args.seeds.split(",") if s.strip())

    manifest = build_ast_sketch_retrieval_manifest(
        parent_checkpoint_uri=args.parent_checkpoint_uri,
        checkpoint_bucket=args.checkpoint_bucket,
        seeds=seeds,
        include_controls=args.include_controls,
    )
    errors = validate_manifest(manifest)
    if errors:
        for error in errors:
            print(f"manifest error: {error}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_json(args.output_dir / "ast_sketch_retrieval_manifest.json")

    if args.mode == "plan-only":
        report = run_fixture_matrix(manifest, run_id="slm133_plan", output_dir=args.output_dir)
    elif args.mode == "fixture":
        report = run_fixture_matrix(manifest, run_id="slm133_fixture", output_dir=args.output_dir)
    else:
        print(
            "frontier mode requires GPU host and durable checkpoints; "
            "emitting fixture plan only",
            file=sys.stderr,
        )
        report = run_fixture_matrix(
            manifest, run_id="slm133_frontier_partial", output_dir=args.output_dir
        )

    markdown = render_markdown(report)
    (args.output_dir / "ast_sketch_retrieval_report.md").write_text(
        markdown, encoding="utf-8"
    )
    report.to_json(args.output_dir / "ast_sketch_retrieval_report.json")
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
