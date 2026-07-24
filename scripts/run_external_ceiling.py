#!/usr/bin/env python3
"""Run the SLM-108 external constrained-decoding semantic ceiling matrix.

Example (fixture, no model download):
  python -m scripts.run_external_ceiling --mode fixture --output-dir outputs/runs/slm108_fixture

Example (SLM-294 frontier arm chunk, resumable under the run cap):
  python -m scripts.run_external_ceiling --mode frontier --arm B \
      --chunk-size 5 --output-dir outputs/runs/slm294_external_ceiling

Example (score all arm raw outputs into scoreboard.json):
  python -m scripts.run_external_ceiling --mode score \
      --output-dir outputs/runs/slm294_external_ceiling
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from slm_training.harnesses.experiments.external_ceiling_matrix import (
    DEFAULT_FROZEN_REQUESTS_FILE,
    DEFAULT_FROZEN_REQUESTS_SHA256,
    FRONTIER_ARMS,
    build_external_ceiling_manifest,
    render_markdown,
    run_fixture_matrix,
    run_frontier_chunk,
    run_score_pass,
    validate_external_ceiling_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-108 external constrained-decoding semantic ceiling matrix"
    )
    parser.add_argument(
        "--mode",
        choices=("fixture", "frontier", "score"),
        default="fixture",
        help=(
            "fixture runs a torch-free wiring check; frontier generates raw "
            "outputs for one arm in resumable chunks; score (re)scores all "
            "arm raw JSONLs into scoreboard.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/slm108_external_ceiling"),
    )
    parser.add_argument(
        "--requests-file",
        type=Path,
        default=DEFAULT_FROZEN_REQUESTS_FILE,
        help="Frozen SLM-294 requests JSONL (sha256 verified before use)",
    )
    parser.add_argument(
        "--requests-sha256",
        default=None,
        help="Expected sha256 of --requests-file (defaults to the pinned freeze)",
    )
    parser.add_argument(
        "--arm",
        choices=tuple(sorted(FRONTIER_ARMS)),
        default=None,
        help="Frontier arm to execute (B: SmolLM2-135M, C: Qwen2.5-7B)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help="Requests processed per frontier invocation (run-cap chunking)",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Alias for --mode score: (re)score existing raw JSONLs only",
    )
    parser.add_argument(
        "--checkpoint-reference-uri",
        default=None,
        help="Durable checkpoint reference for the tiny-SLM baseline arm A",
    )
    parser.add_argument(
        "--tiny-slm-run-id",
        default=None,
        help="Run id of the tiny-SLM baseline to compare against",
    )
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Print resolved manifest without running",
    )
    args = parser.parse_args(argv)

    expected_sha = args.requests_sha256 or DEFAULT_FROZEN_REQUESTS_SHA256

    if args.mode == "score" or args.score_only:
        try:
            scoreboard = run_score_pass(
                output_dir=args.output_dir,
                requests_file=args.requests_file,
                expected_sha256=expected_sha,
            )
        except ValueError as exc:
            print(f"score error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(scoreboard["arms"], indent=2, sort_keys=True))
        return 0

    if args.mode == "frontier":
        if args.arm is None:
            print("frontier mode requires --arm {B,C}", file=sys.stderr)
            return 1
        try:
            summary = run_frontier_chunk(
                arm=args.arm,
                output_dir=args.output_dir,
                requests_file=args.requests_file,
                expected_sha256=expected_sha,
                chunk_size=args.chunk_size,
            )
        except ValueError as exc:
            print(f"frontier error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    manifest = build_external_ceiling_manifest(
        tiny_slm_run_id=args.tiny_slm_run_id,
        checkpoint_reference_uri=args.checkpoint_reference_uri,
    )
    errors = validate_external_ceiling_manifest(manifest)
    if errors:
        for error in errors:
            print(f"manifest error: {error}", file=sys.stderr)
        return 1

    if args.describe:
        print(manifest.to_dict())
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = run_fixture_matrix(
        manifest,
        run_id="slm108_fixture",
        output_dir=args.output_dir,
    )

    markdown = render_markdown(report)
    (args.output_dir / "external_ceiling_report.md").write_text(
        markdown, encoding="utf-8"
    )
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
