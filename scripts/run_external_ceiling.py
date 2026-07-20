#!/usr/bin/env python3
"""Run the SLM-108 external constrained-decoding semantic ceiling matrix.

Example (fixture, no model download):
  python -m scripts.run_external_ceiling --mode fixture --output-dir outputs/runs/slm108_fixture

Example (frontier, requires GPU + pinned checkpoint + HF auth):
  python -m scripts.run_external_ceiling --mode frontier \
      --checkpoint-reference-uri hf://buckets/TKendrick/OpenUI/checkpoints/<run_id>/<ckpt>.ref.json \
      --output-dir outputs/runs/slm108_frontier
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from slm_training.harnesses.experiments.external_ceiling_matrix import (
    build_external_ceiling_manifest,
    render_markdown,
    run_fixture_matrix,
    validate_external_ceiling_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SLM-108 external constrained-decoding semantic ceiling matrix"
    )
    parser.add_argument(
        "--mode",
        choices=("fixture", "frontier"),
        default="fixture",
        help="fixture runs a torch-free wiring check; frontier loads real HF models",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/runs/slm108_external_ceiling"),
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
    if args.mode == "fixture":
        report = run_fixture_matrix(
            manifest,
            run_id="slm108_fixture",
            output_dir=args.output_dir,
        )
    else:
        print(
            "frontier mode requires GPU host and pinned checkpoints; "
            "leaving frontier arms not_run",
            file=sys.stderr,
        )
        report = run_fixture_matrix(
            manifest,
            run_id="slm108_frontier_partial",
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
