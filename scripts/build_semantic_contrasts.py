#!/usr/bin/env python3
"""CLI to build the versioned semantic-contrast corpus (SPV2-01)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from slm_training.data.semantic_contrast import SemanticContrastBuilder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-id",
        default="semantic_contrast_v1",
        help="Versioned dataset identifier (default: semantic_contrast_v1).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/data"),
        help="Root directory for the eval dataset.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Deterministic generation seed."
    )
    parser.add_argument(
        "--source-count",
        type=int,
        default=12,
        help="Number of source ProgramSpecs to generate.",
    )
    parser.add_argument(
        "--splits",
        default="train,held_out",
        help="Comma-separated corpus splits (default: train,test).",
    )
    parser.add_argument(
        "--split-weights",
        default="0.8,0.2",
        help="Comma-separated split weights (default: 0.8,0.2).",
    )
    parser.add_argument(
        "--honesty-mode",
        default="production",
        choices=["production", "oracle_diagnostic"],
        help="Plan-compiler honesty mode.",
    )

    args = parser.parse_args(argv)
    splits = tuple(args.splits.split(","))
    weights = tuple(float(w) for w in args.split_weights.split(","))
    builder = SemanticContrastBuilder(
        output_root=args.output_root,
        dataset_id=args.dataset_id,
        seed=args.seed,
        source_count=args.source_count,
        splits=splits,
        split_weights=weights,
        honesty_mode=args.honesty_mode,
    )
    summary = builder.build()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
