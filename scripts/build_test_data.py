#!/usr/bin/env python3
"""Build versioned testing-data suites (disjoint from train)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.test_data import TestDataConfig, build_test_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="both",
        choices=["rico", "fixture", "both"],
        help="Test data source (default: both RICO eval split + fixtures).",
    )
    parser.add_argument(
        "--seed-path",
        type=Path,
        default=Path("fixtures/test_seeds.jsonl"),
    )
    parser.add_argument(
        "--rico-path",
        type=Path,
        default=Path("fixtures/rico/semantic_test.jsonl"),
    )
    parser.add_argument(
        "--rico-hf-split",
        default=None,
        help="Optional live Hugging Face RICO split for eval screens.",
    )
    parser.add_argument("--rico-limit", type=int, default=None)
    parser.add_argument("--max-children", type=int, default=6)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/test_data"),
    )
    parser.add_argument("--version", default="v1")
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=Path("outputs/train_data/v1/manifest.json"),
        help="Train manifest used for leakage checks (required).",
    )
    parser.add_argument(
        "--allow-without-train-manifest",
        action="store_true",
        help="Escape hatch for unit tests only; do not use for real builds.",
    )
    parser.add_argument(
        "--suites",
        default="smoke,held_out,adversarial,ood,rico_held",
        help="Comma-separated suite names",
    )
    args = parser.parse_args(argv)

    suites = tuple(s.strip() for s in args.suites.split(",") if s.strip())
    train_manifest = None if args.allow_without_train_manifest else args.train_manifest
    result = build_test_data(
        TestDataConfig(
            seed_path=args.seed_path if args.source in {"fixture", "both"} else None,
            rico_path=args.rico_path if args.source in {"rico", "both"} else None,
            source=args.source,
            output_root=args.output_root,
            version=args.version,
            suites=suites,
            train_manifest=train_manifest,
            require_train_manifest=not args.allow_without_train_manifest,
            rico_hf_split=args.rico_hf_split,
            rico_limit=args.rico_limit,
            max_children=args.max_children,
        )
    )
    print(json.dumps(result["stats"], indent=2))
    print(f"wrote {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
