#!/usr/bin/env python3
"""Build versioned testing-data suites."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.test_data import TestDataConfig, build_test_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-path",
        type=Path,
        default=Path("fixtures/test_seeds.jsonl"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/test_data"),
    )
    parser.add_argument("--version", default="v0")
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=None,
        help="Optional train manifest.json for leakage checks",
    )
    parser.add_argument(
        "--suites",
        default="smoke,held_out,adversarial,ood",
        help="Comma-separated suite names",
    )
    args = parser.parse_args(argv)

    suites = tuple(s.strip() for s in args.suites.split(",") if s.strip())
    result = build_test_data(
        TestDataConfig(
            seed_path=args.seed_path,
            output_root=args.output_root,
            version=args.version,
            suites=suites,
            train_manifest=args.train_manifest,
        )
    )
    print(json.dumps(result["stats"], indent=2))
    print(f"wrote {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
