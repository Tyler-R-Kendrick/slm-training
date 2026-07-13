#!/usr/bin/env python3
"""Build versioned training-data artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.train_data import TrainDataConfig, build_train_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-path",
        type=Path,
        default=Path("fixtures/train_seeds.jsonl"),
        help="JSONL seed fixtures",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/train_data"),
    )
    parser.add_argument("--version", default="v0")
    parser.add_argument(
        "--synthesizer",
        default="template",
        choices=["template", "none", "noop", "off"],
    )
    args = parser.parse_args(argv)

    result = build_train_data(
        TrainDataConfig(
            seed_path=args.seed_path,
            output_root=args.output_root,
            version=args.version,
            synthesizer=args.synthesizer,
        )
    )
    print(json.dumps(result["stats"], indent=2))
    print(f"wrote {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
