#!/usr/bin/env python3
"""Build versioned training-data artifacts (RICO by default)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.train_data import TrainDataConfig, build_train_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="rico",
        choices=["rico", "fixture", "both"],
        help="Training data source (default: rico).",
    )
    parser.add_argument(
        "--seed-path",
        type=Path,
        default=Path("fixtures/train_seeds.jsonl"),
        help="JSONL seed fixtures (used when source is fixture/both).",
    )
    parser.add_argument(
        "--rico-path",
        type=Path,
        default=Path("fixtures/rico/semantic_train.jsonl"),
        help="Local RICO semantic JSONL (HF-exported screens).",
    )
    parser.add_argument(
        "--rico-hf-split",
        default=None,
        help="Optional live Hugging Face RICO split (train/validation/test).",
    )
    parser.add_argument("--rico-limit", type=int, default=None)
    parser.add_argument("--max-children", type=int, default=6)
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
            seed_path=args.seed_path if args.source in {"fixture", "both"} else None,
            rico_path=args.rico_path if args.source in {"rico", "both"} else None,
            source=args.source,
            output_root=args.output_root,
            version=args.version,
            synthesizer=args.synthesizer,
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
