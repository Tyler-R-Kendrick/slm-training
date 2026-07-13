#!/usr/bin/env python3
"""Build versioned training-data artifacts (high-quality, deterministic)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.train_data import TrainDataConfig, build_train_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="all",
        choices=["rico", "fixture", "both", "awwwards", "rico+awwwards", "all"],
        help="Training data source (default: all).",
    )
    parser.add_argument(
        "--seed-path",
        type=Path,
        default=Path("fixtures/train_seeds.jsonl"),
        help="JSONL seed fixtures (used when source includes fixtures).",
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
    parser.add_argument("--version", default="v1")
    parser.add_argument(
        "--synthesizer",
        default="quality",
        choices=["quality", "template", "layout", "none", "noop", "off"],
        help="Deterministic synthesizer (default: quality = layout aug + templates).",
    )
    parser.add_argument(
        "--min-quality-score",
        type=float,
        default=0.55,
        help="Drop records below this quality score after validation.",
    )
    parser.add_argument(
        "--allow-missing-design-md",
        action="store_true",
        help="Do not require DESIGN.md on every kept record.",
    )
    parser.add_argument(
        "--max-openui-chars",
        type=int,
        default=None,
        help="Drop layouts longer than this many characters (compact core sets).",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=None,
        help="Drop layouts with more than this many component calls.",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Tag records with curriculum stages A/B/C and inject stress adversarial examples.",
    )
    parser.add_argument(
        "--namespace-augment",
        action="store_true",
        help="Emit namespace-augmented train variants (:acme.* re-prefix).",
    )
    args = parser.parse_args(argv)

    result = build_train_data(
        TrainDataConfig(
            seed_path=args.seed_path if args.source in {"fixture", "both", "all"} else None,
            rico_path=args.rico_path
            if args.source in {"rico", "both", "rico+awwwards", "all"}
            else None,
            source=args.source,
            output_root=args.output_root,
            version=args.version,
            synthesizer=args.synthesizer,
            rico_hf_split=args.rico_hf_split,
            rico_limit=args.rico_limit,
            max_children=args.max_children,
            min_quality_score=args.min_quality_score,
            require_design_md=not args.allow_missing_design_md,
            max_openui_chars=args.max_openui_chars,
            max_components=args.max_components,
            curriculum=args.curriculum,
            namespace_augment=args.namespace_augment,
        )
    )
    print(json.dumps(result["stats"], indent=2))
    print(f"wrote {result['output_dir']}")
    print(f"content_fingerprint={result['manifest'].get('content_fingerprint')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
