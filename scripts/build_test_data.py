#!/usr/bin/env python3
"""Build versioned testing-data suites (disjoint from train)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.levers import DEFAULT_TRAIN_DATA_DIR

from slm_training.harnesses.test_data import TestDataConfig, build_test_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        default="both",
        choices=["rico", "fixture", "both", "certified"],
        help=(
            "Test data source (default: both RICO eval split + fixtures). "
            "'certified' samples the validation/test root-family buckets of "
            "the certified corpus (harnesses.test_data.certified)."
        ),
    )
    parser.add_argument(
        "--certified-corpus",
        type=Path,
        default=None,
        help="records.jsonl of the certified corpus (default: openui_verified_v1).",
    )
    parser.add_argument("--certified-smoke-n", type=int, default=96)
    parser.add_argument("--certified-held-out-n", type=int, default=24)
    parser.add_argument("--certified-seed", type=int, default=0)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish the built version into the immutable Git data store.",
    )
    parser.add_argument(
        "--seed-path",
        type=Path,
        default=Path("src/slm_training/resources/test_seeds.jsonl"),
    )
    parser.add_argument(
        "--rico-path",
        type=Path,
        default=Path("src/slm_training/resources/rico/semantic_test.jsonl"),
    )
    parser.add_argument(
        "--rico-hf-split",
        default=None,
        help="Optional live Hugging Face RICO split for eval screens.",
    )
    parser.add_argument("--rico-limit", type=int, default=None)
    parser.add_argument(
        "--rico-hf-cache",
        type=Path,
        default=Path("src/slm_training/resources/rico/hf_test_cache.jsonl"),
        help="Cache file for HF RICO screens (speeds rebuilds / offline).",
    )
    parser.add_argument(
        "--target-records",
        type=int,
        default=None,
        help="Keep at least this many additional RICO samples after leakage filters.",
    )
    parser.add_argument("--max-children", type=int, default=6)
    parser.add_argument(
        "--sanitize-mode",
        choices=["off", "audit", "enforce"],
        default="enforce",
        help=(
            "Deterministic target sanitization shared with the train build "
            "(eval gold matches the sanitized train distribution). Default: "
            "enforce."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/data/eval"),
    )
    parser.add_argument("--version", default="v1")
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=DEFAULT_TRAIN_DATA_DIR / "manifest.json",
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
    parser.add_argument(
        "--no-rico-path",
        action="store_true",
        help="Do not load local rico fixture JSONL (HF-only when --rico-hf-split set).",
    )
    parser.add_argument(
        "--register-lineage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Register the built eval dataset as a lineage DataSnapshot (idempotent).",
    )
    parser.add_argument(
        "--lineage-root",
        type=Path,
        default=Path("outputs/lineage"),
    )
    args = parser.parse_args(argv)

    suites = tuple(s.strip() for s in args.suites.split(",") if s.strip())
    train_manifest = None if args.allow_without_train_manifest else args.train_manifest
    rico_path = None
    if args.source in {"rico", "both"} and not args.no_rico_path:
        rico_path = args.rico_path
    result = build_test_data(
        TestDataConfig(
            seed_path=args.seed_path if args.source in {"fixture", "both"} else None,
            rico_path=rico_path,
            source=args.source,
            output_root=args.output_root,
            version=args.version,
            suites=suites,
            train_manifest=train_manifest,
            require_train_manifest=not args.allow_without_train_manifest,
            rico_hf_split=args.rico_hf_split,
            rico_limit=args.rico_limit,
            rico_hf_cache_path=args.rico_hf_cache,
            target_records=args.target_records,
            max_children=args.max_children,
            sanitize_mode=args.sanitize_mode,
            certified_corpus=args.certified_corpus,
            certified_smoke_n=args.certified_smoke_n,
            certified_held_out_n=args.certified_held_out_n,
            certified_seed=args.certified_seed,
        )
    )
    print(json.dumps(result["stats"], indent=2))
    print(f"wrote {result['output_dir']}")
    if args.source == "certified":
        from slm_training.autoresearch.screening_sample_size import (
            assert_eval_publish_target_writable,
        )
        from slm_training.harnesses.test_data.certified import (
            write_certified_eval_sidecars,
        )

        assert_eval_publish_target_writable(args.version)
        sidecars = write_certified_eval_sidecars(
            result["output_dir"], eval_version=args.version, stats=result["stats"]
        )
        for name, path in sidecars.items():
            print(f"{name}={path}")
    if args.publish:
        from slm_training.data.store import DataStore

        ref = DataStore().publish("eval", args.version)
        print(f"published {ref.path} fingerprint={ref.fingerprint}")
    if args.register_lineage:
        from slm_training.lineage.data_cycle import register_dataset_snapshot
        from slm_training.lineage.store import LineageStore

        snapshot, snapshot_path, created = register_dataset_snapshot(
            LineageStore(args.lineage_root),
            dataset_dir=Path(result["output_dir"]),
            kind="eval",
        )
        state = "registered" if created else "already-registered"
        print(f"lineage_snapshot={snapshot.sha} ({state}: {snapshot_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
