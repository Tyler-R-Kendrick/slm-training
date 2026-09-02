#!/usr/bin/env python3
"""Materialize the certified train bucket (root families hashed to 0-79).

The certified corpus ``openui_verified_v1`` carries every root family; its
manifest has no leakage fingerprints and training on it would expose the
validation/test families the certified eval suites are drawn from. This
script writes the train bucket as its own dataset (records + manifest with
``ids`` / ``split_group_ids`` / fingerprint lists, quality report, rejected
ledger, synthesis feedback) so ``build_test_data --train-manifest`` and the
leakage test consume it like any ``build_train_data`` snapshot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.test_data.certified import (
    CERTIFIED_CORPUS_DIR,
    CERTIFIED_TRAIN_BUCKET_ID,
    materialize_certified_train_bucket,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=CERTIFIED_CORPUS_DIR / "records.jsonl",
        help="records.jsonl of the certified corpus.",
    )
    parser.add_argument("--version", default=CERTIFIED_TRAIN_BUCKET_ID)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/data/train")
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish the built version into the immutable Git data store.",
    )
    args = parser.parse_args(argv)

    result = materialize_certified_train_bucket(
        args.output_root / args.version,
        dataset_id=args.version,
        corpus_path=args.corpus,
    )
    report = dict(result["quality_report"])
    report.pop("version_stamp", None)
    print(json.dumps(report, indent=2))
    print(f"wrote {result['output_dir']}")
    if args.publish:
        from slm_training.data.store import DataStore

        ref = DataStore().publish("train", args.version)
        print(f"published {ref.path} fingerprint={ref.fingerprint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
