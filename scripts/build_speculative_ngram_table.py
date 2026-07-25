#!/usr/bin/env python3
"""Build the deterministic speculative-ranking n-gram table (decode invariant I3).

The table orders already-legal completion paths so a confident branch point can
be resolved from the corpus instead of a neural forward. It never widens the
legal set — see `docs/design/decode-invariants.md`.

Train split only: the table is a decode-time prior, and building it from eval
records would leak the held-out suites into serving decisions.

    python -m scripts.build_speculative_ngram_table \
      --train-dir outputs/data/train/v1 --order 3 \
      --output outputs/decode/speculative_ngram_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from slm_training.dsl.grammar.fastpath.speculative_rank import build_ngram_table
from slm_training.dsl.schema import ExampleRecord, load_jsonl


def _train_records(train_dir: Path) -> list[ExampleRecord]:
    shards = sorted(train_dir.glob("*.jsonl"))
    if not shards:
        raise SystemExit(f"no .jsonl shards under {train_dir}")
    records: list[ExampleRecord] = []
    for shard in shards:
        records.extend(load_jsonl(shard))
    train_only = [record for record in records if record.split == "train"]
    if not train_only:
        raise SystemExit(f"{train_dir} contains no split='train' records")
    return train_only


def _encode(records: list[ExampleRecord], tokenizer_path: Path | None) -> list[list[int]]:
    from slm_training.models.tokenizer import OpenUITokenizer

    if tokenizer_path is not None:
        tokenizer = OpenUITokenizer.load(tokenizer_path)
    else:
        tokenizer = OpenUITokenizer.build([record.openui for record in records])
    return [
        list(tokenizer.encode(record.openui, add_special=False)) for record in records
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-dir",
        type=Path,
        required=True,
        help="Training data directory holding .jsonl shards (train split only).",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Checkpoint tokenizer json. Omit to build one from the same corpus.",
    )
    parser.add_argument("--order", type=int, default=3, help="Back-off n-gram order.")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination path for the content-addressed table artifact.",
    )
    args = parser.parse_args(argv)

    records = _train_records(args.train_dir)
    sequences = _encode(records, args.tokenizer)
    table = build_ngram_table(sequences, order=int(args.order))
    written = table.write(args.output)
    print(
        json.dumps(
            {
                "output": str(written),
                "order": table.order,
                "sequences": table.sequences,
                "tokens": table.tokens,
                "contexts": len(table.counts),
                "corpus_fingerprint": table.corpus_fingerprint,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
