#!/usr/bin/env python3
"""Build the deterministic speculative-ranking n-gram table (decode invariant I3).

The table orders already-legal completion paths so a confident branch point can
be resolved from the corpus instead of a neural forward. It never widens the
legal set — see `docs/design/decode-invariants.md`.

Train split only: the table is a decode-time prior, and building it from eval
records would leak the held-out suites into serving decisions. Targets are
templatized first, so the table is keyed on symbols and placeholders and never
on free-form string content (symbol-only output contract).

Default input is the immutable certified corpus, which is what the committed
artifact is built from:

    python -m scripts.build_speculative_ngram_table \
      --output src/slm_training/resources/decode/speculative_ngram_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from slm_training.dsl.analysis.templatize import templatize
from slm_training.dsl.grammar.fastpath.speculative_rank import (
    COMMITTED_NGRAM_TABLE,
    build_ngram_table,
)
from slm_training.dsl.schema import ExampleRecord, load_jsonl

ROOT = Path(__file__).resolve().parents[1]
CERTIFIED_CORPUS = (
    ROOT / "src/slm_training/resources/data/train/openui_verified_v1/records.jsonl"
)


def _records(source: Path) -> list[ExampleRecord]:
    if source.is_dir():
        shards = sorted(source.glob("*.jsonl"))
        if not shards:
            raise SystemExit(f"no .jsonl shards under {source}")
        records: list[ExampleRecord] = []
        for shard in shards:
            records.extend(load_jsonl(shard))
    else:
        records = load_jsonl(source)
    train_only = [record for record in records if record.split in {"train", ""}]
    if not train_only:
        raise SystemExit(f"{source} contains no train-split records")
    return train_only


def _encode(records: list[ExampleRecord], codec: str) -> tuple[list[list[int]], int]:
    """Encode templatized targets in the decoder codec the ranker will see."""
    if codec != "lexer":
        raise SystemExit(
            f"unsupported codec {codec!r}; the compiler completion domain the "
            "ranker scores is DSL-native ('lexer')"
        )
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer, SymbolTable

    tokenizer = DSLNativeTokenizer.build()
    sequences: list[list[int]] = []
    skipped = 0
    for record in records:
        try:
            target = templatize(record.openui)
            table = SymbolTable.from_placeholders(
                target.placeholders, max_slots=tokenizer.sym_slots
            )
            ids = list(tokenizer.encode(target.source, table=table, add_special=False))
        except Exception:  # noqa: BLE001 - a record the codec rejects is not a prior
            skipped += 1
            continue
        if tokenizer.unk_id in ids:
            skipped += 1
            continue
        sequences.append(ids)
    if not sequences:
        raise SystemExit("no records survived templatization and encoding")
    return sequences, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        default=CERTIFIED_CORPUS,
        help="Train records .jsonl (or a directory of shards). Default: the "
        "immutable certified corpus.",
    )
    parser.add_argument(
        "--codec",
        default="lexer",
        help="Decoder codec the table is keyed on (only 'lexer' is supported).",
    )
    parser.add_argument("--order", type=int, default=3, help="Back-off n-gram order.")
    parser.add_argument(
        "--output",
        type=Path,
        default=COMMITTED_NGRAM_TABLE,
        help="Destination for the content-addressed table artifact.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Rebuild and compare against the committed artifact instead of "
        "writing it. Non-zero exit on drift.",
    )
    args = parser.parse_args(argv)

    records = _records(args.records)
    sequences, skipped = _encode(records, str(args.codec))
    table = build_ngram_table(sequences, order=int(args.order))
    summary = {
        "codec": args.codec,
        "order": table.order,
        "sequences": table.sequences,
        "tokens": table.tokens,
        "contexts": len(table.counts),
        "skipped_records": skipped,
        "corpus_fingerprint": table.corpus_fingerprint,
    }

    if args.check:
        if not args.output.is_file():
            print(f"missing committed table: {args.output}", file=sys.stderr)
            return 1
        committed = json.loads(args.output.read_text(encoding="utf-8"))
        if committed != table.to_dict():
            print(
                f"committed table {args.output} is stale; rebuild with "
                "`python -m scripts.build_speculative_ngram_table`",
                file=sys.stderr,
            )
            return 1
        print(json.dumps({**summary, "check": "ok"}, indent=2, sort_keys=True))
        return 0

    written = table.write(args.output)
    print(json.dumps({**summary, "output": str(written)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
