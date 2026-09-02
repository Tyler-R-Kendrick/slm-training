#!/usr/bin/env python3
"""Build the deterministic speculative-ranking n-gram table (decode invariant I3).

The table orders already-legal completion paths so a confident branch point can
be resolved from the corpus instead of a neural forward. It never widens the
legal set — see `docs/design/decode-invariants.md`.

Train split only: the table is a decode-time prior, and building it from eval
records would leak the held-out suites into serving decisions. The default
input is therefore the certified TRAIN bucket of the immutable corpus,
``openui_verified_train_v2`` — root-family buckets 0–79 of
``openui_verified_v1`` under ``RootFamilySplitPolicyV1``, decontaminated from
the validation / test buckets by its own build (``scripts/build_certified_train_bucket.py``).
The split is taken from that bucket's manifest, never re-derived here; on top
of it the builder still refuses eval manifests / eval-split records and keeps
only records whose declared ``split`` is ``train`` (or unset).

Targets are templatized first, so the table is keyed on symbols and
placeholders and never on free-form string content (symbol-only output
contract).

    python -m scripts.build_speculative_ngram_table \
      --output src/slm_training/resources/decode/speculative_ngram_v1.json

The artifact records its provenance under ``source``: the bucket's
``dataset_id``, the manifest ``content_fingerprint``, and the sha256 of the
``records.jsonl`` it was built from. ``--check`` rebuilds from the same
defaults and fails when (a) the bucket on disk no longer matches the sha256
its manifest certifies, (b) the committed artifact's ``source`` block does not
match the live manifest (the artifact was built from some other corpus than
the one the docs claim), (c) the committed ``corpus_fingerprint`` is not the
fingerprint of the rebuilt train sequences, or (d) any other field drifts from
the rebuild.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from slm_training.dsl.analysis.templatize import templatize
from slm_training.dsl.grammar.fastpath.speculative_rank import (
    COMMITTED_NGRAM_TABLE,
    build_ngram_table,
)
from slm_training.dsl.schema import ExampleRecord, load_jsonl

ROOT = Path(__file__).resolve().parents[1]
CERTIFIED_TRAIN_BUCKET = (
    ROOT
    / "src/slm_training/resources/data/train/openui_verified_train_v2/records.jsonl"
)
DEFAULT_ORDER = 3
_EVAL_SPLITS = frozenset({"held_out", "smoke", "adversarial", "ood", "rico_held"})


def _manifest_path(source: Path) -> Path | None:
    candidates = []
    if source.is_dir():
        candidates.append(source / "manifest.json")
    else:
        candidates.append(source.parent / "manifest.json")
        if source.parent.name == "suites" or source.parent.parent.name == "suites":
            candidates.append(source.parents[2] / "manifest.json")
            candidates.append(source.parents[3] / "manifest.json")
    for path in candidates:
        if path.is_file():
            return path
    return None


def _manifest(source: Path) -> dict[str, Any]:
    path = _manifest_path(source)
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _manifest_kind(source: Path) -> str | None:
    kind = _manifest(source).get("kind")
    return str(kind) if kind else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _corpus_provenance(source: Path) -> dict[str, Any]:
    """Identify the corpus the table is built from and prove the records file
    is the one its manifest certified (sha256) whenever the manifest lists it.

    The returned block is written into the artifact as ``source`` and is what
    ``--check`` compares against the live manifest, so every value here must be
    a deterministic function of the corpus (no timestamps, no absolute paths).
    """
    payload = _manifest(source)
    certified = None
    if source.is_file():
        for artifact in payload.get("artifacts") or ():
            if isinstance(artifact, dict) and artifact.get("path") == source.name:
                certified = str(artifact.get("sha256") or "") or None
                break
    actual = _sha256(source) if source.is_file() else None
    if certified is not None and actual != certified:
        raise SystemExit(
            f"{source} does not match the sha256 recorded in its manifest "
            f"({actual} != {certified}); refusing to build a prior from an "
            "uncertified copy of the corpus"
        )
    return {
        "records": _relative(source),
        "dataset_id": str(payload.get("dataset_id") or payload.get("version") or "")
        or None,
        "manifest_kind": str(payload.get("kind") or "") or None,
        "manifest_content_fingerprint": str(payload.get("content_fingerprint") or "")
        or None,
        "records_sha256": actual,
    }


def _reject_eval_leakage(source: Path, records: list[ExampleRecord]) -> None:
    """Train-only prior: an eval manifest must never become the decode table."""
    posix = source.resolve().as_posix()
    kind = _manifest_kind(source)
    if kind == "eval" or "/data/eval/" in posix or posix.endswith("/eval"):
        raise SystemExit(
            f"refusing to build speculative n-gram table from eval corpus {source}"
        )
    if records and all(record.split in _EVAL_SPLITS for record in records):
        raise SystemExit(
            f"refusing to build speculative n-gram table from eval-split records in {source}"
        )


def _load(source: Path) -> list[ExampleRecord]:
    if source.is_dir() and (source / "manifest.json").is_file():
        kind = _manifest_kind(source)
        if kind == "eval":
            raise SystemExit(
                f"refusing to build speculative n-gram table from eval corpus {source}"
            )
        shards = [source / "records.jsonl"]
        if not shards[0].is_file():
            shards = sorted(source.glob("*.jsonl"))
        if not shards:
            raise SystemExit(f"no .jsonl shards under {source}")
        records: list[ExampleRecord] = []
        for shard in shards:
            if shard.name == "rejected.jsonl":
                continue
            records.extend(load_jsonl(shard))
    elif source.is_dir():
        shards = sorted(source.glob("*.jsonl"))
        if not shards:
            raise SystemExit(f"no .jsonl shards under {source}")
        records = []
        for shard in shards:
            records.extend(load_jsonl(shard))
    else:
        records = load_jsonl(source)
    _reject_eval_leakage(source, records)
    return records


def _records(source: Path) -> tuple[list[ExampleRecord], int]:
    """Return the declared-train records and the number of records dropped."""
    records = _load(source)
    train_only = [record for record in records if record.split in {"train", ""}]
    if not train_only:
        raise SystemExit(f"{source} contains no train-split records")
    return train_only, len(records) - len(train_only)


def _encode(
    records: list[ExampleRecord], codec: str
) -> tuple[list[list[int]], dict[str, int]]:
    """Encode templatized targets in the decoder codec the ranker will see."""
    if codec != "lexer":
        raise SystemExit(
            f"unsupported codec {codec!r}; the compiler completion domain the "
            "ranker scores is DSL-native ('lexer')"
        )
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer, SymbolTable

    tokenizer = DSLNativeTokenizer.build()
    sequences: list[list[int]] = []
    skipped = {"encode_error": 0, "unk_token": 0}
    for record in records:
        try:
            target = templatize(record.openui)
            table = SymbolTable.from_placeholders(
                target.placeholders, max_slots=tokenizer.sym_slots
            )
            ids = list(tokenizer.encode(target.source, table=table, add_special=False))
        except Exception:  # noqa: BLE001 - a record the codec rejects is not a prior
            skipped["encode_error"] += 1
            continue
        if tokenizer.unk_id in ids:
            skipped["unk_token"] += 1
            continue
        sequences.append(ids)
    if not sequences:
        raise SystemExit("no records survived templatization and encoding")
    return sequences, skipped


def _artifact(table_dict: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    return {**table_dict, "source": source}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        default=CERTIFIED_TRAIN_BUCKET,
        help="Train records .jsonl (or a directory of shards). Default: the "
        "certified train bucket openui_verified_train_v2.",
    )
    parser.add_argument(
        "--codec",
        default="lexer",
        help="Decoder codec the table is keyed on (only 'lexer' is supported).",
    )
    parser.add_argument(
        "--order", type=int, default=DEFAULT_ORDER, help="Back-off n-gram order."
    )
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
        "writing it: the corpus must match its manifest sha256, the "
        "artifact's recorded source (dataset id, manifest content "
        "fingerprint, records sha256) must match the live manifest, its "
        "corpus_fingerprint must be the rebuilt train-sequence fingerprint, "
        "and every other field must equal the rebuild. Non-zero exit on drift.",
    )
    args = parser.parse_args(argv)

    provenance = _corpus_provenance(args.records)
    records, dropped = _records(args.records)
    sequences, skipped = _encode(records, str(args.codec))
    table = build_ngram_table(sequences, order=int(args.order))
    source = {**provenance, "records_total": len(records) + dropped}
    expected = _artifact(table.to_dict(), source)
    summary = {
        **source,
        "declared_train": len(records),
        "skipped": skipped,
        "codec": args.codec,
        "order": table.order,
        "sequences": table.sequences,
        "tokens": table.tokens,
        "contexts": len(table.counts),
        "corpus_fingerprint": table.corpus_fingerprint,
    }

    if args.check:
        if not args.output.is_file():
            print(f"missing committed table: {args.output}", file=sys.stderr)
            return 1
        committed = json.loads(args.output.read_text(encoding="utf-8"))
        recorded_source = committed.get("source")
        if recorded_source != source:
            drifted = sorted(
                key
                for key in set(source) | set(recorded_source or {})
                if (recorded_source or {}).get(key) != source.get(key)
            )
            print(
                f"committed table {args.output} was not built from the corpus "
                f"it claims: source fields {drifted} differ from the live "
                f"manifest of {source['records']} "
                f"(recorded={json.dumps(recorded_source, sort_keys=True)}, "
                f"live={json.dumps(source, sort_keys=True)}). Rebuild with "
                "`python -m scripts.build_speculative_ngram_table`",
                file=sys.stderr,
            )
            return 1
        recorded = str(committed.get("corpus_fingerprint") or "")
        if recorded != table.corpus_fingerprint:
            print(
                f"committed table {args.output} records corpus_fingerprint "
                f"{recorded or '<none>'} but the train sequences of "
                f"{source['dataset_id'] or args.records} (order={args.order}) "
                f"fingerprint to {table.corpus_fingerprint}. Rebuild with "
                "`python -m scripts.build_speculative_ngram_table`",
                file=sys.stderr,
            )
            return 1
        if committed != expected:
            print(
                f"committed table {args.output} is stale; rebuild with "
                "`python -m scripts.build_speculative_ngram_table`",
                file=sys.stderr,
            )
            return 1
        print(json.dumps({**summary, "check": "ok"}, indent=2, sort_keys=True))
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({**summary, "output": str(args.output)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
