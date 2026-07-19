"""Audit root/type diversity for a synthetic training corpus.

Usage:
    python scripts/audit_corpus_diversity.py \
        --records outputs/data/train/v1/records.jsonl \
        --output outputs/runs/sde2-04-diversity-audit/audit-20260719.json

The script writes a JSON summary of unique canonical-root, sketch, topology,
type/action-multiset, prompt-intent, and lineage counts. It does not modify
the source corpus.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

from slm_training.dsl.schema import load_jsonl
from slm_training.harnesses.train_data.diversity import (
    fingerprint_record,
    summarize_fingerprints,
)
from slm_training.versioning import build_version_stamp


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit corpus diversity fingerprints")
    parser.add_argument("--records", required=True, type=Path, help="records.jsonl path")
    parser.add_argument("--output", required=True, type=Path, help="audit JSON path")
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="limit records audited (for quick smoke checks)",
    )
    args = parser.parse_args()

    records = load_jsonl(args.records)
    if args.max_records:
        records = records[: args.max_records]

    fingerprints = [fingerprint_record(record) for record in records]
    summary = summarize_fingerprints(fingerprints)

    per_source: dict[str, dict[str, int]] = {}
    for record, fp in zip(records, fingerprints):
        source = record.source or "unknown"
        bucket = per_source.setdefault(source, Counter())
        bucket["n"] += 1
        bucket["canonical_root_id"] += 1
        bucket["binding_aware_sketch"] += 1
        bucket["topology_sketch"] += 1
        bucket["type_action_multiset"] += 1
        bucket["prompt_intent_fingerprint"] += 1
        bucket["source_lineage_id"] += 1

    payload = {
        "schema_version": "corpus_diversity_audit/v1",
        "audit_date": date.today().isoformat().replace("-", ""),
        "records_path": str(args.records),
        **summary,
        "per_source": {k: dict(v) for k, v in per_source.items()},
        "version_stamp": build_version_stamp(),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote diversity audit to {args.output}")
    print(f"records: {summary['n_records']} unique roots: {summary['unique_counts']['canonical_root_id']}")


if __name__ == "__main__":
    main()
