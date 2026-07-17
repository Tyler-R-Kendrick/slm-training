#!/usr/bin/env python3
"""E1: bits-per-semantic-decision over a corpus (production vs surface streams).

Measures how much grammar a model must learn, tokenizer-independently, and
(optionally) how many parameters are spent per bit. Diagnostic; no ship claim.
See ``slm_training.evals.semantic_bits`` and
``docs/design/iter-e249-semantic-bits-*.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_records(path: Path) -> list:
    from slm_training.dsl.schema import load_jsonl

    if path.is_dir():
        records = []
        for jsonl in sorted(path.rglob("records.jsonl")):
            records.extend(load_jsonl(jsonl))
        return records
    return list(load_jsonl(path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        required=True,
        help="records.jsonl file or a directory searched recursively for them.",
    )
    parser.add_argument(
        "--params",
        type=int,
        default=None,
        help="Trainable parameter count for params-per-bit (optional).",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    from slm_training.evals.semantic_bits import compare_representations

    records = _load_records(args.records)
    if args.limit is not None:
        records = records[: int(args.limit)]
    report = compare_representations(records, params=args.params)
    report["records"] = str(args.records)
    report["n_records"] = len(records)

    summary = {
        "n_records": len(records),
        "production_bits_per_decision": report["production"]["bits_per_decision"],
        "surface_bits_per_decision": report["surface"]["bits_per_decision"],
        "surface_to_production_bit_ratio": report["surface_to_production_bit_ratio"],
        "decision_reduction_ratio": report["decision_reduction_ratio"],
        "production_params_per_bit": report["production"].get("params_per_bit"),
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        summary["out"] = str(args.out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
