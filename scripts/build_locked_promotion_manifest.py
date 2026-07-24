#!/usr/bin/env python3
"""Create a local, immutable promotion holdout from canonical records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.data.locked_eval_manifest import build_locked_manifest, write_locked_manifest
from slm_training.dsl.schema import load_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--source-records", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-locked-records", type=int, default=200)
    parser.add_argument("--partition-size", type=int, default=20)
    args = parser.parse_args(argv)
    manifest = build_locked_manifest(
        load_jsonl(args.candidates),
        source_records=[record for path in args.source_records for record in load_jsonl(path)],
        min_locked_records=args.min_locked_records,
        partition_size=args.partition_size,
    )
    digest = write_locked_manifest(args.output, manifest)
    print(json.dumps({"manifest": str(args.output), "manifest_sha256": digest, **manifest.metadata}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
