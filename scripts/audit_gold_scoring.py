"""CLI to audit gold scoring through oracle replay variants (SLM-260 / VSD0-01)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from slm_training.dsl.schema import load_jsonl
from slm_training.evals.oracle_scoring_replay import (
    build_fixture_records,
    build_replay_manifest,
    build_variant_rows,
    score_rows,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run oracle scoring replay variants through the production eval path."
    )
    parser.add_argument(
        "--records",
        type=Path,
        default=None,
        help="JSONL path of ExampleRecords to replay (default: built-in fixture records)",
    )
    parser.add_argument(
        "--suite",
        default="oracle_replay",
        help="Suite name for the replay manifest",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the replay manifest JSON",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Optional run directory (recorded in status output, not used for scoring)",
    )
    args = parser.parse_args(argv)

    if args.records is None:
        records = build_fixture_records()
    else:
        records = load_jsonl(args.records)

    rows = build_variant_rows(records, suite=args.suite)
    scored_rows = score_rows(rows)
    manifest = build_replay_manifest(rows, scored_rows, suite=args.suite)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if args.run_dir is not None:
        args.run_dir.mkdir(parents=True, exist_ok=True)

    status = {
        "status": "ok",
        "output": str(args.output),
        "n": manifest["n"],
        "suite": args.suite,
        "schema_version": manifest["schema_version"],
    }
    if args.run_dir is not None:
        status["run_dir"] = str(args.run_dir)
    print(json.dumps(status))
    return 0


if __name__ == "__main__":
    sys.exit(main())
