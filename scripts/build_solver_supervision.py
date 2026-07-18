#!/usr/bin/env python3
"""Build a replay-verified solver supervision corpus (VSS3-01 / SLM-69).

Thin CLI over ``slm_training.harnesses.distill.solver_supervision``. Consumes
SLM-64 solver traces (a ``TraceStore`` root or a raw traces JSONL) and emits a
versioned support-set / candidate-cost corpus with a manifest and validation
report. Only replay-valid solver states and certificates produce hard labels.

Examples
--------
    python -m scripts.build_solver_supervision \
        --trace-root outputs/traces \
        --output outputs/data/solver_supervision/<id> \
        --verify-replay --manifest

    python -m scripts.build_solver_supervision \
        --trace-root outputs/traces --describe   # dry run, writes nothing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator

from slm_training.harnesses.distill.solver_supervision import (
    build_solver_supervision,
    is_solver_trace,
    iter_solver_traces,
)


def _load_traces(trace_root: Path) -> Iterator[dict]:
    """Yield solver traces from a TraceStore root dir or a raw JSONL file."""
    if trace_root.is_file():
        with trace_root.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    trace = json.loads(line)
                    if is_solver_trace(trace):
                        yield trace
        return
    yield from iter_solver_traces(trace_root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-root",
        type=Path,
        required=True,
        help="TraceStore root directory or a traces.jsonl file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output corpus directory (omit with --describe for a dry run).",
    )
    parser.add_argument(
        "--verify-replay",
        dest="verify_replay",
        action="store_true",
        default=True,
        help="Reject traces with replay violations (default: on).",
    )
    parser.add_argument(
        "--no-verify-replay",
        dest="verify_replay",
        action="store_false",
        help="Disable replay verification (unsafe; for diagnostics only).",
    )
    parser.add_argument(
        "--oracle-backend-version",
        default=None,
        help="Override the recorded support-oracle backend version.",
    )
    parser.add_argument(
        "--manifest",
        action="store_true",
        help="Write a manifest (already emitted on any non-dry-run build).",
    )
    parser.add_argument(
        "--describe",
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Report counts and validation errors without writing.",
    )
    args = parser.parse_args(argv)

    if args.output is None and not args.dry_run:
        parser.error("provide --output, or use --describe for a dry run")

    build_command = ["python", "-m", "scripts.build_solver_supervision"]
    summary = build_solver_supervision(
        _load_traces(args.trace_root),
        output_dir=None if args.dry_run else args.output,
        verify_replay=args.verify_replay,
        oracle_backend_version=args.oracle_backend_version,
        dry_run=args.dry_run,
        build_command=build_command,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
