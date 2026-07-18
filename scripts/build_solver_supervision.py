#!/usr/bin/env python3
"""Build a replay-verified solver supervision corpus from solver traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.distill.solver_supervision import (
    SupervisionConfig,
    build_solver_supervision,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace-root",
        type=Path,
        required=True,
        help="Root directory containing solver traces (traces.jsonl).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/data/solver_supervision"),
        help="Output root for versioned solver supervision datasets.",
    )
    parser.add_argument(
        "--version",
        default="v1",
        help="Dataset version id.",
    )
    parser.add_argument(
        "--verify-replay",
        action="store_true",
        help=(
            "Replay every certificate before emitting a hard label. "
            "Requires the caller to use the library API and register providers; "
            "the CLI alone has no default solver providers."
        ),
    )
    parser.add_argument(
        "--immutable",
        action="store_true",
        help="Fail if the version already exists instead of overwriting.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute counts and fingerprints without writing files.",
    )
    args = parser.parse_args(argv)

    from slm_training.runtime.telemetry import run_trace

    with run_trace(
        f"solver-supervision-{args.version}",
        "solver.supervision.build",
        attributes={"slm.data.id": args.version, "slm.data.kind": "solver_supervision"},
    ) as trace:
        config = SupervisionConfig(
            trace_root=args.trace_root,
            output_root=args.output_root,
            version=args.version,
            provider_registry=None,
            verify_replay=args.verify_replay,
            dry_run=args.dry_run,
            immutable=args.immutable,
        )
        result = build_solver_supervision(config)
        if not args.dry_run:
            manifest_path = result.output_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["trace_id"] = trace.trace_id
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result.manifest, indent=2))
    print(f"wrote {result.output_dir}")
    if result.rejected_traces:
        print(f"rejected {len(result.rejected_traces)} traces:")
        for reason in result.rejected_traces:
            print(f"  - {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
