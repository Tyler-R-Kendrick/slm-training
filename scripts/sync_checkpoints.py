#!/usr/bin/env python3
"""Sync a local run's checkpoints to the Hugging Face OpenUI bucket."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.model_build.checkpoint_bucket import (
    DEFAULT_CHECKPOINT_BUCKET_URI,
    ensure_checkpoint_bucket,
    sync_run_checkpoints,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        help="outputs/runs/<run_id> (checkpoints/ inferred).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Explicit checkpoints directory (overrides --run-dir/checkpoints).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Remote prefix run id (default: directory name).",
    )
    parser.add_argument(
        "--bucket",
        default=DEFAULT_CHECKPOINT_BUCKET_URI,
        help=f"HF Bucket URI (default: {DEFAULT_CHECKPOINT_BUCKET_URI}).",
    )
    parser.add_argument(
        "--ensure-bucket",
        action="store_true",
        help="Create the bucket if missing before syncing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan uploads without writing to the bucket.",
    )
    args = parser.parse_args(argv)

    if args.checkpoint_dir is None and args.run_dir is None:
        parser.error("provide --run-dir and/or --checkpoint-dir")

    run_dir = args.run_dir
    ckpt_dir = args.checkpoint_dir
    if ckpt_dir is None:
        assert run_dir is not None
        ckpt_dir = run_dir / "checkpoints"
    if run_dir is None:
        run_dir = ckpt_dir.parent
    run_id = args.run_id or run_dir.name

    if args.ensure_bucket and not args.dry_run:
        print(json.dumps(ensure_checkpoint_bucket(args.bucket), indent=2))

    report = sync_run_checkpoints(
        ckpt_dir,
        run_id=run_id,
        bucket=args.bucket,
        run_dir=run_dir,
        dry_run=bool(args.dry_run),
        ensure_bucket=bool(args.ensure_bucket) and not args.dry_run,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
