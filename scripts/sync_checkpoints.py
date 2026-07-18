#!/usr/bin/env python3
"""Sync a local run's checkpoints to the Hugging Face OpenUI bucket."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from slm_training.harnesses.model_build.checkpoint_bucket import (
    DEFAULT_CHECKPOINT_BUCKET_URI,
    ensure_checkpoint_bucket,
    sync_run_checkpoints,
)

CLAIM_CLASSES = ("fixture", "diagnostic", "frontier", "ship_candidate")


def _git_head() -> str | None:
    """Best-effort current commit; ``None`` outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


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
    parser.add_argument(
        "--claim-class",
        choices=CLAIM_CLASSES,
        default="diagnostic",
        help=(
            "Evidentiary class of the checkpoint reference. "
            "frontier/ship_candidate require full provenance and a verified "
            "real sync (default: diagnostic)."
        ),
    )
    parser.add_argument(
        "--training-source-commit",
        default=None,
        help="Commit that produced the checkpoint (default: current git HEAD).",
    )
    parser.add_argument(
        "--evaluation-source-commit",
        default=None,
        help="Commit used to evaluate the checkpoint, when known.",
    )
    parser.add_argument(
        "--provenance-json",
        type=Path,
        default=None,
        help="Optional JSON file of extra provenance fields to attach.",
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

    provenance: dict[str, object] = {}
    if args.provenance_json is not None:
        provenance.update(json.loads(args.provenance_json.read_text(encoding="utf-8")))
    training_commit = args.training_source_commit or _git_head()
    if training_commit is not None:
        provenance.setdefault("training_source_commit", training_commit)
    if args.evaluation_source_commit is not None:
        provenance.setdefault("evaluation_source_commit", args.evaluation_source_commit)

    if args.ensure_bucket and not args.dry_run:
        print(json.dumps(ensure_checkpoint_bucket(args.bucket), indent=2))

    report = sync_run_checkpoints(
        ckpt_dir,
        run_id=run_id,
        bucket=args.bucket,
        run_dir=run_dir,
        dry_run=bool(args.dry_run),
        ensure_bucket=bool(args.ensure_bucket) and not args.dry_run,
        claim_class=args.claim_class,
        provenance=provenance,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
