#!/usr/bin/env python3
"""Serve the TwoTower OpenUI web playground."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from slm_training.models.paths import PLAYGROUND_DEMO_CHECKPOINT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PLAYGROUND_DEMO_CHECKPOINT,
        help="Checkpoint path (default: fixtures/checkpoints/playground_demo/last.pt).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--public",
        action="store_true",
        help="Bind all interfaces; requires --annotation-token or SLM_ANNOTATION_TOKEN.",
    )
    parser.add_argument(
        "--annotation-token",
        default=None,
        help="Bearer token required by the annotation endpoint.",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--enable-jobs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the local control-plane job runner (default: on). "
        "Disable with --no-enable-jobs for a read-only server.",
    )
    parser.add_argument(
        "--annotations-path",
        type=Path,
        default=Path("outputs/annotations/feedback.jsonl"),
        help="Append-only JSONL for thumbs + notes",
    )
    parser.add_argument(
        "--human-train-path",
        type=Path,
        default=Path("fixtures/annotations/human_train.jsonl"),
    )
    parser.add_argument(
        "--human-pairs-path",
        type=Path,
        default=Path("outputs/preferences/human_pairs.jsonl"),
    )
    parser.add_argument(
        "--bad-outputs-path",
        type=Path,
        default=Path("outputs/annotations/bad_outputs.jsonl"),
        help="Append-only JSONL for invalid model outputs (negative training)",
    )
    args = parser.parse_args(argv)
    host = "0.0.0.0" if args.public else args.host
    annotation_token = args.annotation_token or os.getenv("SLM_ANNOTATION_TOKEN")
    if host not in {"127.0.0.1", "::1", "localhost"} and not annotation_token:
        parser.error(
            "public/non-loopback binding requires --annotation-token "
            "or SLM_ANNOTATION_TOKEN"
        )

    import uvicorn

    from slm_training.web.app import create_app

    app = create_app(
        checkpoint=args.checkpoint,
        device=args.device,
        annotations_path=args.annotations_path,
        human_train_path=args.human_train_path,
        human_pairs_path=args.human_pairs_path,
        bad_outputs_path=args.bad_outputs_path,
        annotation_token=annotation_token,
        execution=args.enable_jobs,
    )
    # Eager-load so the first request is fast
    try:
        app.state.service.load()
        print(f"Loaded checkpoint {args.checkpoint}")
    except FileNotFoundError as exc:
        print(f"WARNING: {exc}")

    uvicorn.run(
        app,
        host=host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
