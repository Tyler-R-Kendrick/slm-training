#!/usr/bin/env python3
"""Serve the TwoTower OpenUI web playground."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint path. Default: auto-resolve the latest model we're "
            "building (SLM_PLAYGROUND_CHECKPOINT env pin, then the deployed "
            "lineage pointer, then the promoted champion, then the newest "
            "outputs/ run checkpoint, then the committed playground demo)."
        ),
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
    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "Device: auto|cpu|cuda|npu:0|directml "
            "(auto picks the best available accelerator)."
        ),
    )
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
        default=Path("outputs/data/annotation/feedback.jsonl"),
        help="Append-only JSONL for thumbs + notes",
    )
    parser.add_argument(
        "--human-train-path",
        type=Path,
        default=Path("src/slm_training/resources/annotations/human_train.jsonl"),
    )
    parser.add_argument(
        "--human-pairs-path",
        type=Path,
        default=Path("outputs/data/preference/human_pairs.jsonl"),
    )
    parser.add_argument(
        "--bad-outputs-path",
        type=Path,
        default=Path("outputs/data/annotation/bad_outputs.jsonl"),
        help="Append-only JSONL for invalid model outputs (negative training)",
    )
    parser.add_argument(
        "--generation-attempts-path",
        type=Path,
        default=Path("outputs/data/annotation/generation_attempts.jsonl"),
        help="Append-only JSONL for every server and browser model attempt",
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

    from slm_training.runtime.accel import detect_device
    from slm_training.web.app import create_app

    accel = detect_device(args.device)
    device = (
        accel.device
        if args.device in {"auto", "best", "dml", "directml"}
        else args.device
    )
    print(f"Inference device: {device} ({accel.note})")
    app = create_app(
        checkpoint=args.checkpoint,
        device=device,
        annotations_path=args.annotations_path,
        human_train_path=args.human_train_path,
        human_pairs_path=args.human_pairs_path,
        bad_outputs_path=args.bad_outputs_path,
        generation_attempts_path=args.generation_attempts_path,
        annotation_token=annotation_token,
        execution=args.enable_jobs,
    )
    # Eager-load so the first request is fast
    service = app.state.service
    resolution = service.info()["checkpoint_resolution"]
    print(
        "Serving checkpoint "
        f"{resolution['path']} (provenance: {resolution['provenance']}, "
        f"run: {resolution['run_id'] or 'unknown'})"
    )
    try:
        service.load()
        print(f"Loaded checkpoint {service.checkpoint}")
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
