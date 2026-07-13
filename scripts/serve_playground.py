#!/usr/bin/env python3
"""Serve the TwoTower OpenUI web playground."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reload", action="store_true")
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
    args = parser.parse_args(argv)

    import uvicorn

    from slm_training.web.app import create_app

    app = create_app(
        checkpoint=args.checkpoint,
        device=args.device,
        annotations_path=args.annotations_path,
        human_train_path=args.human_train_path,
        human_pairs_path=args.human_pairs_path,
    )
    # Eager-load so the first request is fast
    try:
        app.state.service.load()
        print(f"Loaded checkpoint {args.checkpoint}")
    except FileNotFoundError as exc:
        print(f"WARNING: {exc}")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
