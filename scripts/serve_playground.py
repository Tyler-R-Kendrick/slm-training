#!/usr/bin/env python3
"""Serve the TwoTower OpenUI web playground."""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/runs/playground_demo/checkpoints/last.pt"),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    import uvicorn

    from slm_training.web.app import create_app

    app = create_app(checkpoint=args.checkpoint, device=args.device)
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
