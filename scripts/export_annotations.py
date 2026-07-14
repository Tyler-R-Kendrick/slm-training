#!/usr/bin/env python3
"""Export playground annotations into SFT seeds and preference pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.annotations import (
    DEFAULT_FEEDBACK_PATH,
    DEFAULT_HUMAN_PAIRS_PATH,
    DEFAULT_HUMAN_TRAIN_PATH,
    export_all,
    load_annotations,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    status = sub.add_parser("status", help="Show annotation counts")
    status.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK_PATH)

    export = sub.add_parser("export", help="Rebuild human train seeds + preference pairs")
    export.add_argument("--feedback", type=Path, default=DEFAULT_FEEDBACK_PATH)
    export.add_argument("--human-train", type=Path, default=DEFAULT_HUMAN_TRAIN_PATH)
    export.add_argument("--pairs", type=Path, default=DEFAULT_HUMAN_PAIRS_PATH)

    args = parser.parse_args(argv)
    if args.cmd == "status":
        rows = load_annotations(args.feedback)
        ups = sum(1 for r in rows if r.rating == "up")
        downs = sum(1 for r in rows if r.rating == "down")
        payload = {
            "feedback": str(args.feedback),
            "total": len(rows),
            "up": ups,
            "down": downs,
        }
        print(json.dumps(payload, indent=2))
        return 0

    result = export_all(
        feedback_path=args.feedback,
        human_train_path=args.human_train,
        pairs_path=args.pairs,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
