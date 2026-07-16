#!/usr/bin/env python3
"""Publish a generated training corpus as a versioned source-controlled resource."""

from __future__ import annotations

import argparse
from pathlib import Path

from slm_training.data.store import DataStore


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-root", type=Path, default=Path("outputs/data/train"))
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=Path("src/slm_training/resources/data/train"),
    )
    args = parser.parse_args(argv)
    store = DataStore(
        local_root=args.source_root.parent,
        published_root=args.destination_root.parent,
    )
    ref = store.publish("train", args.version)
    print(ref.path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
