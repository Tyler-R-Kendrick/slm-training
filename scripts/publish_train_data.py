#!/usr/bin/env python3
"""Publish a generated training corpus as a versioned source-controlled resource."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--source-root", type=Path, default=Path("outputs/train_data"))
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=Path("src/slm_training/resources/train_data"),
    )
    args = parser.parse_args(argv)
    if not args.version.replace("_", "").replace("-", "").replace(".", "").isalnum():
        raise SystemExit("invalid version")

    source = args.source_root / args.version
    destination = args.destination_root / args.version
    if not (source / "records.jsonl").is_file() or not (
        source / "manifest.json"
    ).is_file():
        raise SystemExit(f"incomplete corpus: {source}")

    manifest = json.loads((source / "manifest.json").read_text())
    if not manifest.get("record_count") and not manifest.get("records"):
        raise SystemExit("manifest has no record count")

    destination.mkdir(parents=True, exist_ok=True)
    for name in ("records.jsonl", "manifest.json", "stats.json"):
        source_file = source / name
        if source_file.is_file():
            shutil.copyfile(source_file, destination / name)
    print(
        json.dumps(
            {
                "version": args.version,
                "destination": str(destination),
                "record_count": manifest.get("record_count"),
                "manifest_sha": manifest.get("manifest_sha256"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
