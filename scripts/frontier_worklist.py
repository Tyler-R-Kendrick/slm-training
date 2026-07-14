#!/usr/bin/env python3
"""Write the deterministic train-only frontier artifact worklist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.data.frontier import write_worklist
from slm_training.dsl.schema import load_jsonl


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records", type=Path, default=Path("outputs/train_data/v1/records.jsonl")
    )
    parser.add_argument("--root", type=Path, default=Path("fixtures/frontier"))
    parser.add_argument("--skill-name", default="frontier-describe")
    parser.add_argument("--skill-version", default="1")
    args = parser.parse_args(argv)

    manifest = write_worklist(
        load_jsonl(args.records),
        root=args.root,
        skill_name=args.skill_name,
        skill_version=args.skill_version,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
