#!/usr/bin/env python3
"""Export TwoTower checkpoint as a Cactus-oriented bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.cactus import export_checkpoint_bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/cactus/bundle"))
    args = parser.parse_args(argv)
    out = export_checkpoint_bundle(args.checkpoint, args.out_dir)
    print(json.dumps({"out_dir": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
