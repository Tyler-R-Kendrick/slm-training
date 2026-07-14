#!/usr/bin/env python3
"""Benchmark generate latency and write outputs/cactus/bench.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.runtime.cactus import bench_pytorch_generate, write_bench
from slm_training.dsl.design_md import load_default_design_md


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/cactus/bench.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--with-design-md", action="store_true")
    args = parser.parse_args(argv)
    design = load_default_design_md() if args.with_design_md else None
    bench = bench_pytorch_generate(
        args.checkpoint,
        design_md=design,
        device=args.device,
        repeats=args.repeats,
    )
    write_bench(args.out, bench)
    print(json.dumps(bench.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
