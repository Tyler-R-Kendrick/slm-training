#!/usr/bin/env python3
"""Generate verified typed-AST ProgramSpec roots and a coverage report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.data.progspec import TypedProgramGenerator


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--max-width", type=int, default=4)
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/progspec/programs.jsonl")
    )
    parser.add_argument(
        "--coverage", type=Path, default=Path("outputs/progspec/coverage.json")
    )
    args = parser.parse_args(argv)

    result = TypedProgramGenerator(
        seed=args.seed,
        max_depth=args.max_depth,
        max_width=args.max_width,
    ).generate(args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.coverage.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(spec.to_dict(), sort_keys=True) + "\n"
            for spec in result.programs
        ),
        encoding="utf-8",
    )
    args.coverage.write_text(
        json.dumps(result.coverage, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(result.programs)} ProgramSpecs to {args.output}")
    print(
        f"coverage: {result.coverage['covered_count']}/{result.coverage['target_count']}"
    )
    return 0 if len(result.programs) == args.count else 1


if __name__ == "__main__":
    raise SystemExit(main())
