"""CLI for the CAP0-02 grammar arity analyzer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from slm_training.dsl.analysis.arity import AnalysisBounds, ArityAnalyzer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze bounded grammar arity.")
    parser.add_argument("--dsl", default="toy-layout", help="DSL id to analyze")
    parser.add_argument("--program", help="Single source program to analyze")
    parser.add_argument("--program-file", help="File containing a source program")
    parser.add_argument("--max-ast-nodes", type=int, default=128)
    parser.add_argument("--max-ast-depth", type=int)
    parser.add_argument("--max-live-bindings", type=int, default=0)
    parser.add_argument("--out", help="Output JSON file")
    parser.add_argument(
        "--include-coding-metadata",
        action="store_true",
        help="Attach CAP0-03 coding-theory metadata to the report",
    )
    args = parser.parse_args(argv)

    bounds = AnalysisBounds(
        max_ast_nodes=args.max_ast_nodes,
        max_ast_depth=args.max_ast_depth,
        max_live_bindings=args.max_live_bindings,
    )

    seed_sources: list[str] | None = None
    if args.program:
        seed_sources = [args.program]
    elif args.program_file:
        seed_sources = [Path(args.program_file).read_text(encoding="utf-8")]

    analyzer = ArityAnalyzer(args.dsl, bounds)
    report = analyzer.analyze(seed_sources, include_coding_metadata=args.include_coding_metadata)
    output = report.to_json(indent=2)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
