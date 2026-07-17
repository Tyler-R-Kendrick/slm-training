"""CLI for the CAP0-02/03/04 grammar arity analyzer and certificate reporter."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from slm_training.dsl.analysis.arity import AnalysisBounds, ArityAnalyzer
from slm_training.dsl.analysis.arity.certificate import exact_certificate_from_report
from slm_training.dsl.analysis.arity.render import one_line_summary, to_csv, to_markdown


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
    parser.add_argument(
        "--certificate",
        action="store_true",
        help="Emit a CAP0-04 certificate bundle instead of a plain report",
    )
    parser.add_argument("--out-md", help="Write Markdown certificate report")
    parser.add_argument("--out-csv", help="Write CSV certificate rows")
    parser.add_argument("--one-line", action="store_true", help="Print one-line summary")
    parser.add_argument("--source-commit", help="Git commit for certificate provenance")
    parser.add_argument("--run-id", help="Run ID for certificate provenance")
    parser.add_argument("--trace-id", help="Trace ID for certificate provenance")
    args = parser.parse_args(argv)

    bounds = AnalysisBounds(
        max_ast_nodes=args.max_ast_nodes,
        max_ast_depth=args.max_ast_depth,
        max_live_bindings=args.max_live_bindings,
    )

    seed_sources: list[str] | None = None
    input_hashes: tuple[str, ...] = ()
    if args.program:
        seed_sources = [args.program]
    elif args.program_file:
        source = Path(args.program_file).read_text(encoding="utf-8")
        seed_sources = [source]

    analyzer = ArityAnalyzer(args.dsl, bounds)
    report = analyzer.analyze(seed_sources, include_coding_metadata=args.include_coding_metadata)

    if args.certificate:
        generated_at = datetime.now(timezone.utc).isoformat()
        bundle = exact_certificate_from_report(
            report,
            generated_at=generated_at,
            source_commit=args.source_commit,
            run_id=args.run_id,
            trace_id=args.trace_id,
            input_hashes=input_hashes,
        )
        output = bundle.to_json(indent=2)
        if args.out_md:
            Path(args.out_md).write_text(to_markdown(bundle), encoding="utf-8")
        if args.out_csv:
            Path(args.out_csv).write_text(to_csv(bundle), encoding="utf-8")
        if args.one_line:
            print(one_line_summary(bundle))
    else:
        output = report.to_json(indent=2)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    elif not args.one_line:
        print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
