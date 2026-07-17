"""CLI for the CAP0-02/03/04 grammar arity analyzer and certificate reporter."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from slm_training.dsl.analysis.arity import (
    AnalysisBounds,
    AnalysisProfile,
    ArityAnalyzer,
    StateGraph,
    get_profile,
)
from slm_training.dsl.analysis.arity.certificate import exact_certificate_from_report
from slm_training.dsl.analysis.arity.render import one_line_summary, to_csv, to_markdown
from slm_training.models.choice_tokenizer import ChoiceTokenizer


def _state_graph_markdown(report: object) -> str:
    lines = [
        "# CAP1-01 bounded OpenUI state-graph analysis",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}Z",
        "",
        "## Constraint frame",
        "",
        "```json",
        f"{json.dumps(report.profile.to_dict(), indent=2, sort_keys=True)}",
        "```",
        "",
        "## Result summary",
        "",
        f"- Status: **{report.status}**",
        f"- Exact: {'yes' if report.exact else 'no'}",
        f"- Raw states: {report.raw_states}",
        f"- Minimized states: {report.minimized_states}",
        f"- Transitions: {report.transition_count}",
        f"- Terminal: {report.terminal_count}",
        f"- Invalid: {report.invalid_count}",
        f"- Unknown: {report.unknown_count}",
        "",
        "## Work counters",
        "",
        "```json",
        f"{json.dumps(report.work_counters, indent=2, sort_keys=True)}",
        "```",
        "",
        "## Histograms",
        "",
        f"- Branching: {dict(report.branching_histogram)}",
        f"- Forced suffix length: {dict(report.forced_decision_histogram)}",
        "",
        "## Honest caveat",
        "",
        "This is a bounded structural quotient over the choice-codec owner. "
        "It is wiring evidence only and does not claim that the minimized "
        "count is the latent model capacity or ship-grade state optimum.",
    ]
    return "\n".join(lines) + "\n"


def _run_state_graph(args: argparse.Namespace) -> int:
    profile_id = args.profile or "openui-cap-v1"
    profile = get_profile(profile_id)
    if args.representation:
        profile = AnalysisProfile(
            **{**profile.to_dict(), "representation": args.representation}
        )
    slot_contract: tuple[str, ...] = (
        tuple(args.slot_contract.split(","))
        if args.slot_contract
        else ()
    )
    tokenizer = ChoiceTokenizer.build()
    graph = StateGraph(profile, tokenizer, slot_contract=slot_contract)
    report = graph.explore()
    output = report.to_json(indent=2)

    json_out = args.json_out or args.out
    if json_out:
        Path(json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(json_out).write_text(output, encoding="utf-8")
    if args.markdown_out:
        Path(args.markdown_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown_out).write_text(_state_graph_markdown(report), encoding="utf-8")
    if args.one_line:
        print(report.one_line_summary())
    elif not json_out:
        print(output)
    return 0


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
    # CAP1-01 state-graph path
    parser.add_argument("--dsl-pack", help="DSL pack id (e.g. openui) for CAP1-01")
    parser.add_argument("--profile", help="Analysis profile id (e.g. openui-cap-v1)")
    parser.add_argument(
        "--representation", help="State representation (choice or compiler)"
    )
    parser.add_argument(
        "--slot-contract", help="Comma-separated placeholder contract"
    )
    parser.add_argument("--json-out", help="CAP1-01 JSON output path")
    parser.add_argument("--markdown-out", help="CAP1-01 Markdown output path")
    args = parser.parse_args(argv)

    if args.dsl_pack:
        return _run_state_graph(args)

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
