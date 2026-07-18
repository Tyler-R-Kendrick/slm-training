"""CLI for the CAP0-02 exact arity analyzer (bounded arith-sketch fixture).

Example::

    python -m scripts.analyze_grammar_arity --fixture bounded-expr \\
        --max-ast-nodes 6 --max-live-bindings 2 --dimensions 4 \\
        --out outputs/runs/arity/bounded_expr_report.json

Emits deterministic JSON (``indent=2, sort_keys=True``) to both a scratch
``--out`` path and the durable ``docs/design/`` certificate. Fails closed
(non-zero exit + clear message) when enumeration is incomplete, required bounds
are missing, or version/signature metadata is absent. Heavy modules are imported
lazily inside :func:`main` so ``python -m scripts.analyze_grammar_arity --help``
stays cheap and never loads torch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DURABLE = "docs/design/cap0-02-arity-analyzer-20260718.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_grammar_arity",
        description="Exact arity / K^d capacity certificate for a bounded fixture.",
    )
    parser.add_argument(
        "--fixture", default="bounded-expr", help="committed fixture id"
    )
    parser.add_argument(
        "--max-ast-nodes",
        type=int,
        required=True,
        help="total AST node budget across all statements (required, >= 1)",
    )
    parser.add_argument(
        "--max-ast-depth",
        type=int,
        default=None,
        help="optional per-statement expression depth cap",
    )
    parser.add_argument(
        "--max-live-bindings",
        type=int,
        default=0,
        help="scope window: referenceable prior binders (also caps statements)",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=4,
        help="code length d for the K^d capacity row",
    )
    parser.add_argument(
        "--template-classes",
        default=None,
        help="comma-separated literal template classes (default: fixture's)",
    )
    parser.add_argument(
        "--max-programs",
        type=int,
        default=1_000_000,
        help="safety cap; exceeding it marks the analysis incomplete",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="scratch JSON path (default outputs/runs/arity/<fixture>_report.json)",
    )
    parser.add_argument(
        "--durable-out",
        default=DEFAULT_DURABLE,
        help="durable certificate JSON path under docs/design/",
    )
    return parser


def _resolve(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else ROOT / path


def _write_json(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Lazy imports: keep ``-m`` import cheap and the CLI Torch-free.
    from slm_training.dsl.analysis.arity.report import (
        AnalysisBounds,
        ExactArityReport,
        SchemaError,
        analyze,
    )
    from slm_training.dsl.analysis.arity.state_graph import FIXTURES

    # Fail closed: required bounds / unknown fixture.
    if args.fixture not in FIXTURES:
        print(
            f"error: unknown fixture {args.fixture!r}; known={sorted(FIXTURES)}"
        )
        return 2
    if args.max_ast_nodes < 1:
        print("error: --max-ast-nodes must be >= 1 (bounds required, no unbounded run)")
        return 2
    if args.max_live_bindings < 0:
        print("error: --max-live-bindings must be >= 0")
        return 2
    if args.dimensions < 1:
        print("error: --dimensions must be >= 1")
        return 2

    spec = FIXTURES[args.fixture]
    if args.template_classes is not None:
        template_classes = tuple(
            piece.strip() for piece in args.template_classes.split(",") if piece.strip()
        )
    else:
        template_classes = spec["template_classes"]

    bounds = AnalysisBounds(
        max_ast_nodes=args.max_ast_nodes,
        max_ast_depth=args.max_ast_depth,
        max_live_bindings=args.max_live_bindings,
        template_classes=template_classes,
        result_types=spec["result_types"],
    )

    report = analyze(
        fixture=args.fixture,
        bounds=bounds,
        dimensions=args.dimensions,
        max_programs=args.max_programs,
    )

    # Fail closed: incomplete enumeration must never be published as a certificate.
    if not report.complete:
        print(
            "error: enumeration incomplete (hit --max-programs cap); "
            "raise the cap or tighten bounds before certifying"
        )
        return 1

    # Fail closed: version/signature metadata must round-trip (rejects stale/missing).
    payload = report.to_dict()
    try:
        ExactArityReport.from_dict(payload)
    except SchemaError as exc:
        print(f"error: report failed schema/version validation: {exc}")
        return 1

    text = report.to_json()
    out_path = _resolve(
        args.out or f"outputs/runs/arity/{args.fixture}_report.json"
    )
    durable_path = _resolve(args.durable_out)
    _write_json(out_path, text)
    _write_json(durable_path, text)

    _print_summary(report, out_path, durable_path)
    return 0


def _print_summary(report: object, out_path: Path, durable_path: Path) -> None:
    data = report.to_dict()  # type: ignore[attr-defined]
    cap = data["capacity"]
    print(f"fixture: {data['fixture']}  (complete={data['complete']})")
    print(f"grammar_hash: {data['grammar_hash']}")
    print(
        "counts: "
        f"canonical_asts={data['canonical_ast_count']} "
        f"raw(frontier x scope)={data['raw_state_count']} "
        f"trie={data['trie_state_count']} "
        f"minimized={data['minimized_state_count']}"
    )
    print(
        "arity: "
        f"action_alphabet={data['action_alphabet_size']} "
        f"scope_signatures={data['scope_signature_count']} "
        f"max_local_branching={data['max_local_branching']}"
    )
    print(f"branching_histogram: {data['branching_histogram']}")
    print(f"completion_counts: {data['completion_counts']}")
    print(
        f"capacity: min K={cap['min_k']} with K^{cap['d']} >= "
        f"{cap['state_count']} minimized states"
    )
    print(
        "note: fixture certificate only; external CAP0-01 estimates "
        "(130/351/41/...) are NOT reproduced (see provenance.external_reference)."
    )
    print(f"wrote: {out_path}")
    print(f"wrote: {durable_path}")


if __name__ == "__main__":
    raise SystemExit(main())
