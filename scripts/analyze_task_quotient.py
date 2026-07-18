#!/usr/bin/env python3
"""Analyze a task-confusability graph and neural state quotient (CAP1-03)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.dsl.analysis.arity.task_quotient import (
    AlignedActionRecord,
    TaskDistortionSpec,
    analyze_task_quotient,
)


def _load_records(path: Path) -> list[AlignedActionRecord]:
    records: list[AlignedActionRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        records.append(
            AlignedActionRecord(
                state_fingerprint=str(row["state_fingerprint"]),
                action_id=str(row["action_id"]),
                aligned_family=str(row.get("aligned_family", row["action_id"])),
                probability=row.get("probability"),
                value=row.get("value"),
                semantic_fingerprint=row.get("semantic_fingerprint"),
            )
        )
    return records


def _parse_capacities(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in text.split(","):
        k, d = part.strip().split(":")
        out.append((int(k), int(d)))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        required=True,
        help="JSONL of aligned action records (state_fingerprint, action_id, ...)",
    )
    parser.add_argument("--out", type=Path, required=True, help="JSON report output")
    parser.add_argument(
        "--markdown-out", type=Path, help="Optional Markdown report output"
    )
    parser.add_argument("--spec-id", default="cap-task-v1")
    parser.add_argument(
        "--policy-metric",
        choices=("js", "tv", "cross_entropy_regret", "topk_regret"),
        default="cross_entropy_regret",
    )
    parser.add_argument("--policy-tolerance", type=float, default=0.1)
    parser.add_argument("--value-weight", type=float, default=0.0)
    parser.add_argument("--execution-weight", type=float, default=0.0)
    parser.add_argument("--semantic-fingerprint-weight", type=float, default=0.0)
    parser.add_argument("--cvar-alpha", type=float, default=None)
    parser.add_argument("--cvar-tolerance", type=float, default=None)
    parser.add_argument(
        "--hard-forbidden-confusions",
        default="",
        help="Comma-separated semantic fingerprints that must not be merged",
    )
    parser.add_argument(
        "--capacities",
        default="2:4,3:4,4:4,8:3",
        help="Comma-separated K:d capacity pairs",
    )
    parser.add_argument(
        "--exact-coloring-max-vertices",
        type=int,
        default=128,
    )
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--max-refinement-iterations", type=int, default=4)
    args = parser.parse_args(argv)

    spec = TaskDistortionSpec(
        spec_id=args.spec_id,
        policy_metric=args.policy_metric,
        policy_tolerance=args.policy_tolerance,
        value_weight=args.value_weight,
        execution_weight=args.execution_weight,
        semantic_fingerprint_weight=args.semantic_fingerprint_weight,
        cvar_alpha=args.cvar_alpha,
        cvar_tolerance=args.cvar_tolerance,
        hard_forbidden_confusions=tuple(
            x.strip()
            for x in args.hard_forbidden_confusions.split(",")
            if x.strip()
        ),
    )

    records = _load_records(args.records)
    report = analyze_task_quotient(
        records,
        spec,
        capacities=_parse_capacities(args.capacities),
        exact_max_vertices=args.exact_coloring_max_vertices,
        refine=not args.no_refine,
        max_refinement_iterations=args.max_refinement_iterations,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.markdown_out:
        lines = [
            f"# Task quotient report ({args.spec_id})",
            "",
            f"- States: {report.state_count}",
            f"- Edges: {report.edge_count}",
            f"- Density: {report.density:.4f}",
            f"- Colors: {report.coloring.num_colors} (exact={report.coloring.exact})",
            f"- Lower bound: {report.coloring.lower_bound}",
            f"- Upper bound: {report.coloring.upper_bound}",
            f"- Algorithm: {report.coloring.algorithm}",
            "",
            "## Class size histogram",
            "",
            "| color | size |",
            "| --- | --- |",
        ]
        for color, size in sorted(report.class_size_histogram.items()):
            lines.append(f"| {color} | {size} |")
        lines.extend(["", "## Capacity feasibility", ""])
        for (k, d), feasible in report.capacity_feasibility.items():
            lines.append(f"- K={k}, d={d}: {'feasible' if feasible else 'infeasible'}")
        if report.counterexamples:
            lines.extend(["", "## Refinement counterexamples", ""])
            for ce in report.counterexamples:
                lines.append(f"- iter {ce['iteration']}: {ce['state_a']} vs {ce['state_b']} "
                             f"(color {ce['color']}, distance {ce['policy_distance']:.4g})")
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "states": report.state_count,
                "edges": report.edge_count,
                "colors": report.coloring.num_colors,
                "exact": report.coloring.exact,
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
