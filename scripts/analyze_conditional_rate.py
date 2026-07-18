#!/usr/bin/env python3
"""Analyze conditional task rate, Fano bounds, and posterior effective support.

CAP1-04 (SLM-84): consumes aligned action records and an optional CAP1-03 task
quotient, then emits state-conditioned entropy, Fano lower bounds, effective
support, and a finite rate-distortion curve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.dsl.analysis.arity.conditional_rate import (
    TaskDistortionSpec,
    analyze_conditional_rate,
)
from slm_training.dsl.analysis.arity.task_quotient import (
    AlignedActionRecord,
    QuotientReport,
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


def _load_quotient(path: Path) -> QuotientReport | None:
    data = json.loads(path.read_text(encoding="utf-8"))
    # Minimal reconstruction: we only need the coloring assignment and spec.
    from slm_training.dsl.analysis.arity.task_quotient import (
        ColoringResult,
        ConfusabilityGraph,
    )

    spec = TaskDistortionSpec(**data["spec"])
    colors = data.get("coloring", {}).get("colors")
    if colors is None:
        return None
    coloring = ColoringResult(
        colors={str(k): int(v) for k, v in colors.items()},
        num_colors=data["coloring"]["num_colors"],
        exact=data["coloring"]["exact"],
        lower_bound=data["coloring"]["lower_bound"],
        upper_bound=data["coloring"]["upper_bound"],
        algorithm=data["coloring"]["algorithm"],
    )
    return QuotientReport(
        spec=spec,
        graph=ConfusabilityGraph(vertices=set(colors), edges=set()),
        coloring=coloring,
        state_count=data["state_count"],
        edge_count=data["edge_count"],
        density=data["density"],
        class_size_histogram=data["class_size_histogram"],
        counterexamples=data["counterexamples"],
        capacity_feasibility={
            tuple(int(x) for x in k.replace("K=", "").replace("d=", "").split(",")): v
            for k, v in data["capacity_feasibility"].items()
        },
        estimated=data["estimated"],
    )


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
    parser.add_argument(
        "--task-quotient",
        type=Path,
        help="Optional CAP1-03 quotient report (for mutual-information estimate)",
    )
    parser.add_argument("--spec-id", default="cap-rate-v1")
    parser.add_argument(
        "--policy-metric",
        choices=("js", "tv", "cross_entropy_regret", "topk_regret"),
        default="cross_entropy_regret",
    )
    parser.add_argument("--policy-tolerance", type=float, default=0.1)
    parser.add_argument("--average-tolerance", type=float, default=None)
    parser.add_argument("--cvar-alpha", type=float, default=None)
    parser.add_argument("--cvar-tolerance", type=float, default=None)
    args = parser.parse_args(argv)

    spec = TaskDistortionSpec(
        spec_id=args.spec_id,
        policy_metric=args.policy_metric,
        policy_tolerance=args.policy_tolerance,
        average_tolerance=args.average_tolerance,
        cvar_alpha=args.cvar_alpha,
        cvar_tolerance=args.cvar_tolerance,
    )

    records = _load_records(args.records)
    quotient = _load_quotient(args.task_quotient) if args.task_quotient else None
    report = analyze_conditional_rate(records, spec, quotient=quotient)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if args.markdown_out:
        rd_rows = "\n".join(
            f"| {p.distortion:.4g} | {p.rate_bits:.4g} | {p.beta:.4g} | {p.exact} |"
            for p in report.rate_distortion_curve
        )
        lines = [
            f"# Conditional task-rate report ({args.spec_id})",
            "",
            f"- States: {report.state_count}",
            f"- Action alphabet size: {report.action_alphabet_size}",
            f"- Conditional entropy H(A|Q): {report.conditional_entropy_bits:.4f} bits",
            f"- Mutual information I(color; A): {report.mutual_information_bits}",
            f"- Estimated: {report.estimated}",
            "",
            "## Fano bounds",
            "",
            "| H(A|Q) bits | alphabet | P_e lower bound | exact |",
            "| --- | --- | --- | --- |",
        ]
        for b in report.fano_bounds:
            lines.append(
                f"| {b.conditional_entropy_bits:.4f} | {b.alphabet_size} "
                f"| {b.lower_bound_error:.4f} | {b.exact} |"
            )
        lines.extend(
            [
                "",
                "## Posterior effective support",
                "",
                f"- mean: {report.posterior_support.mean:.4f}",
                f"- median: {report.posterior_support.median:.4f}",
                f"- min: {report.posterior_support.min:.4f}",
                f"- max: {report.posterior_support.max:.4f}",
                "",
                "## Rate-distortion curve (quotient sweep)",
                "",
                "| distortion | rate bits | beta | exact |",
                "| --- | --- | --- | --- |",
            ]
        )
        lines.append(rd_rows)
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "states": report.state_count,
                "conditional_entropy_bits": report.conditional_entropy_bits,
                "mutual_information_bits": report.mutual_information_bits,
                "rd_points": len(report.rate_distortion_curve),
                "out": str(args.out),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
