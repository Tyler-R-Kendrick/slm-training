"""Profile train/held-out preference-gradient alignment by decision kind."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

from slm_training.harnesses.preference.constraint_debt import ConstraintDebtV1
from slm_training.harnesses.preference.local_train import (
    diagnose_decision_gradient_alignment_from_paths,
)


def _summarize_debt_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate constraint-debt rows by decision kind, split, and probability space."""
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["decision_kind"], row["split"], row["probability_space"])
        groups.setdefault(key, []).append(row)

    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    summary: dict[str, Any] = {}
    for (decision_kind, split, probability_space), group in sorted(groups.items()):
        total = len(group)
        numeric_fields = (
            "legal_debt",
            "good_debt",
            "bad_debt",
            "legal_mass_deficit",
            "pre_post_mask_kl",
        )
        aggregates: dict[str, Any] = {"count": total}
        for field in numeric_fields:
            values = [row[field] for row in group if row[field] is not None]
            aggregates[field] = {
                "mean": _mean(values),
                "median": median(values) if values else 0.0,
                "non_null_count": len(values),
                "null_count": total - len(values),
            }
        aggregates["single_legal_action_fraction"] = (
            sum(row["single_legal_action"] for row in group) / total
            if total
            else 0.0
        )
        aggregates["empty_good_partition_fraction"] = (
            sum(row["empty_good_partition"] for row in group) / total
            if total
            else 0.0
        )
        aggregates["empty_bad_partition_fraction"] = (
            sum(row["empty_bad_partition"] for row in group) / total
            if total
            else 0.0
        )
        summary[f"{decision_kind}::{split}::{probability_space}"] = aggregates
    return {
        "row_count": len(rows),
        "by_decision_kind_split_space": summary,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--objective", default="ftpo_set")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--metric-complete", action="store_true")
    parser.add_argument(
        "--probability-space",
        choices=("full_vocab", "legal_tokens"),
        default="full_vocab",
        help="Probability denominator for metric-complete mass objectives.",
    )
    parser.add_argument(
        "--gradient-scaling",
        choices=("raw", "unit_norm"),
        default="raw",
        help="Geometry used to combine metric-complete objective gradients.",
    )
    parser.add_argument(
        "--train-strata",
        choices=("decision_kind", "decision_signature"),
        default="decision_kind",
    )
    parser.add_argument(
        "--held-out-strata",
        choices=("decision_kind", "decision_signature"),
        default="decision_kind",
    )
    parser.add_argument(
        "--emit-debt",
        type=Path,
        default=None,
        help="Optional JSONL file to write one ConstraintDebtV1 row per event/space.",
    )
    parser.add_argument(
        "--debt-summary",
        type=Path,
        default=None,
        help="Optional aggregate JSON file summarizing the emitted debt rows.",
    )
    args = parser.parse_args(argv)

    debt_rows: list[dict[str, Any]] = []
    debt_writer = None
    if args.emit_debt is not None:
        def debt_writer(row: ConstraintDebtV1) -> None:
            debt_rows.append(row.to_dict())

    report = diagnose_decision_gradient_alignment_from_paths(
        args.checkpoint,
        args.events,
        objective=args.objective,
        device=args.device,
        metric_complete=args.metric_complete,
        probability_space=args.probability_space,
        gradient_scaling=args.gradient_scaling,
        train_strata=args.train_strata,
        held_out_strata=args.held_out_strata,
        debt_writer=debt_writer,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.emit_debt is not None:
        args.emit_debt.parent.mkdir(parents=True, exist_ok=True)
        with args.emit_debt.open("w", encoding="utf-8") as handle:
            for row in debt_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    if args.debt_summary is not None:
        args.debt_summary.parent.mkdir(parents=True, exist_ok=True)
        args.debt_summary.write_text(
            json.dumps(_summarize_debt_rows(debt_rows), indent=2) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
