"""Profile train/held-out preference-gradient alignment by decision kind."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.preference.local_train import (
    diagnose_decision_gradient_alignment_from_paths,
)


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
    args = parser.parse_args(argv)
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
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
