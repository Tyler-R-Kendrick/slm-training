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
    args = parser.parse_args(argv)
    report = diagnose_decision_gradient_alignment_from_paths(
        args.checkpoint,
        args.events,
        objective=args.objective,
        device=args.device,
        metric_complete=args.metric_complete,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
