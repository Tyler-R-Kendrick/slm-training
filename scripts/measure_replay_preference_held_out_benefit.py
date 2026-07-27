#!/usr/bin/env python3
"""SLM-418 (DSH5-10) seventh slice: honest held-out pairwise-preference
benefit measurement for a real ``slm preference train`` run against the
replay-preference corpus.

    python -m scripts.measure_replay_preference_held_out_benefit \\
        --baseline-checkpoint outputs/runs/<scratch-id>/last.pt \\
        --held-out-pairs outputs/data/preference/replay_preference_held_out_pairs.jsonl \\
        [--trained-checkpoint outputs/runs/<preference-id>/model.pt]

Loads a baseline (pre-preference-training) checkpoint and, when given, a
trained (post ``slm preference train``) checkpoint, measures each one's
pairwise chosen>rejected preference accuracy on the held-out-split pairs, and
reports an honest ``verdict`` -- ``benefit_observed_fixture_scale`` only when
the trained checkpoint's accuracy strictly exceeds the baseline's, otherwise
``no_benefit_fixture_scale``. This is fixture-scale wiring evidence over a
handful of real pairs, never a certified or powered ship-readiness claim --
see docs/design/dsh5-10-replay-preference-rows.md's "Seventh slice".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.preference import load_pairs
from slm_training.harnesses.preference.train import (
    evaluate_replay_preference_held_out_benefit,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--trained-checkpoint", type=Path, default=None)
    parser.add_argument("--held-out-pairs", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--seed", type=int, default=0,
        help="torch.manual_seed applied before each measurement, for reproducibility.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    held_out_pairs = load_pairs(args.held_out_pairs)
    report = evaluate_replay_preference_held_out_benefit(
        baseline_checkpoint=args.baseline_checkpoint,
        trained_checkpoint=args.trained_checkpoint,
        held_out_pairs=held_out_pairs,
        device=args.device,
        seed=args.seed,
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    print(payload)
    if args.out:
        args.out.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
