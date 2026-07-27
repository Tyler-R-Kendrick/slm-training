#!/usr/bin/env python3
"""SLM-418 (DSH5-10) seventh slice: write the bounded replay-preference
corpus to real ``PreferencePair`` corpus files.

    python -m scripts.build_replay_preference_corpus
    python -m scripts.build_replay_preference_corpus --split train --out outputs/data/preference/replay_preference_train_pairs.jsonl

Writes into this repo's existing preference-pairs corpus root
(``outputs/data/preference/``) -- never a second corpus tree. This script
only builds and writes the corpus; feed the train-split file into the
existing ``slm preference train`` harness and measure the held-out split with
``scripts.measure_replay_preference_held_out_benefit`` (see
docs/design/dsh5-10-replay-preference-rows.md's reproducibility commands).
This is fixture-scale wiring evidence, not a ship-readiness claim, and
completes in well under a second -- far inside
``slm_training.levers.MAX_RUN_MINUTES``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.preference.replay_preference_corpus import (
    DEFAULT_HELD_OUT_OUT_PATH,
    DEFAULT_TRAIN_OUT_PATH,
    write_replay_preference_corpus,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("train", "held_out", "both"),
        default="both",
        help="Which split to write (default: both, one file each).",
    )
    parser.add_argument(
        "--train-out", type=Path, default=DEFAULT_TRAIN_OUT_PATH,
        help="Output path for the train-split pairs file.",
    )
    parser.add_argument(
        "--held-out-out", type=Path, default=DEFAULT_HELD_OUT_OUT_PATH,
        help="Output path for the held-out-split pairs file.",
    )
    args = parser.parse_args(argv)

    reports = []
    if args.split in ("train", "both"):
        reports.append(write_replay_preference_corpus(args.train_out, "train"))
    if args.split in ("held_out", "both"):
        reports.append(write_replay_preference_corpus(args.held_out_out, "held_out"))

    print(json.dumps([report.to_dict() for report in reports], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
