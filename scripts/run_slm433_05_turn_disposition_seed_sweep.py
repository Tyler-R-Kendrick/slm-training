#!/usr/bin/env python3
"""VAR3-05 (SLM-433): seed-replication check for VAR3-04's held_out_improvement.

VAR3-04 (``run_slm433_04_turn_disposition_real_corpus.py``) reported one
real, single-seed ``held_out_improvement`` result and named its own scale
explicitly as a caveat: "one seed, one small held-out split, no
statistical test, no cross-seed replication." Its "Non-goals" section named
the obvious next honest step: does that result replicate across seeds, or
was it that one seed's noise?

This script answers that, and only that: it calls VAR3-04's own,
completely unmodified :func:`run_training` at the identical recipe
(``n_train_roots=8, n_dev_roots=4, dim=8, steps=30, learning_rate=0.05``)
across several seeds and reports the ``disposition`` and held-out
``composite_penalized_error_rate`` each seed actually produced -- no new
corpus-building or scoring code, no cherry-picking, whatever the seeds show.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.run_slm433_04_turn_disposition_real_corpus import (
    DIM,
    LEARNING_RATE,
    N_DEV_ROOTS,
    N_TRAIN_ROOTS,
    STEPS,
    run_training,
)
from slm_training.levers import MAX_RUN_MINUTES
from slm_training.versioning import build_version_stamp

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs/design/var3-05-turn-disposition-seed-sweep-20260727.json"
SEEDS: tuple[int, ...] = (433_04, 1, 2, 3, 4)


def run_sweep(
    *,
    seeds: tuple[int, ...] = SEEDS,
    n_train_roots: int = N_TRAIN_ROOTS,
    n_dev_roots: int = N_DEV_ROOTS,
    dim: int = DIM,
    steps: int = STEPS,
    learning_rate: float = LEARNING_RATE,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        report = run_training(
            n_train_roots=n_train_roots,
            n_dev_roots=n_dev_roots,
            dim=dim,
            steps=steps,
            learning_rate=learning_rate,
            seed=seed,
        )
        held_out = report["arms"]["held_out"]
        rows.append(
            {
                "seed": seed,
                "disposition": report["disposition"],
                "baseline_composite_penalized_error_rate": held_out["disposition_off"][
                    "composite_penalized_error_rate"
                ],
                "trained_composite_penalized_error_rate": held_out["disposition_trained"][
                    "composite_penalized_error_rate"
                ],
                "trained_abstention_rate": held_out["disposition_trained"]["abstention_rate"],
                "train_roots_skipped": report["corpus"]["train_roots_skipped"],
                "dev_roots_skipped": report["corpus"]["dev_roots_skipped"],
            }
        )

    disposition_counts = Counter(row["disposition"] for row in rows)
    n_seeds = len(rows)
    n_improvement = disposition_counts.get("held_out_improvement", 0)
    if n_improvement == n_seeds:
        verdict = "replicates_every_seed"
        note = (
            f"held_out_improvement replicated on all {n_seeds} seeds tried. Still not a "
            "capability or ship claim: every seed shares the same n_dev=4-root real "
            "corpus and the same fixed recipe -- this rules out single-seed noise as "
            "the explanation, it does not establish generalization to a larger or "
            "different held-out set."
        )
    elif n_improvement == 0:
        verdict = "does_not_replicate"
        note = (
            f"held_out_improvement did not recur on any of the {n_seeds} additional "
            "seeds tried; VAR3-04's single-seed result is honestly reported as likely "
            "seed noise rather than a real directional signal, per this issue's own "
            "falsification rule."
        )
    else:
        verdict = "seed_sensitive"
        note = (
            f"held_out_improvement recurred on {n_improvement} of {n_seeds} seeds -- a "
            "real, seed-sensitive effect at this corpus/held-out scale, not a robust "
            "one. Reported exactly as measured; no seed was excluded or re-run."
        )

    return {
        "schema": "var3_05_turn_disposition_seed_sweep/v1",
        "honesty": "fixture_or_scratch",
        "claim_class": "wiring",
        "verdict": verdict,
        "note": note,
        "max_wall_minutes_per_run": float(MAX_RUN_MINUTES),
        "recipe": {
            "n_train_roots": n_train_roots,
            "n_dev_roots": n_dev_roots,
            "dim": dim,
            "steps": steps,
            "learning_rate": learning_rate,
        },
        "disposition_counts": dict(disposition_counts),
        "rows": rows,
        "version_stamp": build_version_stamp(
            "config.levers",
            "dsl.operators.turn_disposition",
            "data.flow.turn_disposition_corpus",
            "harness.experiments.turn_disposition_training",
            "harness.train_data",
            "model.turn_disposition_head",
            "model.operator_feature_encoder",
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = parser.parse_args(argv)

    report = run_sweep(seeds=tuple(args.seeds))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
