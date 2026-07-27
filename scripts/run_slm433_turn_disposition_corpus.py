#!/usr/bin/env python3
"""SLM-433 (VAR3-03): build the real TurnDispositionHead corpus, train, and
compare disposition_off / derived_only / disposition_trained on a held-out,
trace-leakage-safe split -- across a preregistered seed sweep.

See ``slm_training.harnesses.experiments.slm433_turn_disposition_corpus`` for
the full design rationale, including why a single-seed run is not a
reproducible claim (CPU BLAS reduction order can differ across
environments/thread counts even with a fixed torch seed) and why the sweep
is the real evidence surface. Honesty: ``fixture_or_scratch`` -- a real,
freshly-enumerated symbolic corpus (not natural-language data), no
promotion/ship-gate claim regardless of outcome.

Example:
  python -m scripts.run_slm433_turn_disposition_corpus
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slm_training.harnesses.experiments.slm433_turn_disposition_corpus import (
    DEFAULT_SWEEP_SEEDS,
    run_seed_sweep_experiment,
)

DEFAULT_JSON_OUT = Path(
    "docs/design/var3-03-turn-disposition-corpus-training-20260727.json"
)
DEFAULT_MD_OUT = Path(
    "docs/design/var3-03-turn-disposition-corpus-training-20260727.md"
)


def render_markdown(payload: dict) -> str:
    lines = [
        "# VAR3-03 (SLM-433): TurnDispositionHead trained on a real corpus",
        "",
        f"**Disposition: `{payload['disposition']}`** "
        f"(wins={payload['wins']}, ties={payload['ties']}, losses={payload['losses']} "
        f"across {len(payload['seeds'])} seeds)",
        "",
        "Fixture-scale (real, freshly-enumerated symbolic corpus, not "
        "natural-language data). **Not a ship claim.**",
        "",
        "## Why a seed sweep, not a single run",
        "",
        "A single (corpus_seed, train_seed) point is not a reproducible "
        "claim: CPU BLAS reduction order can differ across "
        "environments/thread counts even with a fixed `torch.manual_seed`, "
        "and this toy corpus/model is small enough that one flipped "
        "held-out prediction flips the whole comparison. A reviewer re-run "
        "on a different machine reproduced the corpus/split exactly but got "
        "a different `final_train_loss` and the opposite single-seed "
        "disposition. This sweep across "
        f"{len(payload['seeds'])} independent seeds "
        "(each redrawing both the corpus and the trained head) is the real "
        "evidence surface.",
        "",
        "## Preregistered",
        "",
        f"- Hypothesis: {payload['preregistered']['hypothesis']}",
        f"- Aggregate stop rule: {payload['preregistered']['aggregate_stop_rule']}",
        f"- Leakage safety: {payload['preregistered']['leakage_safety']}",
        "",
        "## Per-seed results (every seed in the sweep, none excluded)",
        "",
        "| seed | trained_composite | derived_composite | outcome | final_train_loss |",
        "| ---: | ---: | ---: | --- | ---: |",
    ]
    for row in payload["per_seed"]:
        lines.append(
            f"| {row['seed']} | {row['trained_composite']:.4f} | "
            f"{row['derived_composite']:.4f} | {row['outcome']} | "
            f"{row['final_train_loss']:.4f} |"
        )
    lines += [
        "",
        "## Honesty",
        "",
        payload["honesty"],
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-tier", type=int, default=40)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--train-lr", type=float, default=0.05)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SWEEP_SEEDS),
        help="preregistered seed sweep (each seed drives both corpus_seed and train_seed)",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args(argv)

    payload = run_seed_sweep_experiment(
        n_per_tier=args.n_per_tier,
        train_fraction=args.train_fraction,
        train_steps=args.train_steps,
        train_lr=args.train_lr,
        seeds=tuple(args.seeds),
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    print(
        f"disposition={payload['disposition']} "
        f"wins={payload['wins']} ties={payload['ties']} losses={payload['losses']} "
        f"-> {args.json_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
