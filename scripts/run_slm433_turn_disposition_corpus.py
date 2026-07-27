#!/usr/bin/env python3
"""SLM-433 (VAR3-03): build the real TurnDispositionHead corpus, train, and
compare disposition_off / derived_only / disposition_trained on a held-out,
trace-leakage-safe split.

See ``slm_training.harnesses.experiments.slm433_turn_disposition_corpus`` for
the full design rationale. Honesty: ``fixture_or_scratch`` -- a real,
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
    run_experiment,
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
        f"**Disposition: `{payload['disposition']}`**",
        "",
        "Fixture-scale (real, freshly-enumerated symbolic corpus, not "
        "natural-language data). **Not a ship claim.**",
        "",
        "## Preregistered",
        "",
        f"- Hypothesis: {payload['preregistered']['hypothesis']}",
        f"- Stop rule: {payload['preregistered']['stop_rule']}",
        f"- Leakage safety: {payload['preregistered']['leakage_safety']}",
        "",
        "## Corpus",
        "",
        f"- `n_per_tier`={payload['corpus_config']['n_per_tier']}, "
        f"`train_fraction`={payload['corpus_config']['train_fraction']}, "
        f"tier margins={payload['corpus_config']['tier_margins']}",
        f"- Total traces: {payload['quality_report']['total_traces']}, "
        f"total rows: {payload['quality_report']['total_rows']}",
        f"- Rows by kind: {payload['quality_report']['rows_by_kind']}",
        f"- Rows by split: {payload['quality_report']['rows_by_split']}",
        f"- `out_of_scope_never_from_nonempty_set`: "
        f"{payload['quality_report']['out_of_scope_never_from_nonempty_set']}",
        f"- Leakage-safe split (disjoint trace ids): {payload['leakage_safe_split']}",
        f"- Scored rows: {payload['train_scored_count']} train / "
        f"{payload['heldout_scored_count']} heldout",
        "",
        "## Three-arm comparison (identical held-out split, matched budget)",
        "",
        "| arm | case_count | wrong_op_rate | abstention_rate | composite_penalized_error_rate |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for arm in payload["arms"]:
        s = payload["arm_scores"][arm]
        lines.append(
            f"| {arm} | {s['case_count']} | {s['wrong_op_rate']:.4f} | "
            f"{s['abstention_rate']:.4f} | {s['composite_penalized_error_rate']:.4f} |"
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
    parser.add_argument("--corpus-seed", type=int, default=433)
    parser.add_argument("--train-steps", type=int, default=200)
    parser.add_argument("--train-lr", type=float, default=0.05)
    parser.add_argument("--train-seed", type=int, default=433)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    args = parser.parse_args(argv)

    payload = run_experiment(
        n_per_tier=args.n_per_tier,
        train_fraction=args.train_fraction,
        corpus_seed=args.corpus_seed,
        train_steps=args.train_steps,
        train_lr=args.train_lr,
        train_seed=args.train_seed,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.md_out.write_text(render_markdown(payload), encoding="utf-8")
    print(
        f"disposition={payload['disposition']} "
        f"derived_only_composite={payload['arm_scores']['derived_only']['composite_penalized_error_rate']:.4f} "
        f"trained_composite={payload['arm_scores']['disposition_trained']['composite_penalized_error_rate']:.4f} "
        f"-> {args.json_out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
