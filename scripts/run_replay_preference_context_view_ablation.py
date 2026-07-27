#!/usr/bin/env python3
"""Run the SLM-418 (DSH5-10) bounded, fixture-scale context-view ablation.

Builds the small deterministic synthetic session corpus
(:func:`synthesize_bounded_session_corpus`), trains one tiny two-feature
pairwise scorer per ``(context_view, turn_depth)`` cell, and prints the real
disposition (:mod:`slm_training.evals.ambiguous_operator_followups`) as JSON.

    python -m scripts.run_replay_preference_context_view_ablation

This is fixture-scale wiring evidence, not a ship-readiness claim -- see
``docs/design/dsh5-10-replay-preference-rows.md``. It completes in well under
a second, far inside ``slm_training.levers.MAX_RUN_MINUTES``.
"""

from __future__ import annotations

import argparse
import json

from slm_training.evals.ambiguous_operator_followups import (
    evaluate_ambiguous_operator_followups,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to also write the JSON report to.",
    )
    args = parser.parse_args(argv)

    report = evaluate_ambiguous_operator_followups()
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    print(payload)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
