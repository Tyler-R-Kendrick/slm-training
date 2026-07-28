#!/usr/bin/env python3
"""Run the SLM-418 (DSH5-10) eighth-slice TypedOperatorPolicyScorer probe.

Builds the representability report over the seventh slice's full 56-row
synthetic corpus, trains ``TypedOperatorPolicyScorer`` on the representable
(``PRONOUN_FOCUS_FOLLOWUP``-only) subset's train split, and prints the real
diagnostic-probe report
(:func:`slm_training.harnesses.preference.replay_preference_typed_policy_adapter.evaluate_typed_policy_argument_probe`)
as JSON.

    python -m scripts.run_replay_preference_typed_policy_probe

This is fixture-scale, diagnostic-probe wiring evidence, never a certified
decision-path or ship-readiness claim -- see
``docs/design/dsh5-10-replay-preference-rows.md``'s "Eighth slice" and the
adapter module's own docstring for exactly what is and is not measured. It
completes in a small fraction of a second, far inside
``slm_training.levers.MAX_RUN_MINUTES``.
"""

from __future__ import annotations

import argparse
import json

from slm_training.harnesses.preference.replay_preference_typed_policy_adapter import (
    evaluate_typed_policy_argument_probe,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to also write the JSON report to.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Training random seed.")
    args = parser.parse_args(argv)

    report = evaluate_typed_policy_argument_probe(seed=args.seed)
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    print(payload)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
