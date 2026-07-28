#!/usr/bin/env python3
"""Run the SLM-418 (DSH5-10) ninth-slice history-control representability probe.

Builds the representability report over the real 47-row (of 56)
non-``PRONOUN_FOCUS_FOLLOWUP`` synthetic-corpus subset this slice's sibling
``HistoryControlPolicyInputV1`` view can see, and reports whether any
same-domain (control-vs-control) chosen/rejected pair exists to train a
pairwise scorer over.

    python -m scripts.run_replay_preference_history_control_probe

Fixture-scale wiring evidence only -- never a certified decision-path or
ship-readiness claim. See ``docs/design/dsh5-10-replay-preference-rows.md``'s
"Ninth slice" and
``slm_training.harnesses.preference.replay_preference_history_control_policy``'s
module docstring for exactly what this does and does not measure. Completes
in a small fraction of a second, far inside
``slm_training.levers.MAX_RUN_MINUTES``.
"""

from __future__ import annotations

import argparse
import json

from slm_training.harnesses.preference.replay_preference_history_control_policy import (
    evaluate_history_control_representability,
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

    report = evaluate_history_control_representability()
    payload = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    print(payload)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
