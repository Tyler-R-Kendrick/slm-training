#!/usr/bin/env python3
"""Run the SLM-418 (DSH5-10) history-control representability probe (+ scorer).

Builds the representability report over the real 47-row (of 56)
non-``PRONOUN_FOCUS_FOLLOWUP`` synthetic-corpus subset this slice's sibling
``HistoryControlPolicyInputV1`` view can see, and reports whether any
same-domain (control-vs-control) chosen/rejected pair exists to train a
pairwise scorer over.

Tenth slice: when ``same_domain_pair_count > 0`` (the tenth slice's
``_pick_rejected`` same-domain-preferring change, ``dsl/operators/
replay_preference.py``), this script also trains and honestly reports the
pairwise linear scorer over ``HistoryControlPolicyInputV1.control_rows``
(``train_history_control_pairwise_scorer``) -- never forced when no
same-domain pairs exist.

    python -m scripts.run_replay_preference_history_control_probe

Fixture-scale wiring evidence only -- never a certified decision-path or
ship-readiness claim. See ``docs/design/dsh5-10-replay-preference-rows.md``'s
"Ninth slice"/"Tenth slice" and
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
    train_history_control_pairwise_scorer,
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

    probe = evaluate_history_control_representability()
    payload: dict = {"probe": probe.to_dict()}
    if probe.same_domain_pair_count > 0:
        scorer = train_history_control_pairwise_scorer()
        payload["scorer"] = scorer.to_dict()
    else:
        payload["scorer"] = None

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
