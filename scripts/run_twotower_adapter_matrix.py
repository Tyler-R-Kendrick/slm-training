"""CLI for the LDI2-03 TwoTower adapter/full-update campaign matrix (SLM-126).

Thin wrapper over :mod:`slm_training.harnesses.preference.twotower_adapter_matrix`.
It reads a LDI2-02 (SLM-125) diagnostic report, resolves the authorization
decision, describes the matched T0–T5 arm matrix, and classifies each arm — it
runs no training. Only an ``authorized`` diagnostic permits trainable arms;
otherwise every trainable arm is blocked and the parent control expires. No
quality claim is produced here.

Examples::

    python scripts/run_twotower_adapter_matrix.py --describe \\
        --geometry-report outputs/data/ldi2_geometry/report.json --authorized-rank 8
    # or force a decision for a dry-run without a report:
    python scripts/run_twotower_adapter_matrix.py --decision no_safe_direction --authorized-rank 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from slm_training.harnesses.preference.twotower_adapter_matrix import (
    build_arms,
    describe_campaign,
    read_authorization,
    run_arm,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true", help="dry-run: emit the matrix only")
    parser.add_argument("--geometry-report", type=Path, default=None, help="LDI2-02 report JSON")
    parser.add_argument(
        "--decision",
        choices=("authorized", "repair_evidence", "no_safe_direction", "expired"),
        default=None,
        help="override the authorization decision (dry-run without a report)",
    )
    parser.add_argument("--authorized-rank", type=int, default=8)
    parser.add_argument("--lower-rank", type=int, default=None)
    parser.add_argument("--higher-rank", type=int, default=None)
    parser.add_argument("--corpus-admitted", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.decision is not None:
        decision, reason = args.decision, "explicit --decision override"
    elif args.geometry_report is not None:
        report = json.loads(args.geometry_report.read_text(encoding="utf-8"))
        decision, reason = read_authorization(report)
    else:
        # No report and no override: fail closed.
        decision, reason = read_authorization({})

    arms = build_arms(
        authorized_rank=args.authorized_rank,
        lower_rank=args.lower_rank,
        higher_rank=args.higher_rank,
    )
    payload: dict[str, Any] = describe_campaign(arms, decision=decision)
    payload["authorization_reason"] = reason

    if not args.describe:
        results = [
            run_arm(
                arm,
                decision=decision,
                corpus_admitted=args.corpus_admitted,
            ).as_dict()
            for arm in arms
        ]
        counts: dict[str, int] = {}
        for res in results:
            counts[res["status"]] = counts.get(res["status"], 0) + 1
        payload["results"] = results
        payload["status_counts"] = {k: counts[k] for k in sorted(counts)}

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
