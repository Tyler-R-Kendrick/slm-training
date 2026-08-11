#!/usr/bin/env python3
"""Hermetic reversal checker fixture (HARN-05).

Emits a single JSON trailer for one direction (forward or reverse).
Scenarios are keyed by --scenario (frozen in fixture metadata).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_ids(raw: str) -> list[str]:
    if not raw.strip():
        return []
    return sorted({part.strip() for part in raw.split(",") if part.strip()})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--statement-sha256", required=True)
    parser.add_argument("--direction", required=True, choices=("forward", "reverse"))
    parser.add_argument("--base-theory-id", required=True)
    parser.add_argument("--interpretation-status", required=True)
    parser.add_argument("--coding-id", default="")
    parser.add_argument("--antecedent-assumptions", default="")
    parser.add_argument(
        "--scenario",
        default="equivalence",
        choices=(
            "equivalence",
            "one_way_forward",
            "one_way_reverse",
            "timeout",
            "unsupported",
            "strength_mismatch",
            "hidden_stronger",
            "changed_proposition",
            "malformed",
            "incomplete",
        ),
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=(
            "auto",
            "ok",
            "refuted",
            "incomplete",
            "malformed",
            "sleep",
            "unsupported",
            "hidden_ok",
        ),
    )
    parser.add_argument("--sleep-seconds", type=float, default=30.0)
    parser.add_argument("--proof-deps", default="")
    args = parser.parse_args(argv)

    antecedents = _split_ids(args.antecedent_assumptions)
    direction = args.direction
    mode = args.mode
    if mode == "auto":
        if args.scenario == "timeout":
            mode = "sleep"
        elif args.scenario == "unsupported":
            mode = "unsupported"
        elif args.scenario == "malformed":
            mode = "malformed"
        elif args.scenario == "incomplete":
            mode = "incomplete"
        elif args.scenario == "one_way_forward":
            mode = "ok" if direction == "forward" else "refuted"
        elif args.scenario == "one_way_reverse":
            mode = "ok" if direction == "reverse" else "refuted"
        elif args.scenario == "hidden_stronger" and direction == "reverse":
            mode = "hidden_ok"
        elif args.scenario in (
            "equivalence",
            "strength_mismatch",
            "changed_proposition",
            "hidden_stronger",
        ):
            # strength_mismatch / changed_proposition appear ok at checker;
            # validator rejects equivalence.
            mode = "ok"
        else:
            mode = "ok"

    if mode == "sleep":
        time.sleep(max(0.0, args.sleep_seconds))
        print(
            json.dumps(
                {
                    "status": "incomplete",
                    "detail": "sleep mode finished without interrupt",
                    "direction": direction,
                    "proof_dependency_ids": [],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    if mode == "malformed":
        print("not-json-reversal-proof", flush=True)
        return 2

    if mode == "unsupported":
        print(
            json.dumps(
                {
                    "status": "incomplete",
                    "detail": "direction unsupported by checker",
                    "direction": direction,
                    "unsupported": True,
                    "proof_dependency_ids": [],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    if mode == "incomplete":
        print(
            json.dumps(
                {
                    "status": "incomplete",
                    "detail": "reversal checker skipped remaining goals",
                    "direction": direction,
                    "proof_dependency_ids": [],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    if mode == "refuted":
        digest = _sha(
            f"refute:{args.task_id}:{direction}:{args.statement_sha256}"
        )
        print(
            json.dumps(
                {
                    "status": "refuted",
                    "proof_sha256": digest,
                    "refutation_digest": digest,
                    "detail": f"one-way: {direction} fails",
                    "direction": direction,
                    "proof_dependency_ids": [],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    deps = _split_ids(args.proof_deps)
    if mode == "hidden_ok":
        if not deps:
            deps = ["lemma.hidden_stronger_principle"]
        digest = _sha(
            f"hidden:{args.task_id}:{direction}:{args.statement_sha256}"
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "proof_sha256": digest,
                    "detail": "hermetic ok with strength-bearing deps",
                    "direction": direction,
                    "antecedent_assumptions": antecedents,
                    "base_theory_id": args.base_theory_id,
                    "interpretation_status": args.interpretation_status,
                    "coding_id": args.coding_id,
                    "proof_dependency_ids": deps,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    if not deps:
        deps = [f"ax.{aid}" for aid in antecedents]
    proof = _sha(
        f"proof:{args.task_id}:{direction}:{args.statement_sha256}:"
        f"{','.join(antecedents)}"
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "proof_sha256": proof,
                "detail": "hermetic reversal ok",
                "direction": direction,
                "antecedent_assumptions": antecedents,
                "base_theory_id": args.base_theory_id,
                "interpretation_status": args.interpretation_status,
                "coding_id": args.coding_id,
                "proof_dependency_ids": deps,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
