#!/usr/bin/env python3
"""Hermetic assumption-ablation checker fixture (HARN-04).

Emits a single JSON trailer. Scenarios are keyed by --scenario (frozen in
fixture metadata). Modes: ok | refuted | incomplete | malformed | sleep |
hidden_ok (ok status with strength-bearing deps for audit to reject).
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
    parser.add_argument("--remaining-assumptions", default="")
    parser.add_argument("--removed-assumptions", default="")
    parser.add_argument(
        "--scenario",
        default="positive",
        choices=(
            "positive",
            "redundant",
            "necessary",
            "timeout",
            "hidden_import",
            "malformed",
            "incomplete",
        ),
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=("auto", "ok", "malformed", "incomplete", "refuted", "sleep", "hidden_ok"),
    )
    parser.add_argument("--sleep-seconds", type=float, default=30.0)
    parser.add_argument("--proof-deps", default="")
    args = parser.parse_args(argv)

    remaining = _split_ids(args.remaining_assumptions)
    removed = _split_ids(args.removed_assumptions)
    mode = args.mode
    if mode == "auto":
        if args.scenario == "timeout":
            mode = "sleep"
        elif args.scenario == "necessary" and removed:
            mode = "refuted"
        elif args.scenario == "hidden_import" and removed:
            mode = "hidden_ok"
        elif args.scenario == "malformed":
            mode = "malformed"
        elif args.scenario == "incomplete":
            mode = "incomplete"
        else:
            # positive / redundant: ok under any explored remaining set
            mode = "ok"

    if mode == "sleep":
        time.sleep(max(0.0, args.sleep_seconds))
        print(
            json.dumps(
                {
                    "status": "incomplete",
                    "detail": "sleep mode finished without interrupt",
                    "remaining_assumptions": remaining,
                    "removed_assumptions": removed,
                    "proof_dependency_ids": [],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    if mode == "malformed":
        print("not-json-ablation-proof", flush=True)
        return 2

    if mode == "incomplete":
        print(
            json.dumps(
                {
                    "status": "incomplete",
                    "detail": "ablation checker skipped remaining goals",
                    "remaining_assumptions": remaining,
                    "removed_assumptions": removed,
                    "proof_dependency_ids": [],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    if mode == "refuted":
        digest = _sha(f"refute:{args.task_id}:{args.statement_sha256}:{','.join(removed)}")
        print(
            json.dumps(
                {
                    "status": "refuted",
                    "proof_sha256": digest,
                    "refutation_digest": digest,
                    "detail": "necessary assumption missing",
                    "remaining_assumptions": remaining,
                    "removed_assumptions": removed,
                    "proof_dependency_ids": [],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    deps = _split_ids(args.proof_deps)
    if mode == "hidden_ok":
        # Appear successful while depending on a strength-bearing lemma that
        # reintroduces a removed assumption — validator must invalidate.
        if not deps:
            deps = ["lemma.hidden_choice"]
        digest = _sha(
            f"hidden:{args.task_id}:{args.statement_sha256}:{','.join(removed)}"
        )
        print(
            json.dumps(
                {
                    "status": "ok",
                    "proof_sha256": digest,
                    "detail": "hermetic ok with strength-bearing deps",
                    "remaining_assumptions": remaining,
                    "removed_assumptions": removed,
                    "proof_dependency_ids": deps,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    if not deps:
        deps = [f"ax.{aid}" for aid in remaining]
    proof = _sha(
        f"proof:{args.task_id}:{args.statement_sha256}:{','.join(remaining)}"
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "proof_sha256": proof,
                "detail": "hermetic ablation ok",
                "remaining_assumptions": remaining,
                "removed_assumptions": removed,
                "proof_dependency_ids": deps,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
