#!/usr/bin/env python3
"""Hermetic revmath checker fixture (HARN-03).

Emits a single JSON trailer line on stdout for the runner to parse.
Modes: ok | malformed | incomplete | refuted | sleep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--statement-sha256", required=True)
    parser.add_argument(
        "--mode",
        default="ok",
        choices=("ok", "malformed", "incomplete", "refuted", "sleep"),
    )
    parser.add_argument("--sleep-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)

    if args.mode == "sleep":
        time.sleep(max(0.0, args.sleep_seconds))
        # If the runner fails to interrupt, still report incomplete.
        print(
            json.dumps(
                {
                    "status": "incomplete",
                    "detail": "sleep mode finished without interrupt",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    if args.mode == "malformed":
        print("not-json-proof-bytes", flush=True)
        return 2

    if args.mode == "incomplete":
        print(
            json.dumps(
                {"status": "incomplete", "detail": "checker skipped remaining goals"},
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    if args.mode == "refuted":
        digest = _sha(f"refute:{args.task_id}:{args.statement_sha256}")
        print(
            json.dumps(
                {
                    "status": "refuted",
                    "proof_sha256": digest,
                    "refutation_digest": digest,
                    "detail": "hermetic counterexample",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    proof = _sha(f"proof:{args.task_id}:{args.statement_sha256}")
    print(
        json.dumps(
            {
                "status": "ok",
                "proof_sha256": proof,
                "detail": "hermetic forward ok",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
