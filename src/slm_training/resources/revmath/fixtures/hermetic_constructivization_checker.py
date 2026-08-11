#!/usr/bin/env python3
"""Hermetic constructivization checker fixture (HARN-06).

Emits a single JSON trailer. Validates that the task statement is the
constructivized claim (not the original) and that weakening metadata is present.
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
    parser.add_argument("--original-statement-sha256", required=True)
    parser.add_argument("--constructivized-statement-sha256", required=True)
    parser.add_argument("--mode", default="auto")
    parser.add_argument(
        "--scenario",
        default="bounded_ok",
        choices=(
            "bounded_ok",
            "witness_ok",
            "oracle_ok",
            "remainder_ok",
            "masquerade",
            "timeout",
            "incomplete",
            "malformed",
        ),
    )
    parser.add_argument("--weakening-description", default="")
    parser.add_argument("--bound-ast-id", default="")
    parser.add_argument("--sleep-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)

    mode = args.mode
    if mode == "auto":
        if args.scenario == "timeout":
            mode = "sleep"
        elif args.scenario == "incomplete":
            mode = "incomplete"
        elif args.scenario == "malformed":
            mode = "malformed"
        elif args.scenario == "masquerade":
            mode = "masquerade"
        else:
            mode = "ok"

    if mode == "sleep":
        time.sleep(max(0.0, args.sleep_seconds))
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

    if mode == "malformed":
        print("not-json-constructivization-proof", flush=True)
        return 2

    if mode == "incomplete":
        print(
            json.dumps(
                {
                    "status": "incomplete",
                    "detail": "constructivization checker skipped remaining goals",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    if mode == "masquerade":
        digest = _sha(f"masquerade:{args.task_id}")
        print(
            json.dumps(
                {
                    "status": "masquerade",
                    "proof_sha256": digest,
                    "detail": "constructivized digest equals original",
                    "original_statement_sha256": args.original_statement_sha256,
                    "constructivized_statement_sha256": (
                        args.constructivized_statement_sha256
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    if args.statement_sha256 != args.constructivized_statement_sha256:
        digest = _sha(f"mismatch:{args.task_id}")
        print(
            json.dumps(
                {
                    "status": "malformed",
                    "detail": "task statement is not the constructivized claim",
                    "proof_sha256": digest,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 2
    if args.statement_sha256 == args.original_statement_sha256:
        digest = _sha(f"masquerade2:{args.task_id}")
        print(
            json.dumps(
                {
                    "status": "masquerade",
                    "proof_sha256": digest,
                    "detail": "task masquerades as original theorem",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    if not args.weakening_description.strip():
        print(
            json.dumps(
                {
                    "status": "malformed",
                    "detail": "missing weakening_description",
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 2

    digest = _sha(
        f"ok:{args.task_id}:{args.constructivized_statement_sha256}:"
        f"{args.weakening_description}"
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "proof_sha256": digest,
                "detail": "constructivized claim checked; original not claimed",
                "original_statement_sha256": args.original_statement_sha256,
                "constructivized_statement_sha256": (
                    args.constructivized_statement_sha256
                ),
                "weakening_description": args.weakening_description,
                "bound_ast_id": args.bound_ast_id or None,
                "masquerades_as_original": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
