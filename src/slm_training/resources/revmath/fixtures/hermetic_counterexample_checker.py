#!/usr/bin/env python3
"""Hermetic computable/finite counterexample checker (HARN-06).

Emits a single JSON trailer. Refutation requires an independently checked,
replayable finite model or computable trace against the exact weakened
proposition + assumptions. Search failure / missing model → incomplete
(unknown upstream) — never refuted.
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
    parser.add_argument("--assumptions", default="")
    parser.add_argument(
        "--scenario",
        default="checked_refutation",
        choices=(
            "checked_refutation",
            "search_failed",
            "no_counterexample",
            "mismatched_proposition",
            "mismatched_assumptions",
            "timeout",
            "incomplete",
            "malformed",
        ),
    )
    parser.add_argument("--mode", default="auto")
    parser.add_argument("--model-digest", default="")
    parser.add_argument("--model-kind", default="")
    parser.add_argument("--model-payload", default="")
    parser.add_argument("--model-target-statement-sha256", default="")
    parser.add_argument("--model-target-assumptions", default="")
    parser.add_argument("--search-status", default="unknown")
    parser.add_argument("--sleep-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)

    assumptions = _split_ids(args.assumptions)
    model_assumps = _split_ids(args.model_target_assumptions)
    mode = args.mode
    if mode == "auto":
        if args.scenario == "timeout":
            mode = "sleep"
        elif args.scenario in ("search_failed", "no_counterexample", "incomplete"):
            mode = "incomplete"
        elif args.scenario == "malformed":
            mode = "malformed"
        elif args.scenario in ("mismatched_proposition", "mismatched_assumptions"):
            mode = "invalid"
        elif args.scenario == "checked_refutation":
            mode = "refuted"
        else:
            mode = "incomplete"

    if mode == "sleep":
        time.sleep(max(0.0, args.sleep_seconds))
        print(
            json.dumps(
                {
                    "status": "incomplete",
                    "detail": "sleep mode finished without interrupt",
                    "search_status": args.search_status,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    if mode == "malformed":
        print("not-json-counterexample-proof", flush=True)
        return 2

    if mode == "incomplete":
        # Explicit: search failure is incomplete, never a refutation.
        print(
            json.dumps(
                {
                    "status": "incomplete",
                    "detail": (
                        "proof search failed or no counterexample; "
                        "not a refutation (KERN-01)"
                    ),
                    "search_status": args.search_status or "failed",
                    "inferred_impossibility_from_search_failure": False,
                    "assumptions": assumptions,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    if mode == "invalid":
        digest = _sha(f"invalid:{args.task_id}:{args.model_digest}")
        print(
            json.dumps(
                {
                    "status": "malformed",
                    "proof_sha256": digest,
                    "detail": "counterexample does not match exact weakened theorem",
                    "model_digest": args.model_digest or None,
                    "model_target_statement_sha256": (
                        args.model_target_statement_sha256 or None
                    ),
                    "model_target_assumptions": model_assumps,
                    "task_statement_sha256": args.statement_sha256,
                    "task_assumptions": assumptions,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 2

    # Refutation path: require model targeting exact statement + assumptions.
    if (
        not args.model_digest
        or not args.model_payload
        or args.model_target_statement_sha256 != args.statement_sha256
        or model_assumps != assumptions
    ):
        print(
            json.dumps(
                {
                    "status": "incomplete",
                    "detail": (
                        "refutation refused: model missing or does not match "
                        "exact weakened theorem"
                    ),
                    "search_status": args.search_status,
                    "inferred_impossibility_from_search_failure": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0

    digest = args.model_digest
    print(
        json.dumps(
            {
                "status": "refuted",
                "proof_sha256": digest,
                "refutation_digest": digest,
                "detail": "checked finite/computable counterexample against weakened theorem",
                "model_kind": args.model_kind,
                "model_digest": digest,
                "model_payload": args.model_payload,
                "assumptions": assumptions,
                "independently_checked": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
