"""CLI surface for finite controller-owned merge verification."""

import argparse
import json
from pathlib import Path


def add_release_parser(sub: argparse._SubParsersAction) -> None:
    command = sub.add_parser(
        "verify-release",
        help="Finite controller-owned resumable local verification; never promotion",
    )
    command.add_argument("--source", type=Path, default=Path.cwd())
    command.add_argument("--identity", help="locked independent verifier identity")
    command.add_argument("--activity-id", default="release-verification")
    command.add_argument("--state-dir", type=Path, required=True)
    command.add_argument("--job-id", default="release-verification")
    command.add_argument("--base-ref", default="origin/main")
    command.add_argument("--total-seconds", type=float, required=True)
    command.add_argument("--max-invocations", type=int, required=True)
    command.add_argument("--max-step-seconds", type=float, default=30)
    command.add_argument("--runtime-root", type=Path, action="append", default=[])
    command.add_argument("--local-feedback", action="store_true")
    command.add_argument(
        "--require-js-runtime",
        action="store_true",
        help="Require explicit complete bridge/SDK runtime grants before workloads",
    )
    command.set_defaults(func=cmd_verify_release)


def cmd_verify_release(args) -> int:
    from scripts.merge_verification_controller import verify_release

    try:
        result = verify_release(args)
    except ValueError as exc:
        result = {
            "status": "invalid_evidence",
            "reason": str(exc),
            "verification_complete": False,
            "release_authorized": False,
        }
    print(json.dumps(result, sort_keys=True))
    if result["verification_complete"]:
        return 0
    return 10 if result["status"] in {"runnable", "waiting_retry"} else 20
