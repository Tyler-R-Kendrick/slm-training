"""Synchronize one AbstractPlanningResultV1 to Linear idempotently (AP-038 / SLM-339).

Dry-run first: with no flags the CLI prints the resolved issue key, the planned
machine-owned evidence comment, and the idempotency marker — zero network calls.

Modes:

- default (dry run): print plan only; transport is never constructed.
- ``--offline-out <file>``: run the full sync through ``OfflineTransport`` and
  write the exact GraphQL payload(s) the live path would send, for CI review.
- ``--live``: perform the sync via ``LinearGraphQLTransport``; requires the
  API-key env var (default ``LINEAR_API_KEY``); missing → clean error, and the
  key is redacted from every error message.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from slm_training.harnesses.autoresearch.linear_sync import (
    DEFAULT_API_KEY_ENV,
    LinearGraphQLTransport,
    LinearSyncError,
    OfflineTransport,
    format_dry_run,
    sync_result,
)
from slm_training.harnesses.autoresearch.planning_result import (
    AbstractPlanningResultV1,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--result", required=True, type=Path, help="result JSON path")
    parser.add_argument("--live", action="store_true", help="perform the sync (default: dry run)")
    parser.add_argument(
        "--offline-out",
        type=Path,
        default=None,
        help="write the exact GraphQL payloads the live path would send (no network)",
    )
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help="env var holding the Linear API key (live mode only)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    result = AbstractPlanningResultV1.from_dict(payload)

    if not args.live and args.offline_out is None:
        report = sync_result(result, dry_run=True)
        print(format_dry_run(report))
        return 0

    if args.offline_out is not None and not args.live:
        chunks: list[str] = []
        transport = OfflineTransport(sink=chunks.append)
        report = sync_result(result, transport, dry_run=False)
        args.offline_out.parent.mkdir(parents=True, exist_ok=True)
        args.offline_out.write_text("".join(chunks), encoding="utf-8")
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        print(f"wrote {len(transport.payloads)} GraphQL payload(s) to {args.offline_out}")
        return 0

    try:
        transport = LinearGraphQLTransport(api_key_env=args.api_key_env)
    except LinearSyncError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    report = sync_result(result, transport, dry_run=False)
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 1 if report.action == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
