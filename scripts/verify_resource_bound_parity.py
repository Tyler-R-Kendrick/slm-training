#!/usr/bin/env python3
"""Verify KERN-10 / SLM-538 resource-bound + trigger parity fixtures.

Run: ``python -m scripts.verify_resource_bound_parity``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from slm_training.formal.resource_bound_parity import (
    SCHEMA,
    ParityMismatchError,
    UnsupportedConstructError,
    canonical_results_json,
    evaluate_all,
    fixture_path,
    load_fixtures,
    schema_path,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-canonical",
        type=str,
        default="",
        help="Optional path to write canonical results JSON",
    )
    args = parser.parse_args(argv)

    if not schema_path().is_file():
        print(f"missing schema: {schema_path()}", file=sys.stderr)
        return 2
    if not fixture_path().is_file():
        print(f"missing fixtures: {fixture_path()}", file=sys.stderr)
        return 2

    try:
        doc = load_fixtures()
        payload = evaluate_all(fixtures=doc)
    except (ParityMismatchError, UnsupportedConstructError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "schema": SCHEMA,
                "cases": len(doc["cases"]),
                "fixture_sha256": payload["fixture_sha256"],
                "results_sha256": payload["results_sha256"],
            },
            sort_keys=True,
        )
    )
    if args.write_canonical:
        Path(args.write_canonical).write_text(
            canonical_results_json(fixtures=doc) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
