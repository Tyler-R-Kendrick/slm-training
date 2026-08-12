#!/usr/bin/env python3
"""RESEARCH-06 / SLM-564: run default-off VSS PBLean PB pilot.

    PYTHONPATH=src uv run python -m scripts.run_research_06_vss_pblean
    SLM_ENABLE_RESEARCH_06=1 PYTHONPATH=src uv run python -m scripts.run_research_06_vss_pblean --write
    PYTHONPATH=src uv run python -m scripts.run_research_06_vss_pblean --write-lock
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from slm_training.harnesses.experiments.research_06_vss_pblean import (
    ENABLE_FLAG,
    EVIDENCE_JSON,
    LOCK_RELPATH,
    load_campaign_lock,
    run_experiment,
    write_campaign_lock,
    write_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write-lock",
        action="store_true",
        help="write/refresh the preregistered CampaignLockV1 artifact",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="execute (requires enable) and write docs/design evidence",
    )
    parser.add_argument(
        "--enable",
        action="store_true",
        help=f"explicitly enable the default-off pilot (or set {ENABLE_FLAG}=1)",
    )
    parser.add_argument("--warm-repeats", type=int, default=5)
    parser.add_argument("--json", action="store_true", help="print result JSON")
    args = parser.parse_args(argv)

    if args.write_lock:
        lock = write_campaign_lock(ROOT / LOCK_RELPATH, root=ROOT)
        print(f"wrote {LOCK_RELPATH} sha256={lock.manifest_sha256}")
        if not args.write:
            return 0

    enabled = args.enable or os.environ.get(ENABLE_FLAG, "").strip() in {
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    }
    if args.write and not enabled:
        print(
            f"refusing --write: default-off pilot requires --enable or {ENABLE_FLAG}=1",
            file=sys.stderr,
        )
        return 2

    lock_path = ROOT / LOCK_RELPATH
    if enabled and not lock_path.is_file():
        write_campaign_lock(lock_path, root=ROOT)
    if enabled:
        _ = load_campaign_lock(lock_path)

    result = run_experiment(
        root=ROOT, enabled=enabled, warm_repeats=args.warm_repeats
    )
    if args.write:
        write_evidence(result, root=ROOT)
        print(f"wrote {EVIDENCE_JSON}")
    if args.json or not args.write:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    else:
        print(
            f"decision={result.get('decision')} "
            f"ratio={result.get('primary_value')} "
            f"disagreements={result.get('witness_disagreement_count')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
