#!/usr/bin/env python3
"""RESEARCH-19 / SLM-558: run default-off anytime-valid promotion pilot."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from slm_training.harnesses.experiments.research_19_lean_rademacher_bound import (
    ENABLE_FLAG,
    EVIDENCE_JSON,
    LOCK_RELPATH,
    load_campaign_lock,
    patch_preregistry,
    run_experiment,
    write_campaign_lock,
    write_evidence,
)

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-lock", action="store_true")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--enable", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.write_lock:
        lock = write_campaign_lock(ROOT / LOCK_RELPATH, root=ROOT)
        patch_preregistry(root=ROOT, lock_sha=lock.manifest_sha256, disposition="executable")
        print(f"wrote {LOCK_RELPATH} sha256={lock.manifest_sha256}")
        if not args.write:
            return 0

    enabled = args.enable or os.environ.get(ENABLE_FLAG, "").strip() in {
        "1", "true", "TRUE", "yes", "YES",
    }
    if args.write and not enabled:
        print(f"refusing --write: requires --enable or {ENABLE_FLAG}=1", file=sys.stderr)
        return 2

    lock_path = ROOT / LOCK_RELPATH
    if enabled and not lock_path.is_file():
        lock = write_campaign_lock(lock_path, root=ROOT)
        patch_preregistry(root=ROOT, lock_sha=lock.manifest_sha256, disposition="executable")
    if enabled:
        _ = load_campaign_lock(lock_path)

    result = run_experiment(root=ROOT, enabled=enabled)
    if args.write:
        write_evidence(result, root=ROOT)
        decision = result.get("decision")
        disposition = "completed" if decision == "accept" else "rejected" if decision == "reject" else "completed"
        patch_preregistry(root=ROOT, lock_sha=str(result.get("campaign_lock_sha256")), disposition=disposition)
        print(f"wrote {EVIDENCE_JSON}")
    if args.json or not args.write:
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
    else:
        print(f"decision={result.get('decision')} primary={result.get('primary_value')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
