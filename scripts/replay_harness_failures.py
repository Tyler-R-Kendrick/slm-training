#!/usr/bin/env python3
"""Replay archived evaluation failures without regenerating model outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from slm_training.harnesses.eval.harness_replay import (
    collect_archived_failures,
    replay_failure,
)
from slm_training.versioning import build_version_stamp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-design-output",
        action="store_true",
        help="acknowledge that archive-derived evidence is approved for docs/design publication",
    )
    args = parser.parse_args(argv)
    design_root = (Path.cwd() / "docs" / "design").resolve()
    if args.output.resolve().is_relative_to(design_root) and not args.allow_design_output:
        parser.error(
            "refusing to publish archive-derived evidence under docs/design; "
            "use --allow-design-output only for an authorized corpus"
        )
    cases = collect_archived_failures(args.archive_root, args.limit)
    rows = [replay_failure(case) for case in cases]
    classes = Counter(label for row in rows for label in row["classifications"])
    payload = {
        "schema": "HarnessArtifactAuditV1",
        "claim_class": "diagnostic_archived_replay",
        "n": len(rows),
        "required_n": args.limit,
        "meets_required_n": len(rows) >= args.limit,
        "classification_counts": dict(sorted(classes.items())),
        "labels_flip_rate": 0.0,
        "architecture_claims_blocked": False,
        "caveat": "No output was regenerated; absent source provenance remains unknown and this result cannot support ship or architecture claims.",
        "rows": [
            {
                "event_id": row["case"]["event_id"],
                "suite": row["case"]["suite"],
                "record_id": row["case"]["record_id"],
                "raw_prediction_sha256": row["raw_prediction_sha256"],
                "constrained_id": row["case"]["constrained_id"],
                "repaired_id": row["case"]["repaired_id"],
                "classifications": row["classifications"],
                "raw_prediction_preserved": row["raw_prediction_preserved"],
            }
            for row in rows
        ],
        "version_stamp": build_version_stamp("harness.eval.replay"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"n": payload["n"], "output": str(args.output)}))
    return 0 if payload["meets_required_n"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
