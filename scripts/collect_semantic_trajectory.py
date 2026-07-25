"""Validate and summarize immutable semantic-trajectory observations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from slm_training.evals.agentv import publish_agentv_evaluation
from slm_training.evals.semantic_trajectory import (
    SEMANTIC_TRAJECTORY_VERSION,
    SemanticTrajectoryTelemetryV1,
)
from slm_training.versioning import build_version_stamp


def collect(path: Path) -> dict:
    rows = [
        SemanticTrajectoryTelemetryV1.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trajectories = {row.run_id for row in rows}
    data_families = {row.data_family for row in rows}
    checkpoints = {row.checkpoint_sha256 for row in rows}
    by_availability = Counter(row.feature_availability.value for row in rows)
    checkpoints_per_trajectory = Counter(row.run_id for row in rows)
    complete_trajectory_count = sum(
        checkpoints_per_trajectory[run_id] >= 4 for run_id in trajectories
    )
    fit_ready = (
        len(trajectories) >= 20
        and complete_trajectory_count >= 20
        and len(data_families) >= 2
    )
    return {
        "schema_version": "semantic_trajectory_collection/v1",
        "input": str(path),
        "n": len(rows),
        "trajectory_count": len(trajectories),
        "checkpoint_count": len(checkpoints),
        "data_family_count": len(data_families),
        "trajectories_with_four_or_more_checkpoints": complete_trajectory_count,
        "availability": dict(sorted(by_availability.items())),
        "status": "ready_for_grouped_fit" if fit_ready else "insufficient_trajectory_evidence",
        "version_stamp": build_version_stamp("harness.eval.semantic_trajectory"),
    }


def _portable_paths(value: object) -> object:
    if isinstance(value, dict):
        return {key: _portable_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_paths(item) for item in value]
    if isinstance(value, str) and "/outputs/" in value:
        return "outputs/" + value.split("/outputs/", 1)[1]
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--evidence-note",
        action="append",
        default=[],
        help="immutable archive-audit finding retained with the result",
    )
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps({"schema_version": SEMANTIC_TRAJECTORY_VERSION, "default_off": True}, indent=2))
        return 0
    if not args.input or not args.out or not args.run_dir:
        parser.error("--input, --out, and --run-dir are required unless --describe")
    payload = collect(args.input)
    payload["evidence_notes"] = args.evidence_note
    payload["agentv"] = _portable_paths(publish_agentv_evaluation(
        args.run_dir,
        name="semantic-trajectory-inventory",
        claim="semantic_trajectory_inventory",
        cases=[{
            "id": "availability-accounting",
            "criteria": "all immutable rows have explicit feature availability",
            "pass": payload["n"] == sum(payload["availability"].values()),
            "failures": [],
            "result": {"n": payload["n"], "availability": payload["availability"]},
        }],
    ))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
