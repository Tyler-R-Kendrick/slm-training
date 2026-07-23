#!/usr/bin/env python3
"""Build and validate replay-exact legal-edit bridge corpora."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from slm_training.data.flow.bridge_corpus import (
    MANIFEST_SCHEMA,
    PLANNER_MANIFEST_SCHEMA,
    build_bridge_rows,
    canonical_json,
    content_digest,
    load_corpus,
    validate_rows,
    write_corpus,
)
from slm_training.data.store import write_common_manifest
from slm_training.harness_core.versioning import build_version_stamp

_FIXTURE_ROOT = Path(
    "src/slm_training/resources/data/train/slm196_legal_edit_bridge_fixture"
)
_FIXTURE_INPUTS = Path("tests/fixtures/slm196_legal_edit_bridge")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _check_planner_manifest(
    payload: dict[str, Any], *, fixture: bool
) -> dict[str, Any]:
    if payload.get("schema") != PLANNER_MANIFEST_SCHEMA:
        raise ValueError("planner manifest must use LegalEditPlannerManifestV1")
    allowed_status = {"dev_fixture"} if fixture else {"frozen_selected"}
    if payload.get("status") not in allowed_status:
        raise ValueError("planner manifest is not a frozen selected policy")
    if payload.get("confirmation_allowed") is not False:
        raise ValueError("confirmation bridge rows are forbidden")
    if not payload.get("claim_manifest_digest"):
        raise ValueError("planner manifest is not bound to the SLM-184 claim ledger")
    floor = float(payload.get("reachability_floor", 0.95))
    if float(payload.get("reachability_rate", 0.0)) < floor:
        raise ValueError("planner reachability is below the frozen publication floor")
    if float(payload.get("transition_replay_rate", 0.0)) != 1.0:
        raise ValueError("planner transition replay is not 100%")
    if payload.get("selected_arm") != "canonical_greedy":
        raise ValueError("this builder only supports the replayed selected planner arm")
    return payload


def _diagnostics(rows: list[Any], failures: list[str]) -> dict[str, Any]:
    sizes = [len(row.complete_candidate_ids) for row in rows]
    positives = [len(row.positive_candidate_ids) for row in rows]
    actions = Counter(
        row.cost_profile.get("planner_selected_action", "UNKNOWN") for row in rows
    )
    source_policies = Counter(row.source_policy for row in rows)
    progress = [row.normalized_progress for row in rows]
    clusters: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        clusters.setdefault(row.target_cluster_id, []).append(
            {
                "row_id": row.row_id,
                "candidate_count": len(row.complete_candidate_ids),
                "positive_count": len(row.positive_candidate_ids),
                "progress": row.normalized_progress,
            }
        )
    return {
        "independent_targets": len({row.target_state_fingerprint for row in rows}),
        "target_clusters": len({row.target_cluster_id for row in rows}),
        "paths": len({row.bridge_id for row in rows}),
        "rows": len(rows),
        "unique_state_fingerprints": len({row.state_fingerprint for row in rows}),
        "unique_candidate_sets": len({row.candidate_set_digest for row in rows}),
        "candidate_set_size": {
            "min": min(sizes, default=0),
            "max": max(sizes, default=0),
            "mean": sum(sizes) / len(sizes) if sizes else 0.0,
            "histogram": dict(sorted(Counter(sizes).items())),
        },
        "multi_positive_rate": (
            sum(value > 1 for value in positives) / len(positives) if positives else 0.0
        ),
        "target_cluster_icc_inputs": clusters,
        "planner_failure_reasons": dict(sorted(Counter(failures).items())),
        "bridge_lengths": dict(sorted(Counter(row.bridge_length for row in rows).items())),
        "progress_values": progress,
        "singleton_ratio": (
            sum(value == 1 for value in sizes) / len(sizes) if sizes else 0.0
        ),
        "unknown_prevalence": (
            sum(len(row.unknown_candidate_ids) for row in rows) / sum(sizes)
            if sum(sizes)
            else 0.0
        ),
        "source_policy_breakdown": dict(sorted(source_policies.items())),
        "edit_family_coverage": dict(sorted(actions.items())),
        "coverage_comparison": {
            "x22_local_corruption": {
                "status": "unavailable",
                "delta": None,
                "reason": "no hash-pinned X22 baseline manifest supplied",
            },
            "solver_traces": {
                "status": "unavailable",
                "delta": None,
                "reason": "no hash-pinned solver-trace manifest supplied",
            },
        },
    }


def _write_auxiliary(
    output: Path,
    quality: dict[str, Any],
    diagnostics: dict[str, Any],
    failures: list[str],
) -> None:
    (output / "quality_report.json").write_text(
        json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "rejected.jsonl").write_text(
        "".join(
            canonical_json({"schema": "LegalEditBridgeRejectionV1", "reason": reason})
            + "\n"
            for reason in failures
        ),
        encoding="utf-8",
    )
    feedback = {
        "schema": "LegalEditBridgeSynthesisFeedbackV1",
        "status": "pass" if not failures else "repair_required",
        "recommendations": (
            []
            if not failures
            else [
                "Repair the named planner/edit-algebra producer; do not weaken corpus gates."
            ]
        ),
        "experiment_candidates": [],
        "diagnostics": diagnostics,
    }
    (output / "synthesis_feedback.json").write_text(
        json.dumps(feedback, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build(
    records_path: Path,
    planner_manifest_path: Path,
    output: Path,
    *,
    fixture: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    planner = _check_planner_manifest(
        json.loads(planner_manifest_path.read_text(encoding="utf-8")),
        fixture=fixture,
    )
    records = _load_jsonl(records_path)
    pins = dict(planner.get("version_pins") or {})
    rows = []
    candidate_sets = {}
    failures: list[str] = []
    for record in records:
        try:
            built_rows, built_sets = build_bridge_rows(
                record, version_pins=pins, max_edits=int(planner.get("max_edits", 12))
            )
            rows.extend(built_rows)
            candidate_sets.update(built_sets)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{record.get('id', 'UNKNOWN')}:{type(exc).__name__}:{exc}")
    validation = validate_rows(rows, candidate_sets)
    diagnostics = _diagnostics(rows, failures)
    elapsed = time.perf_counter() - started
    stamp = build_version_stamp(
        "data.flow.legal_edit_bridge_corpus", "model.legal_edit_batch"
    )
    stamp["stamped_at"] = str(planner.get("created_at", stamp["stamped_at"]))
    publishable = (
        not fixture
        and not failures
        and validation["replay_rate"] == 1.0
        and validation["candidate_reconstruction_rate"] == 1.0
    )
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "fixture_complete" if fixture and not failures else (
            "publishable" if publishable else "blocked"
        ),
        "claim_class": "wiring_fixture" if fixture else "production_corpus",
        "dataset_id": output.name,
        "records_source": {
            "path": records_path.as_posix(),
            "sha256": _sha256(records_path),
            "count": len(records),
        },
        "planner_manifest": {
            "path": planner_manifest_path.as_posix(),
            "sha256": _sha256(planner_manifest_path),
            "selected_arm": planner["selected_arm"],
            "source_policy": planner["source_policy"],
            "claim_manifest_digest": planner["claim_manifest_digest"],
        },
        "content_fingerprint": content_digest(
            [row.to_dict() for row in sorted(rows, key=lambda item: item.row_id)]
        ),
        "counts": {
            "records": len(records),
            "rows": len(rows),
            "candidate_sets": len(candidate_sets),
            "failures": len(failures),
        },
        "validation": validation,
        "diagnostics": diagnostics,
        "construction": {
            "wall_seconds": elapsed,
            "cache_reuse": len(rows) - len(candidate_sets),
        },
        "task_balance": dict(
            sorted(Counter(row.program_family for row in rows).items())
        ),
        "split_policy": "target_cluster_sha20_train_dev",
        "confirmation_rows": 0,
        "publishable": publishable,
        "honest_caveats": [
            "Fixture evidence validates wiring only; it is not production bridge coverage.",
            "X22 and solver coverage deltas remain unavailable until hash-pinned baselines are supplied.",
            "No confirmation rows, model loss, checkpoint, or winner selection is included.",
        ],
        "version_stamp": stamp,
    }
    write_corpus(output, rows, candidate_sets, manifest)
    quality = {
        "schema": "LegalEditBridgeQualityReportV1",
        "status": "pass" if not failures else "fail",
        "publishable": publishable,
        **validation,
        "reachability_rate": (len(records) - len(failures)) / len(records) if records else 0.0,
        "failure_reasons": failures,
        "unknown_is_negative": False,
        "confirmation_rows": 0,
    }
    _write_auxiliary(output, quality, diagnostics, failures)
    manifest = write_common_manifest(
        output,
        kind="train",
        dataset_id=output.name,
        trace_id=f"slm196-{content_digest(manifest)[:16]}",
        immutable=fixture,
    )
    if failures:
        raise ValueError(f"bridge corpus build rejected {len(failures)} record(s)")
    return manifest


def describe() -> str:
    return """\
LegalEditBridgeRowV1 / ExactLegalEditCandidateSetV1

Build:
  python -m scripts.build_legal_edit_bridges --records RECORDS.jsonl \
    --planner-manifest PLANNER.json --output outputs/data/train/DATASET

The exact candidate set is content-addressed and reconstructed during validation.
Candidate IDs hash semantic CanonicalEdit fields, request-local pointers remain
features rather than vocabulary rows, labels are multi-positive, and UNKNOWN is
kept separate from negatives. Production builds require a frozen SLM-184-bound
planner manifest; --fixture accepts only the committed dev-fixture manifest.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--records", type=Path)
    parser.add_argument("--planner-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args(argv)
    if args.describe:
        print(describe())
        return 0
    if args.fixture:
        args.records = args.records or (_FIXTURE_INPUTS / "records.jsonl")
        args.planner_manifest = args.planner_manifest or (
            _FIXTURE_INPUTS / "planner_manifest.json"
        )
        args.output = args.output or _FIXTURE_ROOT
    if args.validate or args.stats:
        if args.output is None:
            parser.error("--validate/--stats require --output")
        rows, candidate_sets, manifest = load_corpus(args.output)
        validation = validate_rows(rows, candidate_sets)
        if args.stats:
            print(json.dumps(manifest["diagnostics"], indent=2, sort_keys=True))
        else:
            print(json.dumps(validation, indent=2, sort_keys=True))
        return 0
    if not args.records or not args.planner_manifest or not args.output:
        parser.error("build requires --records, --planner-manifest, and --output")
    manifest = build(
        args.records, args.planner_manifest, args.output, fixture=args.fixture
    )
    print(json.dumps(manifest["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
