#!/usr/bin/env python3
"""Shard and finalize local Chromium evidence for a semantic-contrast corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from slm_training.data.store import write_common_manifest
from slm_training.data.verify import VerificationContext, run_preview_verifier_many, verify_record
from slm_training.dsl.schema import ExampleRecord
from slm_training.harness_core.versioning import build_version_stamp


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def verify_shard(dataset_dir: Path, start: int, count: int) -> Path:
    records = _read_rows(dataset_dir / "records.jsonl")
    selected = records[start : start + count]
    if not selected:
        raise ValueError("runtime shard is empty")
    sources = { _digest(str(item["record"]["openui"])): str(item["record"]["openui"]) for item in selected }
    runtime = dict(zip(sources, run_preview_verifier_many(tuple(sources.values()), timeout_s=160.0)))
    rows: list[dict[str, Any]] = []
    for item in selected:
        record = ExampleRecord.from_dict(item["record"])
        evidence = runtime[_digest(record.openui)]
        report = verify_record(
            record,
            VerificationContext(source_kind="program", runtime=evidence, require_runtime=True),
        )
        if not report.ok:
            raise RuntimeError(f"runtime verifier rejected {record.id}: {report.failing_gate}")
        rows.append({
            "record_id": record.id,
            "source_sha256": _digest(record.openui),
            "runtime_evidence": evidence.to_dict(),
            "verification": report.to_dict(),
        })
    destination = dataset_dir / "runtime_shards" / f"{start:06d}.jsonl"
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(destination, rows)
    return destination


def finalize(dataset_dir: Path) -> dict[str, Any]:
    records = _read_rows(dataset_dir / "records.jsonl")
    evidence: dict[str, dict[str, Any]] = {}
    for shard in sorted((dataset_dir / "runtime_shards").glob("*.jsonl")):
        for row in _read_rows(shard):
            record_id = str(row["record_id"])
            if record_id in evidence:
                if evidence[record_id]["source_sha256"] != row["source_sha256"]:
                    raise ValueError(f"conflicting runtime evidence for {record_id}")
                continue
            evidence[record_id] = row
    expected = {str(row["record"]["id"]): row["record"]["openui"] for row in records}
    if set(evidence) != set(expected):
        missing = sorted(set(expected) - set(evidence))
        raise ValueError(f"runtime evidence incomplete: {len(missing)} missing")
    if any(evidence[key]["source_sha256"] != _digest(source) for key, source in expected.items()):
        raise ValueError("runtime evidence source hash mismatch")

    def enrich(row: dict[str, Any]) -> dict[str, Any]:
        item = json.loads(json.dumps(row))
        proof = evidence[str(item["record"]["id"])]
        item["verifier_ok"] = True
        item["verifier_tier"] = proof["verification"]["tier"]
        item["meta"]["runtime_evidence"] = proof["runtime_evidence"]
        item["meta"]["verification"] = proof["verification"]
        return item

    enriched_records = [enrich(row) for row in records]
    pairs = _read_rows(dataset_dir / "pairs.jsonl")
    for pair in pairs:
        pair["positive"] = enrich(pair["positive"])
        pair["negative"] = enrich(pair["negative"])
    _write_rows(dataset_dir / "records.jsonl", enriched_records)
    _write_rows(dataset_dir / "pairs.jsonl", pairs)
    summary = json.loads((dataset_dir / "summary.json").read_text(encoding="utf-8"))
    summary["runtime_required"] = True
    summary["runtime_shards"] = len(list((dataset_dir / "runtime_shards").glob("*.jsonl")))
    summary["version_stamp"] = build_version_stamp("data.semantic_contrast", "evals.meaningful_program")
    (dataset_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = write_common_manifest(
        dataset_dir, kind="eval", dataset_id=str(summary["dataset_id"]),
        trace_id=f"semantic-contrast-{summary['dataset_id']}", immutable=False,
    )
    return {"records": len(records), "runtime_shards": summary["runtime_shards"], "manifest": manifest}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args(argv)
    result = finalize(args.dataset_dir) if args.finalize else verify_shard(args.dataset_dir, args.start, args.count)
    print(json.dumps(result if isinstance(result, dict) else {"shard": result.as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
