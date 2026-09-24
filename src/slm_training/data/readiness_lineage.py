"""Content-bound data inputs and canonical leakage checks for repair readiness."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.data.decontam import apply_ngram_decontam
from slm_training.data.leakage import (
    find_leakage, fingerprint_openui, fingerprint_openui_structure,
    fingerprint_pair, fingerprint_prompt,
)
from slm_training.data.readiness_contract import (
    DataReadinessRequest, SnapshotInput, file_digest, scoped_path,
)
from slm_training.dsl.schema import ExampleRecord, load_jsonl


def snapshot_input(root: Path, directory: Path, *, exposure: str, records="records.jsonl"):
    """Bind actual published bytes; does not certify their quality."""
    directory = scoped_path(root, directory.relative_to(root).as_posix())
    manifest = directory / "manifest.json"
    payload = json.loads(manifest.read_text())
    return SnapshotInput(directory=directory.relative_to(root).as_posix(),
                         dataset_id=payload["dataset_id"], kind=payload["kind"],
                         records=records, manifest_sha256=file_digest(manifest),
                         records_sha256=file_digest(scoped_path(directory, records)), exposure=exposure)


def ancestor_snapshots(root: Path, lineage_root: str, starting_run_id: str) -> list[SnapshotInput]:
    """Resolve canonical registered-dataset lineage, including every parent.

    Dataset snapshot records_sha is commonly a dataset content fingerprint,
    not a raw JSONL SHA. Bind both using their actual owners, never compare
    those distinct units. Unsupported aggregate snapshots fail explicitly.
    """
    from slm_training.harness_core.lineage.store import LineageStore
    from slm_training.data.store import DataStore, dataset_fingerprint

    store = LineageStore(scoped_path(root, lineage_root))
    pending, seen, refs = [starting_run_id], set(), {}
    while pending:
        run_id = pending.pop()
        DataStore.validate_id(run_id)
        if run_id in seen:
            continue
        seen.add(run_id)
        run = store.load_run(run_id)
        snapshot = store.load_snapshot(run.data_snapshot_sha)
        if snapshot.sha != run.data_snapshot_sha or len(snapshot.sources) != 1:
            raise ValueError("unsupported or corrupt lineage snapshot")
        source = Path(snapshot.sources[0])
        source = source if source.is_absolute() else root / source
        ref = snapshot_input(root, source, exposure="train_only")
        snapshot_paths(root, ref)
        if snapshot.records_sha not in {ref.records_sha256, dataset_fingerprint(source)}:
            raise ValueError("lineage snapshot content identity mismatch")
        refs[ref.directory] = ref
        pending.extend(run.parent_ids)
    return list(refs.values())


def snapshot_paths(root: Path, ref: SnapshotInput) -> tuple[Path, Path]:
    directory = scoped_path(root, ref.directory)
    manifest_path = directory / "manifest.json"
    records_path = scoped_path(directory, ref.records)
    if file_digest(manifest_path) != ref.manifest_sha256:
        raise ValueError("snapshot manifest identity changed")
    if file_digest(records_path) != ref.records_sha256:
        raise ValueError("snapshot records identity changed")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("dataset_id") != ref.dataset_id or manifest.get("kind") != ref.kind:
        raise ValueError("snapshot role/dataset identity mismatch")
    for artifact in manifest.get("artifacts", []):
        path = scoped_path(directory, artifact["path"])
        if file_digest(path) != artifact["sha256"]:
            raise ValueError("snapshot artifact identity changed")
    if ref.exposure == "access_controlled":
        raise ValueError("readable workspace data has no demonstrated sealed isolation")
    return manifest_path, records_path


def read_snapshot(root: Path, ref: SnapshotInput) -> list[ExampleRecord]:
    return load_jsonl(snapshot_paths(root, ref)[1])


def verify_ancestry(root: Path, request: DataReadinessRequest) -> None:
    """Walk every parent from the canonical lineage, not caller hash inequality."""
    if request.initialization == "parent":
        required = ancestor_snapshots(root, request.lineage_root or "", request.starting_run_id or "")
        if any(ref not in request.training_ancestors for ref in required):
            raise ValueError("missing training ancestor records")
    for table in request.learned_tables:
        if file_digest(scoped_path(root, table.path)) != table.sha256:
            raise ValueError("learned table content identity mismatch")
        if any(source.exposure != "train_only" for source in table.training_sources):
            raise ValueError("learned table provenance is not train-only")


def _fingerprints(records: list[ExampleRecord]) -> dict[str, set[str]]:
    return {
        "ids": {r.id for r in records},
        "split_group_ids": {str(r.meta["split_group_id"]) for r in records
                            if r.meta.get("split_group_id")},
        "prompts": {fingerprint_prompt(r.prompt) for r in records},
        "openuis": {fingerprint_openui(r.openui) for r in records},
        "structures": {fingerprint_openui_structure(r.openui) for r in records},
        "pairs": {fingerprint_pair(r.prompt, r.openui) for r in records},
    }


def leakage_findings(
    train: list[ExampleRecord], suites: dict[str, list[ExampleRecord]],
) -> list[dict]:
    """Reuse exact/structural, family closure, and existing ngram thresholds."""
    from slm_training.data.splits import root_family_index

    findings = []
    for suite_id, rows in suites.items():
        fps = _fingerprints(rows)
        families = root_family_index([*train, *rows])
        reserved = {families[r.id] for r in rows}
        _, flagged = apply_ngram_decontam(train, {suite_id: rows})
        ngram_ids = {item["id"] for item in flagged}
        for record in train:
            reasons = find_leakage(record, fps)
            if families[record.id] in reserved:
                reasons.append("root_family")
            if record.id in ngram_ids:
                reasons.append("ngram")
            if reasons:
                findings.append({"id": record.id, "suite": suite_id,
                                 "reasons": sorted(set(reasons))})
    return findings


def request_leakage(
    root: Path, request: DataReadinessRequest, records: list[ExampleRecord],
) -> list[dict]:
    verify_ancestry(root, request)
    train = records if request.original.kind == "train" else []
    train = [*train, *(row for ref in request.training_ancestors
                       for row in read_snapshot(root, ref))]
    train.extend(row for table in request.learned_tables for ref in table.training_sources
                 for row in read_snapshot(root, ref))
    suites = {f"{ref.dataset_id}:{ref.records}": read_snapshot(root, ref)
              for ref in request.scored_suites}
    if request.original.kind == "eval":
        suites[request.successor_id] = records
    return leakage_findings(train, suites)
