from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_data_corpora import (
    _contrast_alpha_equivalent_duplicates,
    _contrast_fingerprints,
    _contrast_lineage_split_report,
    _contrast_ood_slices,
    _load_contrast_corpus,
    contrast_corpus_leakage_report,
    main,
)


def _plan_dict(
    *,
    symbol_ids: list[str],
    role_ids: list[str],
    provenance: str = "predicted",
    topology_edges: list[dict] | None = None,
) -> dict:
    return {
        "plan_version": "1",
        "identity": {"pack_id": "openui", "provenance": provenance},
        "archetype": {"id": "stack_column"},
        "role_slots": [{"role_id": r, "component_family": "text"} for r in role_ids],
        "topology": {"parent_relation_candidates": topology_edges} if topology_edges else {},
        "symbols": [{"symbol_id": s} for s in symbol_ids],
        "bindings": [
            {"role_slot_id": r, "candidate_symbols": [s]} for r, s in zip(role_ids, symbol_ids)
        ],
    }


def _contrast_record(
    record_id: str,
    *,
    source_program_id: str,
    split: str,
    symbol_ids: list[str],
    role_ids: list[str],
    transform_id: str = "content_swap_family",
    topology_edges: list[dict] | None = None,
) -> dict:
    return {
        "record": {"id": record_id},
        "role": "negative",
        "family": "content",
        "transform_id": transform_id,
        "transform_description": "test fixture",
        "severity": "moderate",
        "source_program_id": source_program_id,
        "source_plan": _plan_dict(
            symbol_ids=symbol_ids, role_ids=role_ids, topology_edges=topology_edges
        ),
        "verifier_ok": True,
        "verifier_tier": "Silver",
        "meaningful_report": {},
        "meta": {"split": split},
    }


def test_lineage_split_violation_detected_for_shared_source_program():
    records = [
        _contrast_record("r1", source_program_id="p1", split="train", symbol_ids=["a"], role_ids=["ra"]),
        _contrast_record("r2", source_program_id="p1", split="held_out", symbol_ids=["b"], role_ids=["rb"]),
        _contrast_record("r3", source_program_id="p2", split="train", symbol_ids=["c"], role_ids=["rc"]),
    ]
    report = _contrast_lineage_split_report(records)
    assert report["lineage_groups"] == 2
    violations = report["cross_split_lineage_violations"]
    assert len(violations) == 1
    assert violations[0]["source_program_id"] == "p1"
    assert set(violations[0]["record_ids"]) == {"r1", "r2"}


def test_no_lineage_violation_when_family_stays_in_one_split():
    records = [
        _contrast_record("r1", source_program_id="p1", split="train", symbol_ids=["a"], role_ids=["ra"]),
        _contrast_record("r2", source_program_id="p1", split="train", symbol_ids=["b"], role_ids=["rb"]),
    ]
    report = _contrast_lineage_split_report(records)
    assert report["cross_split_lineage_violations"] == []


def test_alpha_renamed_identity_collapses_to_same_fingerprint_within_split():
    # Same structural shape (one role, one symbol), different raw names.
    records = [
        _contrast_record(
            "r1", source_program_id="p1", split="train", symbol_ids=["sym_a"], role_ids=["role_a"]
        ),
        _contrast_record(
            "r2", source_program_id="p2", split="train", symbol_ids=["sym_totally_different"], role_ids=["role_zzz"]
        ),
    ]
    fingerprints = _contrast_fingerprints(records)
    assert fingerprints[0]["exact"] == fingerprints[1]["exact"]
    duplicates = _contrast_alpha_equivalent_duplicates(records, fingerprints)
    assert len(duplicates["within_split_duplicates"]) == 1
    assert duplicates["cross_split_duplicates"] == []
    assert duplicates["quarantined"] == []


def test_alpha_renamed_identity_across_splits_is_quarantined_and_prefers_train():
    records = [
        _contrast_record(
            "held_out_dup",
            source_program_id="p1",
            split="held_out",
            symbol_ids=["sym_a"],
            role_ids=["role_a"],
        ),
        _contrast_record(
            "train_original",
            source_program_id="p2",
            split="train",
            symbol_ids=["sym_renamed"],
            role_ids=["role_renamed"],
        ),
    ]
    fingerprints = _contrast_fingerprints(records)
    duplicates = _contrast_alpha_equivalent_duplicates(records, fingerprints)
    assert len(duplicates["cross_split_duplicates"]) == 1
    assert duplicates["within_split_duplicates"] == []
    assert len(duplicates["quarantined"]) == 1
    quarantined = duplicates["quarantined"][0]
    assert quarantined["record_id"] == "held_out_dup"
    assert quarantined["duplicate_of"] == "train_original"
    assert quarantined["reason"] == "alpha_equivalent_cross_split_duplicate"


def test_structurally_distinct_plans_do_not_collide():
    records = [
        _contrast_record("r1", source_program_id="p1", split="train", symbol_ids=["a"], role_ids=["ra"]),
        _contrast_record(
            "r2",
            source_program_id="p2",
            split="train",
            symbol_ids=["b", "c"],
            role_ids=["rb", "rc"],
        ),
    ]
    fingerprints = _contrast_fingerprints(records)
    assert fingerprints[0]["exact"] != fingerprints[1]["exact"]
    duplicates = _contrast_alpha_equivalent_duplicates(records, fingerprints)
    assert duplicates["within_split_duplicates"] == []
    assert duplicates["cross_split_duplicates"] == []


def test_unseen_topology_ood_slice_flags_held_out_only_topology():
    # One parent->child edge: a flat two-role topology.
    train_edges = [{"parent_role_id": "ra", "child_role_id": "rb", "relation": "contains"}]
    # A genuinely different shape: a three-role chain (two edges), not just
    # renamed roles over the same single-edge structure.
    held_out_edges = [
        {"parent_role_id": "rx", "child_role_id": "ry", "relation": "contains"},
        {"parent_role_id": "ry", "child_role_id": "rz", "relation": "contains"},
    ]
    records = [
        _contrast_record(
            "train_rec",
            source_program_id="p1",
            split="train",
            symbol_ids=["a", "b"],
            role_ids=["ra", "rb"],
            topology_edges=train_edges,
        ),
        _contrast_record(
            "held_out_novel_topology",
            source_program_id="p2",
            split="held_out",
            symbol_ids=["x", "y", "z"],
            role_ids=["rx", "ry", "rz"],
            topology_edges=held_out_edges,
        ),
    ]
    fingerprints = _contrast_fingerprints(records)
    ood = _contrast_ood_slices(records, fingerprints)
    assert ood["unseen_topology"] == ["held_out_novel_topology"]
    assert ood["unseen_source_family"] == ["held_out_novel_topology"]


def test_renamed_symbol_ood_slice_uses_transform_id():
    records = [
        _contrast_record(
            "r1",
            source_program_id="p1",
            split="held_out",
            symbol_ids=["a"],
            role_ids=["ra"],
            transform_id="binding_swap_symbol",
        ),
        _contrast_record(
            "r2",
            source_program_id="p2",
            split="held_out",
            symbol_ids=["b"],
            role_ids=["rb"],
            transform_id="content_swap_family",
        ),
    ]
    fingerprints = _contrast_fingerprints(records)
    ood = _contrast_ood_slices(records, fingerprints)
    assert ood["renamed_symbol"] == ["r1"]


def test_real_contrast_corpus_leakage_report_is_stable_and_read_only():
    corpus_dir = "src/slm_training/resources/data/eval/openui_hard_valid_v1"
    before = {
        str(path): path.stat().st_mtime_ns
        for path in _iter_files(corpus_dir)
    }

    report = contrast_corpus_leakage_report()

    after = {
        str(path): path.stat().st_mtime_ns
        for path in _iter_files(corpus_dir)
    }
    assert before == after  # audited without mutation

    assert report["dataset_id"] == "openui_hard_valid_v1"
    assert report["records_audited"] == 2412
    assert report["lineage"]["lineage_groups"] == 210
    assert report["lineage"]["cross_split_lineage_violations"] == []
    assert report["quarantined_record_count"] == 39


def _iter_files(root: str):
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            yield path


def test_contrast_audit_cli_mode_writes_durable_report(tmp_path):
    out_json = tmp_path / "contrast-audit.json"
    out_md = tmp_path / "contrast-audit.md"
    exit_code = main(
        [
            "--mode",
            "contrast-audit",
            "--contrast-out-json",
            str(out_json),
            "--contrast-out-md",
            str(out_md),
        ]
    )
    assert exit_code == 0
    report = json.loads(out_json.read_text())
    assert report["records_audited"] == 2412
    assert "OOD slices" in out_md.read_text()


def test_load_contrast_corpus_matches_manifest_and_verifies_hashes():
    records = _load_contrast_corpus()
    assert len(records) == 2412
    assert all("source_plan" in record for record in records)


def test_report_certification_writes_stamped_durable_result(tmp_path, monkeypatch):
    snapshot = tmp_path / "openui_verified_v1"
    snapshot.mkdir()
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "openui_verified_v1",
                "source_fingerprints": {"fixture": "abc"},
                "source_rows": 2,
                "core_records": 1,
                "tiers": {"Gold": 0, "Silver": 1, "Bronze": 0, "Quarantine": 1},
                "version_stamp": {"stamp_schema": "version_stamp/v1"},
            }
        )
    )
    (snapshot / "quality_report.json").write_text(json.dumps({"quarantine_rate": 0.5}))
    monkeypatch.chdir(tmp_path)

    assert main(["--mode", "report-certification", "--output-dir", str(snapshot)]) == 0
    report = json.loads(
        (tmp_path / "docs/design/iter-slm289-corpus-certification-20260725.json").read_text()
    )
    assert report["source_rows"] == 2
    assert report["version_stamp"]["stamp_schema"] == "version_stamp/v1"
