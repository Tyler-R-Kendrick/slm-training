"""Dataset-build → lineage DataSnapshot registration tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.lineage.data_cycle import register_dataset_snapshot
from slm_training.lineage.store import LineageStore


def _train_dataset(tmp_path: Path, fingerprint: str) -> Path:
    dataset = tmp_path / "data" / "train" / "vtest"
    dataset.mkdir(parents=True)
    (dataset / "records.jsonl").write_text(
        '{"id": "r1", "prompt": "p", "openui": "root = Stack([x])", '
        '"placeholders": [], "split": "train", "source": "fixture", "meta": {}}\n',
        encoding="utf-8",
    )
    (dataset / "manifest.json").write_text(
        json.dumps(
            {
                "version": "vtest",
                "kind": "train_data",
                "profile": "strict",
                "record_count": 1,
                "content_fingerprint": fingerprint,
                "trace_id": "trace-1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return dataset


def test_registration_is_idempotent_per_content_fingerprint(tmp_path: Path) -> None:
    store = LineageStore(tmp_path / "lineage")
    dataset = _train_dataset(tmp_path, "a" * 64)

    snapshot, path, created = register_dataset_snapshot(
        store, dataset_dir=dataset, kind="train"
    )
    assert created is True
    assert snapshot.snapshot_id == "train-vtest"
    assert snapshot.records_sha == "a" * 64
    assert snapshot.record_count == 1
    assert snapshot.metadata["kind"] == "train"
    assert snapshot.metadata["profile"] == "strict"
    assert snapshot.metadata["trace_id"] == "trace-1"
    assert path.is_file()

    again, again_path, again_created = register_dataset_snapshot(
        store, dataset_dir=dataset, kind="train"
    )
    assert again_created is False
    assert again_path == path
    assert again.records_sha == snapshot.records_sha
    assert len(list((store.root / "data_snapshots").glob("train-vtest-*.json"))) == 1


def test_changed_content_registers_a_new_snapshot(tmp_path: Path) -> None:
    store = LineageStore(tmp_path / "lineage")
    dataset = _train_dataset(tmp_path, "a" * 64)
    register_dataset_snapshot(store, dataset_dir=dataset, kind="train")

    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    manifest["content_fingerprint"] = "b" * 64
    (dataset / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    snapshot, _, created = register_dataset_snapshot(
        store, dataset_dir=dataset, kind="train"
    )
    assert created is True
    assert snapshot.records_sha == "b" * 64
    assert len(list((store.root / "data_snapshots").glob("train-vtest-*.json"))) == 2


def test_eval_layout_counts_suite_records(tmp_path: Path) -> None:
    store = LineageStore(tmp_path / "lineage")
    dataset = tmp_path / "data" / "eval" / "vtest"
    for suite in ("smoke", "held_out"):
        suite_dir = dataset / "suites" / suite
        suite_dir.mkdir(parents=True)
        (suite_dir / "records.jsonl").write_text(
            '{"id": "%s1", "prompt": "p", "openui": "root = Stack([x])", '
            '"placeholders": [], "split": "%s", "source": "fixture", "meta": {}}\n'
            % (suite, suite if suite != "held_out" else "held_out"),
            encoding="utf-8",
        )
    (dataset / "manifest.json").write_text(
        json.dumps(
            {"version": "vtest", "kind": "test_data", "content_fingerprint": "e" * 64}
        )
        + "\n",
        encoding="utf-8",
    )

    snapshot, _, created = register_dataset_snapshot(
        store, dataset_dir=dataset, kind="eval"
    )
    assert created is True
    assert snapshot.snapshot_id == "eval-vtest"
    assert snapshot.record_count == 2


def test_missing_fingerprint_fails_closed(tmp_path: Path) -> None:
    store = LineageStore(tmp_path / "lineage")
    empty = tmp_path / "data" / "train" / "vempty"
    empty.mkdir(parents=True)
    with pytest.raises(ValueError, match="no content fingerprint"):
        register_dataset_snapshot(store, dataset_dir=empty, kind="train")
