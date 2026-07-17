from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.data.store import DataStore, write_common_manifest


def _dataset(path: Path, text: str = "{}\n") -> None:
    path.mkdir(parents=True)
    (path / "records.jsonl").write_text(text, encoding="utf-8")
    (path / "manifest.json").write_text(
        json.dumps({"kind": "train", "content_fingerprint": text.strip()}) + "\n",
        encoding="utf-8",
    )
    write_common_manifest(path, kind="train", dataset_id=path.name)


def test_local_first_and_conflict_detection(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    _dataset(store.path("train", "v1"))
    assert store.resolve("train", "v1").storage == "local"
    _dataset(store.published_path("train", "v1"))
    assert store.resolve("train", "v1").storage == "local"
    (store.published_path("train", "v1") / "manifest.json").write_text(
        json.dumps({"kind": "train", "content_fingerprint": "different"}) + "\n"
    )
    with pytest.raises(ValueError, match="differs"):
        store.resolve("train", "v1")


def test_publish_is_explicit_and_immutable(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    local = store.path("train", "ready")
    _dataset(local)
    telemetry = local / "synthesis_telemetry.jsonl"
    telemetry.write_text('{"records": 1}\n', encoding="utf-8")
    manifest_path = local / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["records"] = (local / "records.jsonl").as_posix()
    manifest["synthesis_telemetry"] = telemetry.as_posix()
    manifest_path.write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    write_common_manifest(local, kind="train", dataset_id="ready")

    published = store.publish("train", "ready")
    assert published.storage == "git"
    assert (published.path / "records.jsonl").is_file()
    assert (published.path / "synthesis_telemetry.jsonl").is_file()
    manifest = json.loads((published.path / "manifest.json").read_text())
    assert manifest["immutable"] is True
    assert manifest["records"] == (published.path / "records.jsonl").as_posix()
    assert manifest["synthesis_telemetry"] == (
        published.path / "synthesis_telemetry.jsonl"
    ).as_posix()
    with pytest.raises(FileExistsError):
        store.publish("train", "ready")


def test_migration_is_dry_until_applied(tmp_path: Path) -> None:
    legacy = tmp_path / "outputs" / "train_data"
    _dataset(legacy / "v1")
    store = DataStore(tmp_path)
    assert store.migration_plan()
    assert legacy.exists()
    store.migrate()
    assert not legacy.exists()
    assert store.resolve("train", "v1").storage == "local"
