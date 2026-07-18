"""Run ↔ training-dataset join, quality report, and rejected-ledger endpoints."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from slm_training.web.app import create_app
from slm_training.web.observability import Readers

_FP = "f" * 64


def _seed_evidence(root: Path) -> None:
    dataset = root / "outputs" / "data" / "train" / "vq"
    dataset.mkdir(parents=True)
    (dataset / "records.jsonl").write_text(
        '{"id": "r1", "prompt": "Hero card", "openui": "root = Stack([x])", '
        '"placeholders": [], "split": "train", "source": "fixture", "meta": {}}\n',
        encoding="utf-8",
    )
    (dataset / "manifest.json").write_text(
        json.dumps(
            {
                "version": "vq",
                "kind": "train_data",
                "profile": "strict",
                "record_count": 1,
                "content_fingerprint": _FP,
                "trace_id": "trace-vq",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset / "stats.json").write_text(
        json.dumps({"record_count": 1, "profile": "strict"}) + "\n", encoding="utf-8"
    )
    (dataset / "quality_report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "strict",
                "counts": {
                    "admitted": 1,
                    "candidates": 4,
                    "rejected_total": 3,
                    "by_stage": {"normalize": 1, "quality": 1, "dedup": 1},
                },
                "constraint_fitness": {"parse_rate": 0.75, "judge_pass_rate": 1.0},
                "garbage": {"mean_quality_score": 0.9},
                "redundancy": {"dropped": {"exact_pair": 1, "fuzzy_minhash": 0}},
                "decontamination": {"ngram_flagged": 0},
                "engines": {"similarity": "minhash-char4gram"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (dataset / "rejected.jsonl").write_text(
        "\n".join(
            json.dumps(entry)
            for entry in (
                {"id": "bad1", "stage": "normalize", "reason": "parse_or_contract_error"},
                {"id": "bad2", "stage": "quality", "reason": "quality_gate_failed"},
                {"id": "bad3", "stage": "dedup", "reason": "exact_pair_duplicate"},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    run_dir = root / "outputs" / "runs" / "run-vq"
    run_dir.mkdir(parents=True)
    (run_dir / "train_summary.json").write_text(
        json.dumps(
            {
                "run_id": "run-vq",
                "train_dir": "outputs/data/train/vq",
                "data_manifest_sha": _FP,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    snapshots = root / "outputs" / "lineage" / "data_snapshots"
    snapshots.mkdir(parents=True)
    (snapshots / "train-vq-abc123.json").write_text(
        json.dumps(
            {
                "snapshot_id": "train-vq",
                "sources": ["outputs/data/train/vq"],
                "records_sha": _FP,
                "record_count": 1,
                "target_token_count": 4,
                "created_at": "2026-07-18T00:00:00Z",
                "metadata": {"kind": "train"},
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_run_training_data_joins_run_to_dataset(tmp_path: Path) -> None:
    _seed_evidence(tmp_path)
    readers = Readers(tmp_path)

    payload = readers.run_training_data("run-vq")
    assert payload["provenance"] == "live"
    assert payload["data_manifest_sha"] == _FP
    dataset = payload["dataset"]
    assert dataset is not None
    assert dataset["version"] == "vq"
    assert dataset["fingerprint"] == _FP
    assert dataset["fingerprint_matches_run"] is True
    assert dataset["record_count"] == 1
    assert dataset["profile"] == "strict"
    assert dataset["quality"]["rejected_total"] == 3
    assert dataset["quality"]["admission_rate"] == 0.25
    snapshot = payload["lineage_snapshot"]
    assert snapshot is not None and snapshot["snapshot_id"] == "train-vq"

    # The run detail payload embeds the same join.
    run_payload = readers.run("run-vq")
    assert run_payload["training_data"]["dataset"]["version"] == "vq"


def test_train_data_reverse_join_and_quality_summary(tmp_path: Path) -> None:
    _seed_evidence(tmp_path)
    readers = Readers(tmp_path)

    payload = readers.train_data("vq")
    assert payload["version"] == "vq"
    assert payload["profile"] == "strict"
    assert payload["quality"]["redundancy_dropped"] == 1
    assert payload["used_by_runs"] == ["run-vq"]


def test_train_quality_and_rejected_paging(tmp_path: Path) -> None:
    _seed_evidence(tmp_path)
    readers = Readers(tmp_path)

    quality = readers.train_quality("vq")
    assert quality["provenance"] == "live"
    assert quality["report"]["counts"]["rejected_total"] == 3
    assert quality["summary"]["profile"] == "strict"

    ledger = readers.train_rejected("vq", limit=2)
    assert ledger["count"] == 3
    assert len(ledger["rejected"]) == 2
    assert ledger["stages"] == ["dedup", "normalize", "quality"]

    staged = readers.train_rejected("vq", stage="quality")
    assert staged["count"] == 1
    assert staged["rejected"][0]["id"] == "bad2"

    missing = readers.train_quality("nope-version")
    assert missing["provenance"] == "missing" and missing["report"] is None


def test_routes_survive_cold_repo_state() -> None:
    with TestClient(create_app(execution=False)) as client:
        run_data = client.get("/api/runs/definitely-not-a-run/data")
        assert run_data.status_code == 200
        assert run_data.json()["dataset"] is None

        quality = client.get("/api/data/train/remediated/quality")
        assert quality.status_code == 200
        assert quality.json()["version"] == "remediated"

        rejected = client.get("/api/data/train/remediated/rejected")
        assert rejected.status_code == 200
        assert isinstance(rejected.json()["rejected"], list)
