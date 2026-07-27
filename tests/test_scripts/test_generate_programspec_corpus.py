"""ProgramSpec corpus generation CLI (SLM-267 VSD2-02): stream + wiring-fixture modes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.generate_programspec_corpus import main

_SMALL = ["--components", "TextContent,Button,Separator", "--max-depth", "3", "--max-width", "3"]


def _records(out_dir: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (out_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_describe_is_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    rc = main(
        ["--describe", "--target-unique-roots", "5", "--dataset-id", "describe_only", *_SMALL]
    )
    assert rc == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["resuming_from_accepted"] == 0
    assert not (tmp_path / "outputs" / "data" / "programspec" / "describe_only").exists()


def test_generation_is_deterministic_and_exact_deduped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset_id = "compiler_programs_uniform_test_v1"
    assert main(["--target-unique-roots", "5", "--dataset-id", dataset_id, *_SMALL]) == 0

    out_dir = tmp_path / "outputs" / "data" / "programspec" / dataset_id
    rows = _records(out_dir)
    assert len(rows) == 5
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids))  # exact dedup: no duplicate canonical roots

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["unique_roots"] == 5
    assert manifest["version_stamp"]["stamp_schema"] == "version_stamp/v1"

    # Re-running against an already-satisfied target is a no-op on the corpus.
    assert main(["--target-unique-roots", "5", "--dataset-id", dataset_id, *_SMALL]) == 0
    assert len(_records(out_dir)) == 5
    state = json.loads((out_dir / "state.json").read_text(encoding="utf-8"))
    assert state["accepted_ids"] == ids


def test_resume_produces_monotonic_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset_id = "compiler_programs_uniform_resume_v1"
    assert main(["--target-unique-roots", "3", "--dataset-id", dataset_id, *_SMALL]) == 0
    out_dir = tmp_path / "outputs" / "data" / "programspec" / dataset_id
    first_ids = [row["id"] for row in _records(out_dir)]
    assert len(first_ids) == 3

    assert main(["--target-unique-roots", "8", "--dataset-id", dataset_id, *_SMALL]) == 0
    second_ids = [row["id"] for row in _records(out_dir)]
    assert len(second_ids) == 8
    assert second_ids[:3] == first_ids  # resume replays, never reorders, the prefix
    assert len(set(second_ids)) == 8


def test_mismatched_resume_config_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset_id = "compiler_programs_uniform_mismatch_v1"
    assert main(["--target-unique-roots", "2", "--dataset-id", dataset_id, *_SMALL]) == 0
    with pytest.raises(SystemExit):
        main(
            [
                "--target-unique-roots",
                "2",
                "--dataset-id",
                dataset_id,
                "--seed",
                "7",
                *_SMALL,
            ]
        )


def test_exhaustion_is_detected_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    dataset_id = "compiler_programs_uniform_saturation_v1"
    assert (
        main(["--target-unique-roots", "100000", "--dataset-id", dataset_id, *_SMALL]) == 0
    )
    out_dir = tmp_path / "outputs" / "data" / "programspec" / dataset_id
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    state = json.loads((out_dir / "state.json").read_text(encoding="utf-8"))
    assert manifest["exhausted"] is True
    assert state["exhausted"] is True
    assert manifest["unique_roots"] == manifest["candidate_grid_size"]
    assert manifest["last_run_report"]["disposition"] == "saturated"


def test_plan_only_writes_manifest(tmp_path: Path) -> None:
    assert main(["--mode", "plan-only", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "slm267_programspec_coverage_scaling_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "plan_only"
    assert data["claim_class"] == "wiring"
    assert data["schema"] == "Slm267ProgramspecCoverageScalingManifestV1"
    assert data["version_stamp"]


def test_fixture_writes_design_docs(tmp_path: Path) -> None:
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--output-dir",
                str(tmp_path),
                "--target-count",
                "80",
                "--seed",
                "0",
                "--shards",
                "2",
            ]
        )
        == 0
    )
    run_json = tmp_path / "slm267_programspec_coverage_scaling_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "inconclusive"
    assert data["claim_class"] == "wiring"
    assert data["arms"]
    assert data["manifest_hash"]
