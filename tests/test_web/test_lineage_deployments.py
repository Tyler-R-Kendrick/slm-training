from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.lineage.records import ChampionPointer
from slm_training.lineage.store import LineageStore
from slm_training.web.comparisons import BlindedComparisonStore
from slm_training.web.deployments import DeploymentRegistry


def pointer() -> ChampionPointer:
    return ChampionPointer(
        pointer_id="pointer-a",
        track="twotower",
        run_id="champion-a",
        artifact_uri="model.onnx",
        manifest_sha="manifest",
        evaluation_report_sha="report",
        created_at="2026-01-01T00:00:00Z",
    )


def test_registry_reads_atomic_deployment_identity(tmp_path: Path) -> None:
    lineage = LineageStore(tmp_path / "lineage")
    lineage.deploy(pointer())
    registry = DeploymentRegistry(tmp_path / "lineage/deployments")
    assert registry.selected()["run_id"] == "champion-a"
    assert registry.tracks()["twotower"]["artifact_uri"] == "model.onnx"


def test_blinded_comparison_hides_identity_and_counts_votes(tmp_path: Path) -> None:
    store = BlindedComparisonStore(tmp_path / "comparisons.jsonl")
    pair = store.create(
        prompt="make a card",
        champion_run_id="champion",
        candidate_run_id="candidate",
        champion_openui="champion output",
        candidate_openui="candidate output",
        seed=0,
    )
    assert "candidate_run_id" not in pair
    assert "champion_run_id" not in pair
    vote = store.vote(pair["id"], "left", reviewer_id="reviewer")
    assert vote["blinded"] is True
    assert store.metrics("candidate")["total"] == 1
    with pytest.raises(ValueError, match="already voted"):
        store.vote(pair["id"], "right", reviewer_id="reviewer")
