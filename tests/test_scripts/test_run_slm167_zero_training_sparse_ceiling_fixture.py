"""Tests for the SLM-167 (SDE1-05) zero-training sparse-action ceiling fixture CLI."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.run_slm167_zero_training_sparse_ceiling_fixture as _runner_module
from scripts.run_slm167_zero_training_sparse_ceiling_fixture import main


def test_plan_only_writes_manifest(tmp_path) -> None:
    assert main(["--mode", "plan-only", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "slm167_zero_training_sparse_ceiling_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "plan_only"
    assert data["claim_class"] == "wiring"
    assert data["schema"] == "Slm167ZeroTrainingSparseCeilingReportV1"
    assert "cells" in data
    cells = data["cells"]
    # 8 scoring methods × 2 decode settings × 3 default seeds.
    assert len(cells) == 48
    names = {c["arm_name"] for c in cells}
    assert "bi_encoder_similarity" in names
    assert "hybrid_retrieval_rerank" in names


def test_fixture_writes_design_docs(tmp_path) -> None:
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--output-dir",
                str(tmp_path),
                "--seeds",
                "0",
            ]
        )
        == 0
    )
    run_json = tmp_path / "slm167_zero_training_sparse_ceiling_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "fixture"
    assert data["claim_class"] == "wiring"
    assert data["rows"]
    assert data["version_stamp"]

    root = Path(_runner_module.__file__).resolve().parents[1]
    design_json = root / "docs/design/iter-slm167-zero-training-sparse-ceiling-20260720.json"
    design_md = root / "docs/design/iter-slm167-zero-training-sparse-ceiling-20260720.md"
    assert design_json.exists()
    assert design_md.exists()
    design_data = json.loads(design_json.read_text())
    assert design_data["status"] == "fixture"
    assert design_data["experiment_id"] == "slm167-zero-training-sparse-ceiling"
