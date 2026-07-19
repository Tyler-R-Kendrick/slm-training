"""Tests for run_quality_matrix --matrix-set external-ceiling."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.run_quality_matrix as qm


def test_external_ceiling_describe(capsys) -> None:
    code = qm.main(
        [
            "--matrix-set",
            "external-ceiling",
            "--describe",
            "--checkpoint-reference-uri",
            "hf://buckets/TKendrick/OpenUI/checkpoints/x/ref.json",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["matrix_set"] == "external-ceiling"
    assert any(arm["arm_id"] == "B" for arm in data["arms"])


def test_external_ceiling_fixture_run(tmp_path: Path) -> None:
    code = qm.main(
        [
            "--matrix-set",
            "external-ceiling",
            "--mode",
            "fixture",
            "--run-root",
            str(tmp_path),
            "--checkpoint-reference-uri",
            "hf://buckets/TKendrick/OpenUI/checkpoints/x/ref.json",
        ]
    )
    assert code == 0
    assert (tmp_path / "external_ceiling_matrix_results.json").exists()
    assert (tmp_path / "external_ceiling_matrix_results.md").exists()
    payload = json.loads((tmp_path / "external_ceiling_matrix_results.json").read_text())
    assert payload["status"] == "fixture"
    assert payload["matrix_set"] == "external-ceiling"


def test_external_ceiling_requires_checkpoint_for_frontier(tmp_path: Path) -> None:
    code = qm.main(
        [
            "--matrix-set",
            "external-ceiling",
            "--mode",
            "fixture",
            "--run-root",
            str(tmp_path),
        ]
    )
    assert code == 1
