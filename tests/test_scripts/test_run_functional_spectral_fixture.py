"""CLI tests for the SLM-217 functional-spectral fixture."""

from __future__ import annotations

import json

from scripts.run_functional_spectral_fixture import main


def test_cli_writes_versioned_honest_result(tmp_path) -> None:
    json_path = tmp_path / "result.json"
    markdown_path = tmp_path / "result.md"
    assert main(["--json", str(json_path), "--markdown", str(markdown_path)]) == 0
    payload = json.loads(json_path.read_text())
    assert payload["schema"] == "FunctionalSpectralReportV1"
    assert payload["verdict"] == "inconclusive"
    assert payload["checkpoint_references"] == []
    assert all(
        row["schema"] == "FunctionalSpectralSnapshotV1"
        for row in payload["snapshots"]
    )
    assert "no model checkpoint" in markdown_path.read_text()
