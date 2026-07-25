"""CLI tests for the SLM-220 fixture report."""

from __future__ import annotations

import json

from scripts.run_causal_subspace_fixture import main


def test_cli_writes_fail_closed_report(tmp_path) -> None:
    json_path = tmp_path / "result.json"
    markdown_path = tmp_path / "result.md"
    assert main(["--json", str(json_path), "--markdown", str(markdown_path)]) == 0
    payload = json.loads(json_path.read_text())
    assert payload["schema"] == "CausalSubspaceReportV1"
    assert payload["verdict"] == "rejected"
    assert payload["eligible_perturbation_bands"] == []
    assert payload["inventory"]["current_checkpoint_retrospective_eligible"] is False
    assert max(row["exact_jvp_abs_error"] for row in payload["snapshots"]) < 1e-12
    assert "Eligible matrices/bands" in markdown_path.read_text()


def test_cli_hash_is_deterministic(tmp_path) -> None:
    hashes = []
    for index in range(2):
        json_path = tmp_path / f"result-{index}.json"
        markdown_path = tmp_path / f"result-{index}.md"
        assert main(["--json", str(json_path), "--markdown", str(markdown_path)]) == 0
        hashes.append(json.loads(json_path.read_text())["report_hash"])
    assert hashes[0] == hashes[1]
