"""CLI tests for the SLM-218 zero-training retrospective."""

from __future__ import annotations

import json

from scripts.run_cross_attention_retention import main


def test_cli_writes_compact_fail_closed_report(tmp_path) -> None:
    json_path = tmp_path / "result.json"
    markdown_path = tmp_path / "result.md"
    assert main(["--json", str(json_path), "--markdown", str(markdown_path)]) == 0
    payload = json.loads(json_path.read_text())
    assert payload["schema"] == "CrossAttentionRetentionReportV1"
    assert payload["overall_verdict"] == "inconclusive"
    assert payload["ranked_candidates"] == []
    assert "No cross-attention role is ranked" in markdown_path.read_text()
