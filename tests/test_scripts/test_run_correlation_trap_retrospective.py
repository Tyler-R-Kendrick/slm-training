"""CLI tests for the SLM-219 retrospective."""

from __future__ import annotations

import json

from scripts.run_correlation_trap_retrospective import main


def test_cli_writes_fail_closed_report(tmp_path) -> None:
    json_path = tmp_path / "result.json"
    markdown_path = tmp_path / "result.md"
    assert main(["--json", str(json_path), "--markdown", str(markdown_path)]) == 0
    payload = json.loads(json_path.read_text())
    assert payload["schema"] == "CorrelationTrapReportV1"
    assert payload["verdict"] == "inconclusive"
    assert payload["recommendation"] is None
    assert payload["trajectory_inventory"]["eligible_historical_trajectories"] == 0
    assert (
        payload["trajectory_inventory"]["actual_reproduction"]["checkpoint_count"] == 6
    )
    assert payload["warning_evaluation"]["true_positive"] == 1
    assert payload["warning_evaluation"]["false_negative"] == 2
    assert payload["warning_evaluation"]["false_positive_rate"] is None
    assert payload["weightwatcher_comparison"]["status"] == "completed"
    assert "not authorized as an early-stopping rationale" in markdown_path.read_text()


def test_cli_report_hash_is_stable_across_two_runs(tmp_path) -> None:
    hashes = []
    for index in range(2):
        json_path = tmp_path / f"result-{index}.json"
        markdown_path = tmp_path / f"result-{index}.md"
        assert main(["--json", str(json_path), "--markdown", str(markdown_path)]) == 0
        hashes.append(json.loads(json_path.read_text())["report_hash"])
    assert hashes[0] == hashes[1]
