from pathlib import Path

from slm_training.harnesses.experiments.valid_edit_flow_closeout import render_markdown, run_closeout


def test_closeout_is_fail_closed_against_fixture_evidence() -> None:
    report = run_closeout(Path("."))
    payload = report.to_dict()
    assert payload["decision"] == "no_learned_flow_value_supported"
    assert payload["selected_stack"]["learned_objective"] == "none"
    assert all(item["sha256"] != "missing" for item in payload["artifact_lock"])
    assert any(row["classification"] == "reject" for row in payload["dispositions"])
    assert "default-off" in render_markdown(report)
