"""CLI tests for the proof-carrying goal-support fixture campaign."""

from __future__ import annotations

import json

from scripts.run_goal_support_domain_adequacy import main
from slm_training.harnesses.experiments.goal_support_domain_adequacy import (
    ARM_IDS,
    METRIC_IDS,
)


def test_plan_only_reports_locked_surface_without_writing(capsys) -> None:
    assert main(["--mode", "plan-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["arms"] == list(ARM_IDS)
    assert payload["metrics"] == list(METRIC_IDS)
    assert len(payload["manifest_sha256"]) == 64


def test_fixture_runs_twice_and_writes_durable_json_and_markdown(
    tmp_path, capsys
) -> None:
    json_path = tmp_path / "results.json"
    markdown_path = tmp_path / "results.md"
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--out-dir",
                str(tmp_path / "raw"),
                "--docs-out",
                str(json_path),
                "--docs-md-out",
                str(markdown_path),
            ]
        )
        == 0
    )
    stdout = json.loads(capsys.readouterr().out)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["determinism"]["match"] is True
    assert (
        payload["determinism"]["canonical_result_digest_a"]
        == payload["determinism"]["canonical_result_digest_b"]
        == stdout["canonical_result_digest"]
    )
    assert payload["version_stamp"]["stamp_schema"] == "version_stamp/v1"
    assert payload["recipe"]["suite_n"] == 4
    assert (tmp_path / "raw" / "proof-carrying-goal-support-fixture").is_dir()
    assert payload["checkpoint_created"] is False
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "fixture/diagnostic only" in markdown
    assert "No checkpoint was created" in markdown
