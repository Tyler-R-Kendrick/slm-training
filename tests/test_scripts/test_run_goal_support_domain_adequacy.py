"""CLI smoke for scripts.run_goal_support_domain_adequacy."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_goal_support_domain_adequacy import main


def test_plan_only_exits_zero(capsys) -> None:
    assert main(["--mode", "plan-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "plan_only"
    assert payload["experiment_id"] == "pgs-h01"
    assert payload["promotion"] is False
    assert len(payload["fixture_ids"]) >= 14


def test_fixture_writes_docs(tmp_path: Path, capsys) -> None:
    docs_out = tmp_path / "iter-slm510-pgs-h01.json"
    out_dir = tmp_path / "run"
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--out-dir",
                str(out_dir),
                "--docs-out",
                str(docs_out),
            ]
        )
        == 0
    )
    payload = json.loads(docs_out.read_text(encoding="utf-8"))
    assert payload["kind"] == "goal_support_domain_adequacy_campaign_fixture/v1"
    assert payload["claim_class"] == "diagnostic"
    assert payload["promotion"] is False
    assert "version_stamp" in payload
    assert payload["version_stamp"]["stamp_schema"] == "version_stamp/v1"
    assert payload["falsifier"]["false_hard_prune_count"] == 0
    assert docs_out.with_suffix(".md").is_file()
    captured = capsys.readouterr().out
    assert "result_digest=" in captured
    assert "falsifier_holds=" in captured
