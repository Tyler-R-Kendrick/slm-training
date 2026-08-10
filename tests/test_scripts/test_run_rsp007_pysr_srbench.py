"""CLI smoke for scripts.run_rsp007_pysr_srbench."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_rsp007_pysr_srbench import main


def test_plan_only_exits_zero(capsys) -> None:
    assert main(["--mode", "plan-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "plan_only"
    assert payload["manifest"]["experiment_id"] == "exp-sr-11"
    assert payload["claim_class_execution"] == "diagnostic"
    assert payload["promotion"] is False
    assert payload["no_sota_claims"] is True
    assert payload["evidence_when_blocked"] == "external-blocked"


def test_fixture_writes_docs(tmp_path: Path, capsys) -> None:
    docs_out = tmp_path / "iter-slm486-rsp-007.json"
    out_dir = tmp_path / "run"
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--seed",
                "0",
                "--out-dir",
                str(out_dir),
                "--docs-out",
                str(docs_out),
            ]
        )
        == 0
    )
    payload = json.loads(docs_out.read_text(encoding="utf-8"))
    assert payload["kind"] == "rsp007_pysr_srbench_fixture/v1"
    assert payload["claim_class"] == "diagnostic"
    assert payload["promotion"] is False
    assert payload["no_sota_claims"] is True
    assert payload["catalogue_id"] == "exp-sr-11"
    assert "version_stamp" in payload
    assert payload["version_stamp"]["stamp_schema"] == "version_stamp/v1"
    assert docs_out.with_suffix(".md").is_file()
    captured = capsys.readouterr().out
    assert "srbench_matched_score_gap=" in captured
