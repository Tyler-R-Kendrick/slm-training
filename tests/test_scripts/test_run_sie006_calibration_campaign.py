"""SIE-006 (SLM-483) regression tests: the EXP-SR-3 calibration campaign
runner script."""

from __future__ import annotations

import json

from scripts.run_sie006_calibration_campaign import main


def test_plan_only_mode_previews_the_catalogue_identity(capsys):
    exit_code = main(["--mode", "plan-only"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "plan_only"
    assert payload["claim_class"] == "diagnostic"


def test_diagnostic_mode_writes_durable_docs(tmp_path):
    docs_out = tmp_path / "iter-sie006.json"
    exit_code = main(
        [
            "--mode",
            "diagnostic",
            "--out-dir",
            str(tmp_path / "run"),
            "--docs-out",
            str(docs_out),
        ]
    )
    assert exit_code == 0
    payload = json.loads(docs_out.read_text())
    assert payload["catalogue_id"] == "exp-sr-3"
    assert payload["claim_class"] == "diagnostic"
    assert payload["promotion"] is False
    assert "version_stamp" in payload
    md_text = docs_out.with_suffix(".md").read_text()
    assert "Acceptance snapshot" in md_text
    assert "mock" in md_text
