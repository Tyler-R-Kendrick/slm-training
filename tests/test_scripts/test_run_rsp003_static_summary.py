"""CLI smoke for scripts.run_rsp003_static_summary."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_rsp003_static_summary import main


def test_plan_only_exits_zero(capsys) -> None:
    assert main(["--mode", "plan-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "plan_only"
    assert payload["manifest"]["experiment_id"] == "exp-sr-7"
    assert payload["claim_class_execution"] == "fixture"
    assert payload["promotion"] is False


def test_fixture_writes_docs(tmp_path: Path, capsys) -> None:
    docs_out = tmp_path / "iter-slm485-rsp-003.json"
    out_dir = tmp_path / "run"
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--cold-trials-per-arm",
                "1",
                "--warm-trials",
                "2",
                "--out-dir",
                str(out_dir),
                "--docs-out",
                str(docs_out),
            ]
        )
        == 0
    )
    payload = json.loads(docs_out.read_text(encoding="utf-8"))
    assert payload["kind"] == "rsp003_static_summary_fixture/v1"
    assert payload["claim_class"] == "fixture"
    assert payload["promotion"] is False
    assert payload["catalogue_id"] == "exp-sr-7"
    assert payload["static_summary_domain_parity_rate"] == 1.0
    assert payload["parity_analysis"]["certified"] is True
    assert "version_stamp" in payload
    assert payload["version_stamp"]["stamp_schema"] == "version_stamp/v1"
    assert docs_out.with_suffix(".md").is_file()
    captured = capsys.readouterr().out
    assert "static_summary_domain_parity_rate=1.0" in captured
