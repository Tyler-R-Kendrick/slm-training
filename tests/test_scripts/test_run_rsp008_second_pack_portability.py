"""CLI smoke for scripts.run_rsp008_second_pack_portability."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_rsp008_second_pack_portability import main


def test_plan_only_exits_zero(capsys) -> None:
    assert main(["--mode", "plan-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "plan_only"
    assert payload["manifest"]["experiment_id"] == "exp-sr-12"
    assert payload["claim_class_execution"] == "diagnostic"
    assert payload["promotion"] is False


def test_diagnostic_writes_docs(tmp_path: Path, capsys) -> None:
    docs_out = tmp_path / "iter-slm487-rsp-008.json"
    out_dir = tmp_path / "run"
    assert (
        main(
            [
                "--mode",
                "diagnostic",
                "--seeds",
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
    assert payload["kind"] == "rsp008_second_pack_portability_diagnostic/v1"
    assert payload["claim_class"] == "diagnostic"
    assert payload["promotion"] is False
    assert payload["catalogue_id"] == "exp-sr-12"
    assert "version_stamp" in payload
    assert payload["version_stamp"]["stamp_schema"] == "version_stamp/v1"
    assert docs_out.with_suffix(".md").is_file()
    captured = capsys.readouterr().out
    assert "second_pack_portability_parity_rate=" in captured
    assert payload["certified"] is False
    assert payload["analysis"]["falsifier_holds"] is True
