"""CLI smoke for scripts.run_sie008_voc_controller."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_sie008_voc_controller import main


def test_plan_only_exits_zero(capsys) -> None:
    assert main(["--mode", "plan-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "plan_only"
    assert payload["manifest"]["experiment_id"] == "exp-sr-5"
    assert payload["promotion"] is False


def test_fixture_writes_docs(tmp_path: Path, capsys) -> None:
    docs_out = tmp_path / "iter-slm484-sie-008.json"
    out_dir = tmp_path / "run"
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--n-records",
                "48",
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
    assert payload["kind"] == "sie008_voc_controller_fixture/v1"
    assert payload["claim_class"] == "fixture"
    assert payload["promotion"] is False
    assert "version_stamp" in payload
    assert payload["version_stamp"]["stamp_schema"] == "version_stamp/v1"
    assert payload["legal_support_parity"]["legal_support_parity_exact"] is True
    assert payload["recommendation"]["disposition"] in {
        "adopt_optional",
        "reject",
        "inconclusive",
    }
    assert docs_out.with_suffix(".md").is_file()
    captured = capsys.readouterr().out
    assert "compute_value_regret=" in captured
    assert "disposition=" in captured
