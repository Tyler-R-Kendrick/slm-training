"""CLI smoke for scripts.run_rsp006_quality_diversity."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_rsp006_quality_diversity import main


def test_plan_only_exits_zero(capsys) -> None:
    assert main(["--mode", "plan-only"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "plan_only"
    assert payload["manifest"]["experiment_id"] == "exp-sr-10"
    assert payload["claim_class_execution"] == "fixture"
    assert payload["promotion"] is False
    assert payload["automatic_adoption"] is False


def test_fixture_writes_docs(tmp_path: Path, capsys) -> None:
    docs_out = tmp_path / "iter-slm481-rsp-006.json"
    out_dir = tmp_path / "run"
    assert (
        main(
            [
                "--mode",
                "fixture",
                "--generation-budget",
                "24",
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
    assert payload["kind"] == "rsp006_quality_diversity_fixture/v1"
    assert payload["claim_class"] == "fixture"
    assert payload["promotion"] is False
    assert payload["automatic_adoption"] is False
    assert payload["catalogue_id"] == "exp-sr-10"
    assert "corpus_topology_coverage" in payload
    assert "version_stamp" in payload
    assert payload["version_stamp"]["stamp_schema"] == "version_stamp/v1"
    assert docs_out.with_suffix(".md").is_file()
    captured = capsys.readouterr().out
    assert "corpus_topology_coverage=" in captured
