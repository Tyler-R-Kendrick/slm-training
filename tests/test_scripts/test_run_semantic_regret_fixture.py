"""Tests for scripts/run_semantic_regret_fixture.py (SLM-143)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import run_semantic_regret_fixture


def test_fixture_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    out = tmp_path / "regret_report.json"
    rc = run_semantic_regret_fixture.main(["--out", str(out)])
    assert rc == 0
    assert out.exists()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema"] == "SemanticRegretReportV1"
    assert report["claim_class"] == "wiring"
    assert report["status"] == "fixture"
    assert "arms" in report
    assert "greedy" in report["arms"]
    assert "oracle" in report["arms"]
    assert "representation_unreachable" in report["arms"]
    assert "plan_deltas" in report
    assert "archetype" in report["plan_deltas"]
    assert "version_stamp" in report


def test_fixture_cli_prints_path(capsys, tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    rc = run_semantic_regret_fixture.main(["--out", str(out)])
    captured = capsys.readouterr()
    assert rc == 0
    assert str(out) in captured.out


def test_fixture_default_out_path_uses_date() -> None:
    # The default path is under outputs/runs/ and includes today's date. We
    # verify the CLI still exits 0 when --out is omitted; it writes to the real
    # outputs/runs directory, which is acceptable for a fixture run.
    rc = run_semantic_regret_fixture.main([])
    assert rc == 0
