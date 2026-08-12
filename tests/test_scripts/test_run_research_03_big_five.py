"""CLI smoke for RESEARCH-03 runner."""

from __future__ import annotations

from scripts.run_research_03_big_five import main


def test_cli_default_off_json(capsys) -> None:
    assert main(["--json"]) == 0
    out = capsys.readouterr().out
    assert "skipped_default_off" in out
    assert "RESEARCH-03" in out
