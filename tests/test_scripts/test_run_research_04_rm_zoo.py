"""Tests for scripts/run_research_04_rm_zoo.py."""

from __future__ import annotations

from scripts import run_research_04_rm_zoo


def test_default_off_json(capsys) -> None:
    rc = run_research_04_rm_zoo.main(["--json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "skipped_default_off" in out
