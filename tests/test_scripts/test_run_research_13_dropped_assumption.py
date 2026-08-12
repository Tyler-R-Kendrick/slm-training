"""RESEARCH-13 runner smoke test."""

from __future__ import annotations

from scripts import run_research_13_dropped_assumption


def test_default_off_json() -> None:
    rc = run_research_13_dropped_assumption.main(["--json"])
    assert rc == 0
