"""RESEARCH-12 runner smoke test."""

from __future__ import annotations

from scripts import run_research_12_successor_theorem


def test_default_off_json() -> None:
    rc = run_research_12_successor_theorem.main(["--json"])
    assert rc == 0
