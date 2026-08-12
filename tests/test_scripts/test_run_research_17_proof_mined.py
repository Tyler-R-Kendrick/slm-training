"""RESEARCH-17 runner smoke test."""

from __future__ import annotations

from scripts import run_research_17_proof_mined


def test_default_off_json() -> None:
    rc = run_research_17_proof_mined.main(["--json"])
    assert rc == 0
