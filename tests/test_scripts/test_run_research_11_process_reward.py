"""RESEARCH-11 runner smoke test."""

from __future__ import annotations

from scripts import run_research_11_process_reward


def test_default_off_json() -> None:
    rc = run_research_11_process_reward.main(["--json"])
    assert rc == 0
