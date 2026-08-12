"""RESEARCH-14 runner smoke test."""

from __future__ import annotations

from scripts.run_research_14_checker_diversity import main


def test_runner_default_off() -> None:
    assert main([]) == 0
