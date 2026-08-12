"""RESEARCH-16 runner smoke test."""

from __future__ import annotations

from scripts.run_research_16_rational_float import main


def test_runner_default_off() -> None:
    assert main([]) == 0
