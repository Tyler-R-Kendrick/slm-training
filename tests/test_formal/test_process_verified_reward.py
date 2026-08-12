"""RESEARCH-11 / SLM-574: process-verified reward backend tests."""

from __future__ import annotations

from slm_training.formal.process_verified_reward import (
    default_attempts,
    paired_reward_report,
    process_verified_reward,
    terminal_reward,
)


def test_corpus_has_five_attempts() -> None:
    assert len(default_attempts()) == 5


def test_leakage_control_uses_terminal_only() -> None:
    leakage = next(a for a in default_attempts() if a.leakage_fields_present)
    assert process_verified_reward(leakage) == terminal_reward(leakage)


def test_paired_report_runs() -> None:
    report = paired_reward_report()
    assert report["schema"] == "process_verified_reward_report/v1"
    assert "decision" in report
