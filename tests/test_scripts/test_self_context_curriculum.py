"""Tests for scripts/run_self_context_curriculum.py (SLM-300)."""

from __future__ import annotations

from pathlib import Path

from scripts import run_self_context_curriculum


def test_plan_only_mode(tmp_path: Path) -> None:
    out = tmp_path / "plan"
    rc = run_self_context_curriculum.main(
        ["--mode", "plan-only", "--output-dir", str(out), "--seeds", "0"]
    )
    assert rc == 0
    assert (out / "self_context_curriculum_manifest.json").exists()
    assert (out / "self_context_curriculum_report.json").exists()
    assert (out / "self_context_curriculum_report.md").exists()


def test_fixture_mode_with_parent(tmp_path: Path) -> None:
    out = tmp_path / "fixture"
    rc = run_self_context_curriculum.main(
        [
            "--mode",
            "fixture",
            "--output-dir",
            str(out),
            "--seeds",
            "0,1",
            "--self-context-rates",
            "0.0,0.10",
            "--parent-checkpoint-uri",
            "hf://bucket/checkpoint/ref.json",
        ]
    )
    assert rc == 0
    report_text = (out / "self_context_curriculum_report.json").read_text()
    assert "fixture" in report_text
    assert "slm300_fixture" in report_text


def test_invalid_rates_returns_error(tmp_path: Path) -> None:
    out = tmp_path / "bad"
    rc = run_self_context_curriculum.main(
        [
            "--mode",
            "plan-only",
            "--output-dir",
            str(out),
            "--self-context-rates",
            "0.10,0.25",
            "--seeds",
            "0",
        ]
    )
    assert rc == 1


def test_frontier_mode_emits_fixture_plan(tmp_path: Path) -> None:
    out = tmp_path / "frontier"
    rc = run_self_context_curriculum.main(
        [
            "--mode",
            "frontier",
            "--output-dir",
            str(out),
            "--seeds",
            "0",
            "--parent-checkpoint-uri",
            "hf://bucket/checkpoint/ref.json",
            "--checkpoint-bucket",
            "hf://bucket",
        ]
    )
    assert rc == 0
    report_text = (out / "self_context_curriculum_report.json").read_text()
    assert "slm300_frontier_partial" in report_text
