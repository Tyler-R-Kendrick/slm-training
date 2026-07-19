"""Tests for scripts/run_corruption_curriculum.py (SLM-120)."""

from __future__ import annotations

from pathlib import Path

from scripts import run_corruption_curriculum


def test_plan_only_mode(tmp_path: Path) -> None:
    out = tmp_path / "plan"
    rc = run_corruption_curriculum.main(
        ["--mode", "plan-only", "--output-dir", str(out), "--seeds", "0"]
    )
    assert rc == 0
    assert (out / "corruption_curriculum_manifest.json").exists()
    assert (out / "corruption_curriculum_report.json").exists()
    assert (out / "corruption_curriculum_report.md").exists()


def test_fixture_mode_with_parent(tmp_path: Path) -> None:
    out = tmp_path / "fixture"
    rc = run_corruption_curriculum.main(
        [
            "--mode",
            "fixture",
            "--output-dir",
            str(out),
            "--seeds",
            "0,1",
            "--near-solved-shares",
            "0.0,0.10",
            "--parent-checkpoint-uri",
            "hf://bucket/checkpoint/ref.json",
        ]
    )
    assert rc == 0
    report_text = (out / "corruption_curriculum_report.json").read_text()
    assert "fixture" in report_text
    assert "slm120_fixture" in report_text


def test_invalid_shares_returns_error(tmp_path: Path) -> None:
    out = tmp_path / "bad"
    rc = run_corruption_curriculum.main(
        [
            "--mode",
            "plan-only",
            "--output-dir",
            str(out),
            "--near-solved-shares",
            "0.05,0.10",
            "--seeds",
            "0",
        ]
    )
    assert rc == 1
