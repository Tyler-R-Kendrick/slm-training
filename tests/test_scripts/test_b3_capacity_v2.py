"""Tests for scripts/run_b3_capacity_v2.py (SLM-124)."""

from __future__ import annotations

from pathlib import Path

from scripts import run_b3_capacity_v2


def test_plan_only_mode(tmp_path: Path) -> None:
    out = tmp_path / "plan"
    rc = run_b3_capacity_v2.main(
        ["--mode", "plan-only", "--output-dir", str(out), "--seeds", "0"]
    )
    assert rc == 0
    assert (out / "b3_capacity_v2_manifest.json").exists()
    assert (out / "b3_capacity_v2_report.json").exists()
    assert (out / "b3_capacity_v2_report.md").exists()


def test_fixture_mode_with_parent(tmp_path: Path) -> None:
    out = tmp_path / "fixture"
    rc = run_b3_capacity_v2.main(
        [
            "--mode",
            "fixture",
            "--output-dir",
            str(out),
            "--seeds",
            "0,1",
            "--widths",
            "64,128",
            "--parent-checkpoint-uri",
            "hf://bucket/checkpoint/ref.json",
        ]
    )
    assert rc == 0
    report_text = (out / "b3_capacity_v2_report.json").read_text()
    assert "fixture" in report_text
    assert "slm124_fixture" in report_text


def test_invalid_representation_returns_error(tmp_path: Path) -> None:
    out = tmp_path / "bad"
    rc = run_b3_capacity_v2.main(
        [
            "--mode",
            "plan-only",
            "--output-dir",
            str(out),
            "--representations",
            "lexer,bad_rep",
            "--seeds",
            "0",
        ]
    )
    assert rc == 1


def test_frontier_mode_emits_partial_fixture(tmp_path: Path) -> None:
    out = tmp_path / "frontier"
    rc = run_b3_capacity_v2.main(
        [
            "--mode",
            "frontier",
            "--output-dir",
            str(out),
            "--seeds",
            "0",
            "--widths",
            "64",
            "--parent-checkpoint-uri",
            "hf://bucket/checkpoint/ref.json",
        ]
    )
    assert rc == 0
    assert (out / "b3_capacity_v2_report.json").exists()
