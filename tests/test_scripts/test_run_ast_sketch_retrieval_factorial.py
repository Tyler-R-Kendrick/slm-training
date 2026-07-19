"""Tests for scripts/run_ast_sketch_retrieval_factorial.py (SLM-133)."""

from __future__ import annotations

from pathlib import Path

from scripts import run_ast_sketch_retrieval_factorial


def test_plan_only_mode(tmp_path: Path) -> None:
    out = tmp_path / "plan"
    rc = run_ast_sketch_retrieval_factorial.main(
        ["--mode", "plan-only", "--output-dir", str(out), "--seeds", "0"]
    )
    assert rc == 0
    assert (out / "ast_sketch_retrieval_manifest.json").exists()
    assert (out / "ast_sketch_retrieval_report.json").exists()
    assert (out / "ast_sketch_retrieval_report.md").exists()


def test_fixture_mode_with_parent(tmp_path: Path) -> None:
    out = tmp_path / "fixture"
    rc = run_ast_sketch_retrieval_factorial.main(
        [
            "--mode",
            "fixture",
            "--output-dir",
            str(out),
            "--seeds",
            "0,1",
            "--include-controls",
            "--parent-checkpoint-uri",
            "hf://bucket/checkpoint/ref.json",
        ]
    )
    assert rc == 0
    report_text = (out / "ast_sketch_retrieval_report.json").read_text()
    assert "fixture" in report_text
    assert "slm133_fixture" in report_text


def test_frontier_mode_emits_partial_fixture(tmp_path: Path) -> None:
    out = tmp_path / "frontier"
    rc = run_ast_sketch_retrieval_factorial.main(
        [
            "--mode",
            "frontier",
            "--output-dir",
            str(out),
            "--seeds",
            "0",
            "--parent-checkpoint-uri",
            "hf://bucket/checkpoint/ref.json",
        ]
    )
    assert rc == 0
    assert (out / "ast_sketch_retrieval_report.json").exists()
