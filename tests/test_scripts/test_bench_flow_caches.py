"""Tests for scripts/bench_flow_caches.py."""

from __future__ import annotations

from pathlib import Path

from scripts.bench_flow_caches import main


def test_describe(capsys) -> None:
    assert main(["--describe"]) == 0
    out = capsys.readouterr().out
    assert "SLM-193" in out
    assert "exact_closure_cold" in out


def test_plan_only(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    assert main(["--plan-only", "--output-dir", str(output_dir)]) == 0
    report = output_dir / "slm193_flow_caches_report.json"
    assert report.exists()
    text = report.read_text()
    assert "plan_only" in text
    assert "FlowCacheManifestV1" in text


def test_fixture_writes_report_and_design_docs(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    assert (
        main(
            [
                "--fixture",
                "--output-dir",
                str(output_dir),
                "--design-json",
                str(tmp_path / "design.json"),
                "--design-md",
                str(tmp_path / "design.md"),
                "--confirm",
            ]
        )
        == 0
    )
    report = output_dir / "slm193_flow_caches_report.json"
    assert report.exists()
    assert "fixture" in report.read_text()
    assert (tmp_path / "design.json").exists()
    assert (tmp_path / "design.md").exists()


def test_fixture_no_design_docs(tmp_path: Path) -> None:
    output_dir = tmp_path / "out"
    assert (
        main(
            [
                "--fixture",
                "--output-dir",
                str(output_dir),
                "--no-write-design-docs",
                "--confirm",
            ]
        )
        == 0
    )
    report = output_dir / "slm193_flow_caches_report.json"
    assert report.exists()
