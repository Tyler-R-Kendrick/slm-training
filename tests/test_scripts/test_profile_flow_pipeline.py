"""Tests for scripts/profile_flow_pipeline.py CLI."""

from __future__ import annotations

from scripts.profile_flow_pipeline import main


def test_describe(capsys) -> None:
    assert main(["--describe"]) == 0
    captured = capsys.readouterr()
    assert "SLM-192" in captured.out
    assert "bridge_planner_canonical_greedy" in captured.out


def test_plan_only_writes_report(tmp_path) -> None:
    out_dir = tmp_path / "runs"
    assert main(["--plan-only", "--output-dir", str(out_dir)]) == 0
    report = out_dir / "slm192_profile_flow_pipeline_report.json"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "FlowPipelineManifestV1" in text
    assert "plan_only" in text


def test_fixture_writes_report(tmp_path) -> None:
    out_dir = tmp_path / "runs"
    assert (
        main(
            [
                "--fixture",
                "--output-dir",
                str(out_dir),
                "--n-repeats",
                "2",
                "--seed",
                "0",
                "--no-write-design-docs",
            ]
        )
        == 0
    )
    report = out_dir / "slm192_profile_flow_pipeline_report.json"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "fixture" in text
    assert "cost_profile_wired" in text
