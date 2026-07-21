"""Tests for scripts/run_termination_matrix.py CLI."""

from __future__ import annotations


from scripts.run_termination_matrix import main


def test_describe(capsys) -> None:
    assert main(["--describe"]) == 0
    captured = capsys.readouterr()
    assert "TerminationPolicy" in captured.out
    assert "explicit_stop" in captured.out


def test_plan_only_writes_report(tmp_path) -> None:
    out_dir = tmp_path / "runs"
    assert main(["--plan-only", "--output-dir", str(out_dir), "--k-values", "2", "4"]) == 0
    report = out_dir / "slm191_termination_matrix_report.json"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "TerminationManifestV1" in text
    assert "plan_only" in text


def test_exact_fixture_writes_report(tmp_path) -> None:
    out_dir = tmp_path / "runs"
    assert (
        main(
            [
                "--exact-fixture",
                "--output-dir",
                str(out_dir),
                "--n-samples",
                "3",
                "--k-values",
                "2",
                "--seed",
                "0",
                "--no-write-design-docs",
            ]
        )
        == 0
    )
    report = out_dir / "slm191_termination_matrix_report.json"
    assert report.exists()
    text = report.read_text(encoding="utf-8")
    assert "fixture" in text
    assert "explicit_stop" in text
