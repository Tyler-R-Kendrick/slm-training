"""CLI coverage for the SLM-226 absolute spectral target gate."""

from __future__ import annotations

from scripts.run_absolute_spectral_gate import main


def test_cli_writes_and_checks_report(tmp_path) -> None:
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    args = [
        "--json",
        str(json_path),
        "--markdown",
        str(markdown_path),
        "--null-draws",
        "5",
        "--seeds",
        "0",
    ]
    assert main(args) == 0
    assert main([*args, "--check"]) == 0
