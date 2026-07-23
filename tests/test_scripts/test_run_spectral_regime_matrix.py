"""CLI tests for SLM-216."""

from __future__ import annotations

import json

from scripts.run_spectral_regime_matrix import main


def test_cli_writes_json_and_markdown(tmp_path) -> None:
    json_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    output_dir = tmp_path / "run"
    assert main(
        [
            "--json",
            str(json_path),
            "--markdown",
            str(markdown_path),
            "--output-dir",
            str(output_dir),
            "--seeds",
            "0",
            "--null-draws",
            "3",
        ]
    ) == 0
    payload = json.loads(json_path.read_text())
    assert payload["schema"] == "SpectralRegimeReportV1"
    assert payload["gate"]["verdict"] == "inconclusive"
    assert markdown_path.is_file()
