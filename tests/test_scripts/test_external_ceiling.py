"""Tests for scripts/run_external_ceiling.py."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.run_external_ceiling as runner


def test_fixture_run_writes_report(tmp_path: Path) -> None:
    code = runner.main(
        [
            "--mode",
            "fixture",
            "--output-dir",
            str(tmp_path),
            "--checkpoint-reference-uri",
            "hf://buckets/TKendrick/OpenUI/checkpoints/x/ref.json",
        ]
    )
    assert code == 0
    assert (tmp_path / "external_ceiling_report.json").exists()
    assert (tmp_path / "external_ceiling_report.md").exists()
    payload = json.loads((tmp_path / "external_ceiling_report.json").read_text())
    assert payload["status"] == "fixture"


def test_describe_prints_manifest(capsys) -> None:
    code = runner.main(
        [
            "--describe",
            "--checkpoint-reference-uri",
            "hf://buckets/TKendrick/OpenUI/checkpoints/x/ref.json",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "external-ceiling" in captured.out


def test_missing_checkpoint_reference_fails_for_frontier_arm(tmp_path: Path) -> None:
    code = runner.main(["--mode", "fixture", "--output-dir", str(tmp_path)])
    assert code == 1
