"""Tests for scripts/run_e228_exposure_ladder.py."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.run_e228_exposure_ladder as runner


def test_fixture_run_writes_report(tmp_path: Path) -> None:
    code = runner.main(
        [
            "--mode",
            "fixture",
            "--output-dir",
            str(tmp_path),
            "--parent-checkpoint-uri",
            "hf://buckets/TKendrick/OpenUI/checkpoints/e228/ref.json",
        ]
    )
    assert code == 0
    assert (tmp_path / "e228_exposure_manifest.json").exists()
    assert (tmp_path / "e228_exposure_report.json").exists()
    assert (tmp_path / "e228_exposure_report.md").exists()
    payload = json.loads((tmp_path / "e228_exposure_report.json").read_text())
    assert payload["status"] == "fixture"
    assert payload["ladder_id"] == "e228-exposure"


def test_frontier_without_parent_fails(tmp_path: Path) -> None:
    code = runner.main(["--mode", "fixture", "--output-dir", str(tmp_path)])
    assert code == 1


def test_plan_only_does_not_require_bucket(tmp_path: Path) -> None:
    code = runner.main(
        [
            "--mode",
            "plan-only",
            "--output-dir",
            str(tmp_path),
            "--parent-checkpoint-uri",
            "hf://buckets/TKendrick/OpenUI/checkpoints/e228/ref.json",
        ]
    )
    assert code == 0
    payload = json.loads((tmp_path / "e228_exposure_report.json").read_text())
    assert payload["status"] == "fixture"
