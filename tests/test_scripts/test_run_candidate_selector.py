"""Tests for scripts/run_candidate_selector.py (SLM-127)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import run_candidate_selector
from slm_training.harnesses.experiments.candidate_selector import (
    make_fixture_groups,
    write_selection_groups,
)


def test_fixture_mode_writes_report(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    rc = run_candidate_selector.main(["--fixture", "--out", str(out)])
    assert rc == 0
    assert out.exists()
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema"] == "CandidateSelectorReportV1"
    assert report["claim_class"] == "wiring"
    assert report["status"] == "fixture"
    assert report["fixture_groups"] == 8
    assert "arm_results" in report
    assert "model_score" in report["arm_results"]
    assert "learned_abstain" in report["arm_results"]
    assert "version_stamp" in report


def test_unknown_selector_fails(tmp_path: Path) -> None:
    groups = make_fixture_groups()
    groups_path = tmp_path / "groups.jsonl"
    write_selection_groups(str(groups_path), groups)
    rc = run_candidate_selector.main(
        ["--groups", str(groups_path), "--selector", "unknown_selector"]
    )
    assert rc == 2


def test_groups_stdout_mode(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    groups = make_fixture_groups()
    groups_path = tmp_path / "groups.jsonl"
    write_selection_groups(str(groups_path), groups)
    rc = run_candidate_selector.main(
        ["--groups", str(groups_path), "--selector", "model_score"]
    )
    assert rc == 0


def test_groups_learned_abstain_runs(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    groups = make_fixture_groups()
    groups_path = tmp_path / "groups.jsonl"
    write_selection_groups(str(groups_path), groups)
    out = tmp_path / "learned_report.json"
    rc = run_candidate_selector.main(
        [
            "--groups",
            str(groups_path),
            "--selector",
            "learned_abstain",
            "--calibrate-target-risk",
            "0.1",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["arm_results"]["learned_abstain"]["n_groups"] == 8
