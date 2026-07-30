"""Tests for the SLM-296 (AP-013) verifier-filtered teacher-admission fixture CLI."""

from __future__ import annotations

import json

import pytest

import scripts.run_verified_teacher_admission as _runner_module
from scripts.run_verified_teacher_admission import main


@pytest.fixture(autouse=True)
def _isolate_design_artifacts(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(_runner_module, "_DESIGN_JSON", str(tmp_path / "design.json"))
    monkeypatch.setattr(_runner_module, "_DESIGN_MD", str(tmp_path / "design.md"))


def test_plan_only_writes_manifest(tmp_path) -> None:
    assert main(["--mode", "plan-only", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "slm296_verified_teacher_admission_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "plan_only"
    assert data["claim_class"] == "wiring"
    assert data["schema"] == "Slm296VerifiedTeacherAdmissionManifestV1"
    assert data["version_stamp"]


def test_fixture_writes_design_docs(tmp_path) -> None:
    assert main(["--mode", "fixture", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "slm296_verified_teacher_admission_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text())
    assert data["status"] == "yield_limitation_no_real_teacher_available"
    assert data["claim_class"] == "wiring"
    assert data["counts"]
    assert data["manifest_hash"]
