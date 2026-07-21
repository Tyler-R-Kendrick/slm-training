"""Tests for scripts/audit_edit_reachability.py."""

from __future__ import annotations

import json
from pathlib import Path

import scripts.audit_edit_reachability as audit


def test_describe_mode_prints_schema(capsys) -> None:
    assert audit.main(["--mode", "describe"]) == 0
    captured = capsys.readouterr()
    assert "TransitionCertificateV1" in captured.out


def test_plan_only_mode_writes_manifest(tmp_path: Path) -> None:
    assert audit.main(["--mode", "plan-only", "--output-dir", str(tmp_path)]) == 0
    run_json = tmp_path / "slm188_edit_algebra_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text(encoding="utf-8"))
    assert data["status"] == "plan_only"
    assert data["matrix_set"] == audit.EXPERIMENT_ID.replace("-", "_")


def test_fixture_mode_writes_report_and_design_docs(tmp_path: Path) -> None:
    assert audit.main(["--mode", "fixture", "--output-dir", str(tmp_path), "--design-json", str(tmp_path / "design.json"), "--design-md", str(tmp_path / "design.md")]) == 0
    run_json = tmp_path / "slm188_edit_algebra_report.json"
    assert run_json.exists()
    data = json.loads(run_json.read_text(encoding="utf-8"))
    assert data["status"] == "fixture"
    assert data["n_cases"] > 0


def test_target_mode_audits_single_target(tmp_path: Path) -> None:
    target = 'root = Stack([cta], "column")\ncta = Button(":x")'
    assert audit.main([
        "--mode", "fixture",
        "--target", target,
        "--slots", ":x",
        "--output-dir", str(tmp_path),
        "--design-json", str(tmp_path / "design.json"),
        "--design-md", str(tmp_path / "design.md"),
    ]) == 0
    run_json = tmp_path / "slm188_edit_algebra_report.json"
    data = json.loads(run_json.read_text(encoding="utf-8"))
    assert any(c["result"] == "reachable" for c in data["cases"])


def test_emit_bridges_writes_bridges(tmp_path: Path) -> None:
    assert audit.main([
        "--mode", "fixture",
        "--output-dir", str(tmp_path),
        "--emit-bridges",
        "--design-json", str(tmp_path / "design.json"),
        "--design-md", str(tmp_path / "design.md"),
    ]) == 0
    bridges = tmp_path / "bridges.jsonl"
    assert bridges.exists()
    lines = bridges.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    cert = json.loads(lines[0])
    assert cert["schema"] == "TransitionCertificateV1"
