"""Tests for scripts/audit_dca_activation.py (SLM-270+)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import audit_dca_activation


def test_slm270_default_run_reports_blocked(tmp_path: Path) -> None:
    out = tmp_path / "gate"
    rc = audit_dca_activation.main(["--issue", "SLM-270", "--out", str(out)])
    assert rc == 0

    data = json.loads((out / "dca0_02_objective_surgery_gate.json").read_text())
    assert data["verdict"] == "blocked"
    assert data["issue"]["alias"] == "DCA0-02"
    assert (out / "dca0_02_objective_surgery_gate.md").exists()


def test_run_against_explicit_root(tmp_path: Path) -> None:
    out = tmp_path / "gate_root"
    rc = audit_dca_activation.main(
        ["--issue", "SLM-270", "--root", ".", "--out", str(out)]
    )
    assert rc == 0
    markdown = (out / "dca0_02_objective_surgery_gate.md").read_text()
    assert "SLM-270" in markdown
    assert "blocked" in markdown


def test_slm271_run_reports_blocked(tmp_path: Path) -> None:
    out = tmp_path / "gate271"
    rc = audit_dca_activation.main(["--issue", "SLM-271", "--out", str(out)])
    assert rc == 0
    data = json.loads((out / "dca0_03_mask_curriculum_gate.json").read_text())
    assert data["verdict"] == "blocked"
    assert data["issue"]["alias"] == "DCA0-03"
