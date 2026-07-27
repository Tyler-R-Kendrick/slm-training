"""Tests for scripts/audit_dca0_target_policy_activation.py (SLM-269)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import audit_dca0_target_policy_activation


def test_default_run_reports_blocked(tmp_path: Path) -> None:
    out = tmp_path / "activation"
    rc = audit_dca0_target_policy_activation.main(["--out", str(out)])
    assert rc == 0

    data = json.loads((out / "activation_report.json").read_text())
    assert data["activation_status"] == "blocked"
    assert data["disposition"] == "inconclusive"
    assert data["manifest_sha256"]
    assert (out / "activation_report.md").exists()


def test_run_against_explicit_root(tmp_path: Path) -> None:
    out = tmp_path / "activation_root"
    rc = audit_dca0_target_policy_activation.main(
        ["--root", ".", "--out", str(out)]
    )
    assert rc == 0
    markdown = (out / "activation_report.md").read_text()
    assert "SLM-269" in markdown
    assert "blocked" in markdown
