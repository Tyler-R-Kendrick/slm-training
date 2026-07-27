"""Tests for scripts/evaluate_lot1_02_activation_gate.py (SLM-251)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import evaluate_lot1_02_activation_gate


def test_default_run_reports_not_authorized(tmp_path: Path) -> None:
    out = tmp_path / "gate"
    rc = evaluate_lot1_02_activation_gate.main(["--out", str(out)])
    assert rc == 0

    data = json.loads((out / "lot1_02_launch_gate.json").read_text())
    assert data["verdict"] == "not_authorized"
    assert data["lot1_01_disposition"] == "not_authorized"
    assert (out / "lot1_02_launch_gate.md").exists()


def test_run_against_explicit_contract_paths(tmp_path: Path) -> None:
    out = tmp_path / "gate_explicit"
    rc = evaluate_lot1_02_activation_gate.main(
        [
            "--fidelity-contract",
            "docs/design/lotus-openui-fidelity-contract-v1.json",
            "--trace-gate-contract",
            "docs/design/compiler-reasoning-trace-v1.json",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    markdown = (out / "lot1_02_launch_gate.md").read_text()
    assert "SLM-251" in markdown
    assert "not_authorized" in markdown
