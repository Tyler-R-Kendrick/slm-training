"""Tests for scripts/evaluate_lot_downstream_gate.py (SLM-252+)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import evaluate_lot_downstream_gate


def test_slm252_default_run_reports_not_authorized(tmp_path: Path) -> None:
    out = tmp_path / "gate"
    rc = evaluate_lot_downstream_gate.main(["--issue", "SLM-252", "--out", str(out)])
    assert rc == 0

    data = json.loads((out / "lot2_01_supervision_routing_gate.json").read_text())
    assert data["verdict"] == "not_authorized"
    assert data["issue"]["linear_issue"] == "SLM-252"
    assert (out / "lot2_01_supervision_routing_gate.md").exists()


def test_run_against_explicit_contract_paths(tmp_path: Path) -> None:
    out = tmp_path / "gate_explicit"
    rc = evaluate_lot_downstream_gate.main(
        [
            "--issue",
            "SLM-252",
            "--fidelity-contract",
            "docs/design/lotus-openui-fidelity-contract-v1.json",
            "--trace-gate-contract",
            "docs/design/compiler-reasoning-trace-v1.json",
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    markdown = (out / "lot2_01_supervision_routing_gate.md").read_text()
    assert "SLM-252" in markdown
    assert "not_authorized" in markdown
