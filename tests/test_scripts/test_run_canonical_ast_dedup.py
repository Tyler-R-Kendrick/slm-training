"""Regression test for the SLM-130 canonical AST dedup CLI."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_canonical_ast_dedup import main


def test_run_canonical_ast_dedup_fixture_cli(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    rc = main(["--fixture", "--out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    assert data["claim_class"] == "wiring / fixture only"
    assert "version_stamp" in data
    arms = data["arm_reports"]
    assert "A_raw_no_dedup" in arms
    assert "C_terminal_canonical_ast" in arms
    assert arms["C_terminal_canonical_ast"]["unique_canonical_ast"] <= arms["A_raw_no_dedup"]["pool_size"]
    assert data["group_count"] >= 2
    assert len(data["truncated_ids_k3"]) <= 3


def test_run_canonical_ast_dedup_requires_fixture() -> None:
    rc = main([])
    assert rc == 2
