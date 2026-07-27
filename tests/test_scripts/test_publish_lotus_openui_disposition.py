"""Tests for scripts/publish_lotus_openui_disposition.py (SLM-258)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import publish_lotus_openui_disposition


def test_default_run_publishes_inconclusive_disposition(tmp_path: Path) -> None:
    out = tmp_path / "disposition"
    rc = publish_lotus_openui_disposition.main(["--out", str(out)])
    assert rc == 0

    data = json.loads((out / "lotus_openui_disposition.json").read_text())
    assert data["adoption_verdict"] == "inconclusive_missing_evidence"
    assert data["linear_issue"] == "SLM-258"
    assert data["disposition_hash"]
    assert (out / "lotus_openui_disposition.md").exists()


def test_markdown_renders_verdict(tmp_path: Path) -> None:
    out = tmp_path / "disposition_md"
    rc = publish_lotus_openui_disposition.main(["--out", str(out)])
    assert rc == 0
    markdown = (out / "lotus_openui_disposition.md").read_text()
    assert "inconclusive_missing_evidence" in markdown
    assert "SLM-258" in markdown
