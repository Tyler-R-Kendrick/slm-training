"""Tests for scripts/build_causal_disposition.py (SLM-276)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import build_causal_disposition


def test_default_run_publishes_bundle(tmp_path: Path) -> None:
    out = tmp_path / "disposition"
    rc = build_causal_disposition.main(["--out", str(out)])
    assert rc == 0

    data = json.loads((out / "causal_disposition_bundle.json").read_text())
    assert data["outcome"] == "no_model_ship_ready_measurement_or_data_blocker"
    assert data["linear_issue"] == "SLM-276"
    assert data["bundle_fingerprint"]
    assert (out / "causal_disposition_bundle.md").exists()


def test_check_fails_on_evidence_gaps(tmp_path: Path) -> None:
    out = tmp_path / "disposition_check"
    rc = build_causal_disposition.main(
        ["--root", str(tmp_path), "--out", str(out), "--check"]
    )
    assert rc == 1
