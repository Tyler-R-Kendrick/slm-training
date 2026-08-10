"""Tests for SLM-492 external-blocked prepared-package closeout."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from slm_training.harnesses.experiments.slm492_external_blocked_closeout import (
    probe_environment,
    run_closeout,
)


def test_probe_environment_records_missing_pysr() -> None:
    env = probe_environment()
    assert env["pysr_import_available"] is False
    assert env["live_execution_ready"] is False
    assert "julia_binary_available" in env
    assert "manifest_fingerprint" in env


def test_closeout_external_blocked_not_a_loss(tmp_path: Path) -> None:
    payload = run_closeout(root=tmp_path)
    assert payload["status"] == "external_blocked"
    assert payload["incomplete"] is True
    assert payload["srbench_matched_score_gap"] is None
    assert payload["no_sota_claims"] is True
    rsp007 = payload["rsp007_result"]
    assert rsp007["evidence"] == "external-blocked"
    assert rsp007["complete"] is False
    pysr = rsp007["pysr_adapter_matched"]
    assert pysr["validation_loss"] is None


def test_no_pysr_import_required() -> None:
    assert importlib.util.find_spec("pysr") is None
