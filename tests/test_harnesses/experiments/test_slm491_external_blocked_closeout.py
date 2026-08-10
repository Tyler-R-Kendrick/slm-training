"""Tests for SLM-491 external-blocked prepared-package closeout."""

from __future__ import annotations

from pathlib import Path

from slm_training.harnesses.experiments.slm491_external_blocked_closeout import (
    probe_environment,
    run_closeout,
)


def test_probe_environment_honest_defaults() -> None:
    env = probe_environment()
    assert env["real_human_raters_available"] is False
    assert "external_judge_api_key_set" in env


def test_closeout_external_blocked_null_kappa(tmp_path: Path) -> None:
    payload = run_closeout(root=tmp_path)
    assert payload["status"] == "external_blocked"
    assert payload["incomplete"] is True
    assert payload["evaluator_human_agreement_kappa"] is None
    assert payload["external_blocked"] is True
    assert payload["prepared_package"]["sie005_frozen_doc"]
    sie006 = payload["sie006_result"]
    assert sie006["evaluator_human_agreement_kappa"] is None
