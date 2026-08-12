"""RESEARCH-19 harness tests."""

from pathlib import Path

from slm_training.harnesses.experiments import research_19_lean_rademacher_bound as r19

ROOT = Path(__file__).resolve().parents[3]


def test_default_off() -> None:
    result = r19.run_experiment(root=ROOT, enabled=False)
    assert result["decision"] == "skipped_default_off"


def test_lock_stable() -> None:
    assert r19.lock_campaign(root=ROOT).manifest_sha256 == r19.lock_campaign(root=ROOT).manifest_sha256
