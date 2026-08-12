"""RESEARCH-20 harness tests."""

from pathlib import Path

from slm_training.harnesses.experiments import (
    research_20_cost_kernel_cslib_calf_aara_compare as r20,
)

ROOT = Path(__file__).resolve().parents[3]


def test_default_off() -> None:
    result = r20.run_experiment(root=ROOT, enabled=False)
    assert result["decision"] == "skipped_default_off"


def test_lock_stable() -> None:
    assert r20.lock_campaign(root=ROOT).manifest_sha256 == r20.lock_campaign(
        root=ROOT
    ).manifest_sha256
