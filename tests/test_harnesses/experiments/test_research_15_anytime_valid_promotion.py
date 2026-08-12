"""RESEARCH-15 / SLM-569: harness tests (default-off)."""

from __future__ import annotations

from pathlib import Path

from slm_training.harnesses.experiments import (
    research_15_anytime_valid_promotion as r15,
)

ROOT = Path(__file__).resolve().parents[3]


def test_default_off_skips_without_enable() -> None:
    result = r15.run_experiment(root=ROOT, enabled=False)
    assert result["executed"] is False
    assert result["decision"] == "skipped_default_off"


def test_lock_digest_stable() -> None:
    a = r15.lock_campaign(root=ROOT)
    b = r15.lock_campaign(root=ROOT)
    assert a.manifest_sha256 == b.manifest_sha256
    assert len(a.manifest_sha256) == 64
