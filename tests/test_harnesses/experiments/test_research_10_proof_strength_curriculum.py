"""RESEARCH-10 / SLM-566 harness tests (default-off)."""

from __future__ import annotations

from pathlib import Path

from slm_training.harnesses.experiments import (
    research_10_proof_strength_curriculum as r10,
)

ROOT = Path(__file__).resolve().parents[3]


def test_default_off_skips_without_enable() -> None:
    result = r10.run_experiment(root=ROOT, enabled=False)
    assert result["executed"] is False
    assert result["decision"] == "skipped_default_off"


def test_lock_digest_stable() -> None:
    a = r10.lock_campaign(root=ROOT)
    b = r10.lock_campaign(root=ROOT)
    assert a.manifest_sha256 == b.manifest_sha256
    assert len(a.manifest_sha256) == 64
