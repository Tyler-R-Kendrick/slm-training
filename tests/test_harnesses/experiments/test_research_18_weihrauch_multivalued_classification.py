"""RESEARCH-18 / SLM-551 harness tests (default-off)."""

from __future__ import annotations

from pathlib import Path

from slm_training.harnesses.experiments import (
    research_18_weihrauch_multivalued_classification as r18,
)

ROOT = Path(__file__).resolve().parents[3]


def test_default_off_skips_without_enable() -> None:
    result = r18.run_experiment(root=ROOT, enabled=False)
    assert result["executed"] is False
    assert result["decision"] == "skipped_default_off"


def test_lock_digest_stable() -> None:
    a = r18.lock_campaign(root=ROOT)
    b = r18.lock_campaign(root=ROOT)
    assert a.manifest_sha256 == b.manifest_sha256
    assert len(a.manifest_sha256) == 64
