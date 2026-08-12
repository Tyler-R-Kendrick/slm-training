"""RESEARCH-04 / SLM-550 harness tests (default-off)."""

from __future__ import annotations

from pathlib import Path

from slm_training.harnesses.experiments import research_04_rm_zoo_benchmark as r04


ROOT = Path(__file__).resolve().parents[3]


def test_default_off_skips_without_enable() -> None:
    result = r04.run_experiment(root=ROOT, enabled=False)
    assert result["executed"] is False
    assert result["decision"] == "skipped_default_off"


def test_lock_digest_stable() -> None:
    a = r04.lock_campaign(root=ROOT)
    b = r04.lock_campaign(root=ROOT)
    assert a.manifest_sha256 == b.manifest_sha256
    assert len(a.manifest_sha256) == 64
