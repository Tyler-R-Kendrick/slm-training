"""RESEARCH-03 / SLM-549 harness tests (default-off)."""

from __future__ import annotations

from pathlib import Path

from slm_training.harnesses.experiments import research_03_big_five_interpretation as r03


ROOT = Path(__file__).resolve().parents[3]


def test_default_off_skips_without_enable() -> None:
    result = r03.run_experiment(root=ROOT, enabled=False)
    assert result["executed"] is False
    assert result["decision"] == "skipped_default_off"


def test_lock_digest_stable() -> None:
    a = r03.lock_campaign(root=ROOT)
    b = r03.lock_campaign(root=ROOT)
    assert a.manifest_sha256 == b.manifest_sha256
    assert len(a.manifest_sha256) == 64


def test_production_isolation_scan_clean() -> None:
    hits = r03._scan_production_imports(ROOT)
    assert hits == []


def test_control_and_treatment_gates() -> None:
    assert r03._control_print_axioms_rejected()["rejected"] is True
    treatment = r03._treatment_interpretation_ok()
    assert treatment["classification_accepted"] is True
    assert treatment["label_accepted"] is True


def test_four_axis_shape() -> None:
    axis = r03._four_axis(lake_ok=True, classification_ok=True)
    assert axis["assumption_strength"]["status"] == "proved_axis"
    assert axis["computability"]["status"] == "proved_axis"
    assert axis["resource_bounds"]["status"] == "not_claimed"
    assert axis["implementation_refinement"]["status"] == "proved_axis"
