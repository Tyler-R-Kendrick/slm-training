"""VCE-009 (SLM-468): oracle/contrast fixture campaign tests.

Drives the shipped ``vce009_oracle_contrast_campaign`` entry points -- and,
through them, the real VCE-005 oracle intervention arms and VCE-006/VCE-007
semantic-contrast/metamorphic generators -- rather than reimplementing any
of them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harnesses.experiments.vce009_oracle_contrast_campaign import (
    CONTRAST_ARMS,
    ORACLE_ARMS,
    Vce009CampaignV1,
    run_campaign,
)

_VALID_DISPOSITIONS = {"match", "mismatch", "inconclusive"}


def test_campaign_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="source_count"):
        Vce009CampaignV1(source_count=1)
    with pytest.raises(ValueError, match="fixture"):
        Vce009CampaignV1(claim_class="diagnostic")
    with pytest.raises(ValueError, match="MAX_RUN_MINUTES"):
        Vce009CampaignV1(max_wall_minutes=1000.0)


def test_manifest_is_a_real_valid_campaign_with_ten_arms() -> None:
    manifest = Vce009CampaignV1().manifest()
    assert isinstance(manifest, ExperimentCampaignV1)
    arm_ids = {arm.arm_id for arm in manifest.arms}
    assert arm_ids == {*ORACLE_ARMS, *CONTRAST_ARMS}
    assert manifest.claim_class == "fixture"
    # AC: no champion/default/checkpoint promotion claim -- never promotion-class.
    assert manifest.claim_class not in {"promotion_candidate", "ship_gate"}
    assert manifest.locked_eval_manifest_sha256 is None
    control_arms = {arm.arm_id for arm in manifest.arms if arm.role == "control"}
    assert control_arms == {"oracle_baseline"}


def test_manifest_rollback_gate_is_not_vacuous() -> None:
    """No arm intentionally uses plan_source='gold', so a nonzero contaminated
    count is always an anomaly -- the gate must fire both ways, not always
    (or never) regardless of the measured value."""
    manifest = Vce009CampaignV1().manifest()
    gate = next(
        g for g in manifest.rollback_gates if g.gate_id == "unexpected_oracle_contamination"
    )
    assert gate.operator == "gt"
    assert not (0.0 > gate.threshold)
    assert 1.0 > gate.threshold


def test_manifest_negative_control_matches_declared_negative_controls() -> None:
    manifest = Vce009CampaignV1().manifest()
    negative_control_ids = {c.control_id for c in manifest.controls if c.kind == "negative"}
    assert negative_control_ids == set(manifest.negative_controls)
    assert negative_control_ids  # at least one real negative control


def test_run_campaign_end_to_end_with_real_arms(tmp_path: Path) -> None:
    campaign = Vce009CampaignV1()
    result = run_campaign(campaign, root=tmp_path)

    assert result["claim_class"] == "fixture"
    assert len(result["arms"]) == len(ORACLE_ARMS) + len(CONTRAST_ARMS)
    arm_ids = {row["arm"] for row in result["arms"]}
    assert arm_ids == {*ORACLE_ARMS, *CONTRAST_ARMS}
    for row in result["arms"]:
        assert row["disposition"] in _VALID_DISPOSITIONS
        assert "compute" in row
        assert set(row["compute"]) == {"forwards", "verifier_calls", "wall_ms"}
    assert 0.0 <= result["arm_contract_match_rate"] <= 1.0
    assert "scope_disclaimer" in result and result["scope_disclaimer"]
    assert "version_stamp" in result

    store = CampaignStore(campaign.campaign_id, tmp_path)
    locked = store.load_experiment_campaign(campaign.campaign_id)
    assert locked.manifest_sha256 == result["manifest_sha256"]


def test_run_campaign_is_deterministic_on_repeat(tmp_path: Path) -> None:
    """AC: repeat identical fixture runs, compare artifact hashes."""
    campaign = Vce009CampaignV1()
    first = run_campaign(campaign, root=tmp_path / "run1")
    second = run_campaign(campaign, root=tmp_path / "run2")

    assert first["manifest_sha256"] == second["manifest_sha256"]
    first_dispositions = [(row["arm"], row["disposition"]) for row in first["arms"]]
    second_dispositions = [(row["arm"], row["disposition"]) for row in second["arms"]]
    assert first_dispositions == second_dispositions
    assert first["arm_contract_match_rate"] == second["arm_contract_match_rate"]


def test_oracle_arms_are_never_contaminated_by_construction(tmp_path: Path) -> None:
    """AC: oracle contamination banner survives all exports. None of these
    arms uses plan_source='gold', so none should ever be bannered -- and if
    one were, it would still show up here rather than being silently
    dropped."""
    result = run_campaign(Vce009CampaignV1(), root=tmp_path)
    assert result["contaminated_arm_ids"] == []
    for row in result["arms"]:
        assert row.get("is_contaminated") in (False, None)
        assert row.get("contamination_banner") is None


def test_shuffled_arm_reports_inconclusive_not_fabricated_when_no_compatible_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC: null/negative/incomplete results are first-class outcomes -- a
    real 'no compatible candidate' case must never be silently reported as
    a match."""
    import slm_training.harnesses.experiments.vce009_oracle_contrast_campaign as mod

    monkeypatch.setattr(mod, "select_shuffled_oracle", lambda *a, **k: None)
    result = run_campaign(Vce009CampaignV1(), root=tmp_path)

    shuffled = next(row for row in result["arms"] if row["arm"] == "oracle_shuffled")
    assert shuffled["disposition"] == "inconclusive"
    assert shuffled["reason"] == "no_compatible_candidate_in_frozen_slice"
    assert shuffled["disposition"] != "match"


def test_contrast_and_metamorphic_arms_reuse_real_evidence_writers(tmp_path: Path) -> None:
    result = run_campaign(Vce009CampaignV1(), root=tmp_path)
    by_arm = {row["arm"]: row for row in result["arms"]}

    contrast = by_arm["contrast_corpus_scoreboard"]
    assert contrast["pairs"] > 0
    assert contrast["scoreboard"]
    assert any(family["family"] == "positive" for family in contrast["scoreboard"])

    metamorphic = by_arm["metamorphic_generators"]
    assert metamorphic["cases"]
    for case in metamorphic["cases"]:
        assert "family" in case
        assert case["disposition"] in _VALID_DISPOSITIONS
