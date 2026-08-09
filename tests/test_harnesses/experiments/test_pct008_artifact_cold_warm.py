"""PCT-008 (SLM-464): fixture cold/warm artifact-parity evidence campaign tests.

Drives the shipped ``pct008_artifact_cold_warm`` entry points -- and, through
them, the real PCT-003 cold/warm bench and PCT-005/PCT-006 completion-artifact
seam -- rather than reimplementing any of them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harnesses.experiments.pct008_artifact_cold_warm import (
    COLD_ARMS,
    WARM_ARM,
    Pct008CampaignV1,
    check_domain_parity,
    run_campaign,
)


def test_campaign_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="cold_trials_per_arm"):
        Pct008CampaignV1(cold_trials_per_arm=0)
    with pytest.raises(ValueError, match="warm_trials"):
        Pct008CampaignV1(warm_trials=0)
    with pytest.raises(ValueError, match="fixture"):
        Pct008CampaignV1(claim_class="diagnostic")
    with pytest.raises(ValueError, match="MAX_RUN_MINUTES"):
        Pct008CampaignV1(max_wall_minutes=1000.0)


def test_manifest_is_a_real_valid_campaign_with_five_arms() -> None:
    manifest = Pct008CampaignV1().manifest()
    assert isinstance(manifest, ExperimentCampaignV1)
    arm_ids = {arm.arm_id for arm in manifest.arms}
    assert arm_ids == {*COLD_ARMS, WARM_ARM}
    assert manifest.claim_class == "fixture"
    # AC: no model-quality/production-speedup claim -- never promotion-class.
    assert manifest.claim_class not in {"promotion_candidate", "ship_gate"}
    assert manifest.locked_eval_manifest_sha256 is None
    control_arms = {arm.arm_id for arm in manifest.arms if arm.role == "control"}
    assert control_arms == {"no_artifact"}


def test_manifest_rollback_gate_is_not_vacuous() -> None:
    """A real anomaly guard, satisfiable both ways -- not a placeholder that
    always (or never) fires regardless of the measured value."""
    manifest = Pct008CampaignV1().manifest()
    gate = manifest.rollback_gates[0]
    normal_timing_ms = 50.0
    hung_trial_ms = 1_000_000.0
    fires = {
        "ge": lambda v: v >= gate.threshold,
        "gt": lambda v: v > gate.threshold,
        "le": lambda v: v <= gate.threshold,
        "lt": lambda v: v < gate.threshold,
        "eq": lambda v: v == gate.threshold,
    }[gate.operator]
    assert not fires(normal_timing_ms)
    assert fires(hung_trial_ms)


def test_domain_parity_holds_for_the_fixed_prefix() -> None:
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    parity = check_domain_parity(tok)
    assert parity["equal"] is True
    assert parity["reasons"] == []
    assert parity["oracle_candidate_count"] == parity["production_candidate_count"]


def test_domain_parity_check_fails_closed_on_a_real_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A genuine divergence must be reported as unequal, not silently passed."""
    import slm_training.dsl.grammar.fastpath.static_control_domain as scd
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tok = DSLNativeTokenizer.build()
    real_snapshot = scd.production_domain_snapshot

    def _widened(*args, **kwargs):
        snap = real_snapshot(*args, **kwargs)
        # Fabricate an extra candidate: a real widening bug production
        # code must never exhibit, but the parity check must catch it.
        return scd.DomainSnapshot(
            status=snap.status,
            coverage=snap.coverage,
            reason=snap.reason,
            candidates=(*snap.candidates, (999999,)),
            candidate_kinds=(*snap.candidate_kinds, "fabricated"),
            terminals=snap.terminals,
            authority=snap.authority,
            gamma_applied=snap.gamma_applied,
            gamma_only_tightens=snap.gamma_only_tightens,
            certified_projection=snap.certified_projection,
        )

    monkeypatch.setattr(scd, "production_domain_snapshot", _widened)
    parity = check_domain_parity(tok)
    assert parity["equal"] is False
    assert "candidate_multiset_mismatch" in parity["reasons"]


def test_run_campaign_end_to_end_with_real_arms(tmp_path: Path) -> None:
    """Real fresh-process cold trials for all four cold arms, plus a warm arm."""
    campaign = Pct008CampaignV1(cold_trials_per_arm=1, warm_trials=2)
    result = run_campaign(campaign, root=tmp_path, timeout_s=20.0)

    assert result["domain_parity"]["equal"] is True
    assert result["claim_class"] == "fixture"
    assert len(result["trials"]) == len(COLD_ARMS) * 1 + 1
    cold_trials = [t for t in result["trials"] if t["trial_kind"] == "cold"]
    warm_trials = [t for t in result["trials"] if t["trial_kind"] == "warm"]
    assert len(cold_trials) == len(COLD_ARMS)
    assert len(warm_trials) == 1
    for trial in cold_trials:
        assert trial["completeness"] == "complete"
        assert [p["phase"] for p in trial["phases"]] == [
            "process_launch",
            "tokenizer_pack_construction",
            "artifact_step",
            "first_domain_query",
        ]
    assert result["summary"]["excluded_incomplete"] == 0
    assert len(result["summary"]["arms"]) == len(COLD_ARMS) + 1
    # AC: cold and warm are reported separately, never blended.
    kinds = {arm["trial_kind"] for arm in result["summary"]["arms"].values()}
    assert kinds == {"cold", "warm"}
    assert "scope_disclaimer" in result and result["scope_disclaimer"]

    store = CampaignStore(campaign.campaign_id, tmp_path)
    locked = store.load_experiment_campaign(campaign.campaign_id)
    assert locked.manifest_sha256 == result["manifest_sha256"]


def test_run_campaign_marks_timeouts_incomplete_not_fabricated(
    tmp_path: Path,
) -> None:
    """AC: partial/timeouts remain incomplete -- never counted as a complete arm."""
    campaign = Pct008CampaignV1(cold_trials_per_arm=1, warm_trials=1)
    # A real cold trial takes hundreds of ms (fresh interpreter + package
    # import); a near-zero timeout forces every cold arm to time out.
    result = run_campaign(campaign, root=tmp_path, timeout_s=0.001)

    cold_trials = [t for t in result["trials"] if t["trial_kind"] == "cold"]
    assert len(cold_trials) == len(COLD_ARMS)
    assert all(t["completeness"] == "partial_timeout" for t in cold_trials)
    assert all(t["completeness"] != "complete" for t in cold_trials)
    # No cold arm may appear in the "complete" summary as if it had finished.
    cold_arm_summaries = [
        arm for arm in result["summary"]["arms"].values() if arm["trial_kind"] == "cold"
    ]
    assert cold_arm_summaries == []
    assert result["summary"]["excluded_incomplete"] >= len(COLD_ARMS)


def test_run_campaign_aborts_before_any_trial_on_parity_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import slm_training.harnesses.experiments.pct008_artifact_cold_warm as mod

    def _fake_mismatch(_tokenizer):
        return {
            "equal": False,
            "reasons": ["candidate_multiset_mismatch"],
            "oracle_candidate_count": 1,
            "production_candidate_count": 2,
        }

    monkeypatch.setattr(mod, "check_domain_parity", _fake_mismatch)

    def _ban_cold_trial(*_a, **_k):
        raise AssertionError("no cold trial may run once parity has failed")

    monkeypatch.setattr(mod, "run_cold_trial", _ban_cold_trial)

    campaign = Pct008CampaignV1(cold_trials_per_arm=1, warm_trials=1)
    with pytest.raises(RuntimeError, match="domain parity failed"):
        run_campaign(campaign, root=tmp_path, timeout_s=20.0)
