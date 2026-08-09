"""SIE-001 (SLM-467): EXP-SR-1..EXP-SR-12 registry/schema tests.

Drives the shipped ``exp_sr_catalogue`` entry points and the real
``ExperimentCampaignV1`` / ``CampaignStore`` contract they build on — not a
parallel reimplementation of either.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.autoresearch.experiment_campaign import (
    CampaignGateV1,
    ExperimentCampaignV1,
    campaign_manifest_sha256,
)
from slm_training.autoresearch.schemas import CampaignSpec
from slm_training.autoresearch.storage import CampaignStore
from slm_training.harnesses.experiments.exp_sr_catalogue import (
    EXP_SR_FAMILY_SPECS,
    build_exp_sr_catalogue,
    exp_sr_campaign,
    plan_only_manifest,
)
from slm_training.levers import MAX_RUN_MINUTES

_EXPECTED_FAMILY_IDS = tuple(f"exp-sr-{n}" for n in range(1, 13))


def _gate_fires(gate: CampaignGateV1, value: float) -> bool:
    return {
        "ge": value >= gate.threshold,
        "gt": value > gate.threshold,
        "le": value <= gate.threshold,
        "lt": value < gate.threshold,
        "eq": value == gate.threshold,
    }[gate.operator]


def test_catalogue_has_twelve_unique_stable_identities() -> None:
    campaigns = build_exp_sr_catalogue()
    assert len(campaigns) == 12
    assert all(isinstance(item, ExperimentCampaignV1) for item in campaigns)
    ids = tuple(item.experiment_id for item in campaigns)
    assert ids == _EXPECTED_FAMILY_IDS
    assert len(set(ids)) == 12
    for item in campaigns:
        assert item.campaign_id == item.experiment_id


def test_no_entry_can_mutate_defaults_or_promote_from_registration_alone() -> None:
    """AC: fixture-only success is mechanically non-promotable.

    None of the twelve registrations use a promotion-class ``claim_class``;
    the underlying ``ExperimentCampaignV1`` contract already refuses
    ``promotion_candidate``/``ship_gate`` without a locked eval manifest and
    full causal-shape evidence, so simply cataloging an identity here can
    never satisfy those requirements on its own.
    """
    for item in build_exp_sr_catalogue():
        assert item.claim_class not in {"promotion_candidate", "ship_gate"}
        assert item.locked_eval_manifest_sha256 is None


def test_every_family_declares_a_falsifier_and_matched_and_leakage_controls() -> None:
    for spec, campaign in zip(
        EXP_SR_FAMILY_SPECS, build_exp_sr_catalogue(), strict=True
    ):
        assert campaign.stopping_rules[0] == f"Falsifier: {spec.falsifier}"
        kinds = {control.kind for control in campaign.controls}
        assert "quality" in kinds
        assert "negative" in kinds
        assert campaign.negative_controls == (f"{spec.family_id}_leakage_control",)
        # The repository-wide hard run cap (AGENTS.md "Hard run cap"), not
        # the CampaignBudget schema's own looser 60-minute ceiling.
        assert campaign.budget.max_wall_minutes <= MAX_RUN_MINUTES


def test_rollback_gates_are_not_vacuous() -> None:
    """Every rollback gate must be satisfiable both ways.

    A rollback gate pinned at a metric's natural boundary with an inclusive
    comparison (e.g. ``le 1.0`` on a [0, 1] rate) matches every possible
    outcome and would make ``validate_result_claim`` reject every
    ``CampaignResultV1`` for that family forever, since
    ``rollback_gates_not_passed`` fires whenever any rollback gate matches.
    For every family, the gate must not fire at the "perfect" end of the
    metric's declared direction, and must fire on a clearly bad outcome.
    """
    for spec, campaign in zip(
        EXP_SR_FAMILY_SPECS, build_exp_sr_catalogue(), strict=True
    ):
        gate = campaign.rollback_gates[0]
        perfect = 1.0 if spec.primary_direction == "increase" else 0.0
        bad = 0.0 if spec.primary_direction == "increase" else 1.0
        assert not _gate_fires(gate, perfect), (spec.family_id, gate)
        assert _gate_fires(gate, bad), (spec.family_id, gate)


def test_exp_sr_campaign_lookup_and_unknown_family_raises() -> None:
    looked_up = exp_sr_campaign("exp-sr-4")
    assert looked_up.experiment_id == "exp-sr-4"
    with pytest.raises(KeyError, match="exp-sr-99"):
        exp_sr_campaign("exp-sr-99")


def test_plan_only_manifest_never_locks_and_matches_the_registered_digest() -> None:
    preview = plan_only_manifest("exp-sr-1")
    assert preview["status"] == "plan_only"
    assert preview["manifest"]["experiment_id"] == "exp-sr-1"
    # The digest must describe the returned payload itself, not merely agree
    # with a second independently-built object from the same inputs (which
    # would pass even if the payload didn't match its own claimed digest).
    round_tripped = ExperimentCampaignV1.model_validate(preview["manifest"])
    assert campaign_manifest_sha256(round_tripped) == preview["manifest_sha256"]
    assert preview["manifest_sha256"] == campaign_manifest_sha256(
        exp_sr_campaign("exp-sr-1")
    )


def test_duplicate_relock_with_different_content_is_rejected(tmp_path: Path) -> None:
    """The real registry (CampaignStore), not a hand-copy, rejects ID collisions."""
    manifest = exp_sr_campaign("exp-sr-4")
    store = CampaignStore(manifest.campaign_id, tmp_path)
    store.initialize(
        CampaignSpec(
            campaign_id=manifest.campaign_id,
            objective="Register the EXP-SR-4 witness-guided CEGIS repair identity.",
            primary_metric=manifest.endpoints[0].metric,
            researcher_mode="fixture",
        )
    )
    store.lock_experiment_campaign(manifest)
    mutated = manifest.model_copy(
        update={"hypothesis": manifest.hypothesis + " (mutated)"}
    )
    with pytest.raises(FileExistsError, match="different content"):
        store.lock_experiment_campaign(mutated)


def test_same_content_relock_is_idempotent(tmp_path: Path) -> None:
    manifest = exp_sr_campaign("exp-sr-7")
    store = CampaignStore(manifest.campaign_id, tmp_path)
    store.initialize(
        CampaignSpec(
            campaign_id=manifest.campaign_id,
            objective="Register the EXP-SR-7 packed semantic-summary identity.",
            primary_metric=manifest.endpoints[0].metric,
            researcher_mode="fixture",
        )
    )
    first = store.lock_experiment_campaign(manifest)
    second = store.lock_experiment_campaign(manifest)
    assert first.manifest_sha256 == second.manifest_sha256
