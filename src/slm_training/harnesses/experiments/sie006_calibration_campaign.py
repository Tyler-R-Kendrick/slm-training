"""SIE-006 (SLM-483): evaluator construct-validity and human calibration
campaign (EXP-SR-3).

Runs VCE-010's already-complete, already-honest
:func:`~slm_training.evals.evaluator_calibration_protocol.run_protocol`
through the pre-registered ``exp-sr-3`` catalogue identity
(:func:`~slm_training.harnesses.experiments.exp_sr_catalogue.exp_sr_campaign`),
composing rather than forking it -- no second calibration/agreement
mechanism, no second blinded-packet/mock-judge machinery. This module's
entire job is to extract the catalogue's own locked primary metric
(``evaluator_human_agreement_kappa``) from VCE-010's result, evaluate it
against the catalogue's own preregistered gate (``kill_threshold=0.0``,
``kill_operator="le"``, ``minimum_effect=0.1``), and persist that under its
own ``CampaignStore`` campaign identity with a provenance link back to
VCE-010's own separately-locked evidence -- matching the SIE-002/SIE-004/
SIE-008 "thin driver over an already-owned mechanism" shape exactly. No new
ownership_map subsystem is registered: unlike RSP-004/RSP-005 (which
introduce genuinely new evaluation-only IR/data contracts), this module
introduces no new dataclass beyond a thin campaign-parameter wrapper, the
same shape already covered by the ``experiment_campaign``/
``verifier_gate_stack`` owners the pre-registered downstream_extension_map
row for SIE-006 already names.

Critical honesty note (SLM-483's own escape hatch: 'If the external/human
run cannot be completed, close as external-blocked with the prepared
package rather than fabricating evidence'): this environment has no real
external-judge API credentials and no real human raters. VCE-010's own
external-judge adapter and human-rater arms are *labeled mocks* (a wired,
seeded, synthetic stand-in -- never a real external-family judge or human
annotation), and this module inherits that scope exactly. ``claim_class``
is fixed to ``"diagnostic"`` (matching the locked exp-sr-3 catalogue entry)
and every result explicitly disclaims real construct validity: no
downstream experiment may cite this module's output as evidence a
promotion/risk claim is calibrated against genuine human judgment. This is
therefore NOT a fabricated substitute for the real external/human run the
issue describes -- it is the honest fixture/diagnostic-scale wiring
evidence the issue's own scope note anticipates producing while the real
run stays unavailable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from slm_training.autoresearch.experiment_campaign import (
    ArtifactRequirementV1,
    CampaignArmV1,
    CampaignControlV1,
    CampaignEndpointV1,
    CampaignGateV1,
    SELECTION_RULE_BEST_BY_PRIMARY_THEN_SMALLEST,
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
    campaign_manifest_sha256,
)
from slm_training.autoresearch.schemas import CampaignBudget, CampaignSpec
from slm_training.autoresearch.storage import CampaignStore
from slm_training.evals.evaluator_calibration_protocol import (
    ARM_IDS,
    EvaluatorCalibrationProtocolV1,
    run_protocol,
)
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.experiments.exp_sr_catalogue import (
    exp_sr_campaign,
    plan_only_manifest,
)
from slm_training.levers import MAX_RUN_MINUTES

CATALOGUE_ID = "exp-sr-3"
PRIMARY_METRIC = "evaluator_human_agreement_kappa"


@dataclass(frozen=True)
class Sie006CampaignV1:
    """Frozen campaign parameters. ``claim_class`` is fixed to
    ``"diagnostic"`` in ``__post_init__``, matching the locked exp-sr-3
    catalogue entry -- this experiment never claims promotion-grade
    construct validity."""

    campaign_id: str = "sie006-exp-sr-3-calibration-diagnostic"
    protocol_seed: int = 0
    min_raters: int = 2
    calibration_error_max: float = 0.35
    source_commit: str = "0" * 40
    claim_class: str = "diagnostic"
    max_wall_minutes: float = float(MAX_RUN_MINUTES)

    def __post_init__(self) -> None:
        if self.min_raters < 2:
            raise ValueError("min_raters must be >= 2 (>=2 human labels per pair)")
        if not 0.0 < self.calibration_error_max <= 1.0:
            raise ValueError("calibration_error_max must be in (0, 1]")
        if self.claim_class != "diagnostic":
            raise ValueError(
                "SIE-006 durable evidence must stay diagnostic, matching the "
                "locked exp-sr-3 catalogue entry"
            )
        if self.max_wall_minutes <= 0 or self.max_wall_minutes > MAX_RUN_MINUTES:
            raise ValueError("max_wall_minutes must obey MAX_RUN_MINUTES")

    @property
    def fingerprint(self) -> str:
        return content_sha(asdict(self))

    def catalogue_manifest(self) -> ExperimentCampaignV1:
        return exp_sr_campaign(CATALOGUE_ID)

    def minimum_effect(self) -> float:
        return float(self.catalogue_manifest().endpoints[0].minimum_effect)

    def protocol(self) -> EvaluatorCalibrationProtocolV1:
        return EvaluatorCalibrationProtocolV1(
            protocol_id=f"{self.campaign_id}-vce010-protocol",
            seed=self.protocol_seed,
            min_raters=self.min_raters,
            calibration_error_max=self.calibration_error_max,
            claim_class=self.claim_class,
            source_commit=self.source_commit,
            max_wall_minutes=self.max_wall_minutes,
        )

    def manifest(self) -> ExperimentCampaignV1:
        fingerprint = self.fingerprint
        catalogue = self.catalogue_manifest()
        arms = tuple(
            CampaignArmV1(
                arm_id=arm,
                role="control" if arm == "deterministic_only" else "candidate",
                config_sha256=content_sha({"campaign": fingerprint, "arm": arm}),
            )
            for arm in ARM_IDS
        )
        return ExperimentCampaignV1(
            campaign_id=self.campaign_id,
            experiment_id=CATALOGUE_ID,
            hypothesis=catalogue.hypothesis,
            decision=(
                "Report whether the deterministic evaluator's agreement with "
                "mocked external-judge and synthetic human evidence clears "
                "the preregistered kappa floor; no evaluator-promotion or "
                "model-quality claim is made from mock/synthetic evidence. "
                "The real external-judge/human run stays unavailable in "
                "this environment (SLM-483's own external-blocked escape "
                "hatch), so 'authorized' here is never a substitute for a "
                "genuine construct-validity result."
            ),
            endpoints=(
                CampaignEndpointV1(
                    endpoint_id=f"{CATALOGUE_ID}_primary",
                    metric=PRIMARY_METRIC,
                    role="primary",
                    direction="increase",
                    minimum_effect=self.minimum_effect(),
                ),
            ),
            arms=arms,
            selection_rule=SELECTION_RULE_BEST_BY_PRIMARY_THEN_SMALLEST,
            seeds=(self.protocol_seed,),
            budget=CampaignBudget(
                max_experiments=len(ARM_IDS),
                max_wall_minutes=self.max_wall_minutes,
            ),
            stopping_rules=(
                "An arm missing external or human evidence is reported "
                "blocked, never scored as if the evidence were present "
                "(inherited from VCE-010's own protocol).",
                "Every record is generated in-process; no subprocess timeout applies.",
                "No GPU / model forward; mock-judge/synthetic-rater "
                "fixture proxy only.",
            ),
            controls=(
                CampaignControlV1(
                    control_id="deterministic_only_control",
                    description=(
                        "The deterministic-only arm never consults external "
                        "or human evidence; it establishes the ground-truth "
                        "admission label the other arms are calibrated "
                        "against."
                    ),
                    kind="quality",
                ),
                CampaignControlV1(
                    control_id="mock_evidence_disclosed_control",
                    description=(
                        "The external judge and human raters are labeled "
                        "mocks (seeded synthetic stand-ins); the result "
                        "must carry an explicit scope_disclaimer stating so "
                        "rather than presenting mock evidence as real."
                    ),
                    kind="negative",
                ),
            ),
            negative_controls=("mock_evidence_disclosed_control",),
            multiplicity_families=(
                MultiplicityFamilyV1(
                    family_id=f"{CATALOGUE_ID}_family",
                    hypothesis_ids=(f"{CATALOGUE_ID}_primary",),
                    alpha=0.05,
                ),
            ),
            promotion_gates=(
                CampaignGateV1(
                    gate_id="diagnostic_only",
                    endpoint_id=f"{CATALOGUE_ID}_primary",
                    operator="ge",
                    threshold=0.0,
                ),
            ),
            rollback_gates=(
                CampaignGateV1(
                    gate_id=f"{CATALOGUE_ID}_kill",
                    endpoint_id=f"{CATALOGUE_ID}_primary",
                    operator="le",
                    threshold=0.0,
                ),
            ),
            artifact_requirements=(
                ArtifactRequirementV1(kind="version_stamp"),
                ArtifactRequirementV1(kind="endpoint_result"),
                ArtifactRequirementV1(kind="agentevals"),
                ArtifactRequirementV1(kind="agentv"),
            ),
            claim_class=self.claim_class,
            source_commit=self.source_commit,
            source_dirty=False,
            author="slm-training",
            created_at="1970-01-01T00:00:00Z",
        )


def run_campaign(campaign: Sie006CampaignV1, *, root: Path) -> dict[str, Any]:
    """Lock the exp-sr-3 identity, run VCE-010's protocol under a nested
    ``CampaignStore``, and persist the catalogue-scoped kappa evidence."""
    store = CampaignStore(campaign.campaign_id, root)
    store.initialize(
        CampaignSpec(
            campaign_id=campaign.campaign_id,
            objective="SIE-006 / EXP-SR-3 diagnostic evaluator construct-validity + calibration campaign",
            primary_metric=PRIMARY_METRIC,
            budget=CampaignBudget(
                max_experiments=len(ARM_IDS),
                max_wall_minutes=campaign.max_wall_minutes,
            ),
            created_at="1970-01-01T00:00:00Z",
        )
    )
    lock = store.lock_experiment_campaign(campaign.manifest())
    catalogue = campaign.catalogue_manifest()

    vce010_result = run_protocol(campaign.protocol(), root=root / "vce010")

    kappa = vce010_result.get("external_human_cohen_kappa")
    minimum_effect = campaign.minimum_effect()
    killed_by_catalogue_gate = kappa is None or kappa <= 0.0
    meets_minimum_effect = kappa is not None and kappa >= minimum_effect
    authorized = (
        not killed_by_catalogue_gate
        and meets_minimum_effect
        and bool(vce010_result.get("independence_ok"))
        and not bool(vce010_result.get("rollback_gate_fired"))
    )

    reason_family_confusion = next(
        (
            arm.get("reason_family_confusion")
            for arm in vce010_result.get("arms", [])
            if arm.get("arm") == "full_triple_judge"
        ),
        None,
    )

    result: dict[str, Any] = {
        "schema": "sie006_evaluator_calibration_diagnostic_result/v1",
        "campaign_id": campaign.campaign_id,
        "catalogue_id": CATALOGUE_ID,
        "catalogue_manifest_sha256": campaign_manifest_sha256(catalogue),
        "manifest_sha256": lock.manifest_sha256,
        "claim_class": campaign.claim_class,
        "promotion": False,
        "vce010_protocol_id": campaign.protocol().protocol_id,
        "vce010_campaign_id": vce010_result.get("campaign_id"),
        "vce010_manifest_sha256": vce010_result.get("manifest_sha256"),
        "vce010_disposition": vce010_result.get("disposition"),
        "pair_n": vce010_result.get("pair_n"),
        "primary_metric": PRIMARY_METRIC,
        "evaluator_human_agreement_kappa": kappa,
        "minimum_effect": minimum_effect,
        "kill_threshold": 0.0,
        "meets_minimum_effect": meets_minimum_effect,
        "killed_by_catalogue_gate": killed_by_catalogue_gate,
        "independence_ok": vce010_result.get("independence_ok"),
        "ambiguous_pair_ids": vce010_result.get("ambiguous_pair_ids"),
        "external_human_calibration_error": vce010_result.get(
            "external_human_calibration_error"
        ),
        "admission_divergence_rate": vce010_result.get("admission_divergence_rate"),
        "reason_family_confusion": reason_family_confusion,
        "authorized": authorized,
        "scope_disclaimer": (
            "Diagnostic-scale evidence scoped to the locked exp-sr-3 "
            "catalogue identity, computed entirely from VCE-010's own "
            "evaluator-calibration protocol (no forked mechanism). The "
            "external judge and its transport are a labeled mock (no real "
            "external-model credentials or network call); human raters are "
            "synthetic, seeded labels standing in for a real blinded pair "
            "study, per SLM-478's (SIE-005) frozen packet design. No real "
            "external-family judge or human calibration was performed in "
            "this environment -- SLM-483's own escape hatch: this is honest "
            "fixture/diagnostic-scale wiring evidence, not a "
            "construct-validity claim. No downstream experiment may cite "
            "'authorized' here as calibrated evidence for a promotion/risk "
            "claim; a real external-judge/blinded-human run remains "
            "required before any such claim."
        ),
    }
    artifact = store.write_artifact("sie006_calibration_result", result)
    store.append_event(
        "sie006_calibration_completed",
        experiment_id=campaign.campaign_id,
        status=vce010_result.get("disposition", "unknown"),
        artifact_sha256=artifact.stem,
        detail={
            "manifest_sha256": lock.manifest_sha256,
            "evaluator_human_agreement_kappa": kappa,
            "authorized": authorized,
        },
    )
    return result


def plan_only_preview() -> dict[str, Any]:
    return plan_only_manifest(CATALOGUE_ID)


__all__ = [
    "CATALOGUE_ID",
    "PRIMARY_METRIC",
    "Sie006CampaignV1",
    "plan_only_preview",
    "run_campaign",
]
