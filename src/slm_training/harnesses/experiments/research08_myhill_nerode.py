"""RESEARCH-08 (SLM-540): Myhill–Nerode static residual quotient campaign.

Default-off / research-only fixture campaign. Control = terminal-continuation
DFA from the current static compiled residual adapter. Treatment = deterministic
Myhill–Nerode quotient over that regular abstraction only. Never production
decode / ship-gate / serving authority.
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
    ExperimentCampaignV1,
    MultiplicityFamilyV1,
    campaign_manifest_sha256,
)
from slm_training.autoresearch.schemas import CampaignBudget, CampaignSpec
from slm_training.autoresearch.storage import CampaignStore
from slm_training.dsl.grammar.fastpath.myhill_nerode import (
    DEFAULT_OFF,
    run_research08_pilot,
)
from slm_training.dsl.grammar.fastpath.static_control_domain import static_lalr_adapter
from slm_training.harness_core.lineage.records import content_sha
from slm_training.levers import MAX_RUN_MINUTES
from slm_training.research_preregistry import (
    assert_execution_allowed,
    experiment_by_key,
)

EXPERIMENT_KEY = "RESEARCH-08"
LINEAR_ID = "SLM-540"
CAMPAIGN_ID = "research08-myhill-nerode-static-residual"
PRIMARY_METRIC = "cold_artifact_load_initialization_time"
_NOISE_FLOOR_MS = 0.01

__all__ = [
    "CAMPAIGN_ID",
    "EXPERIMENT_KEY",
    "LINEAR_ID",
    "PRIMARY_METRIC",
    "Research08CampaignV1",
    "plan_only_preview",
    "run_campaign",
]


@dataclass(frozen=True)
class Research08CampaignV1:
    campaign_id: str = CAMPAIGN_ID
    cold_trials: int = 31
    seed: int = 0
    source_commit: str = "0" * 40
    claim_class: str = "fixture"
    max_wall_minutes: float = float(MAX_RUN_MINUTES)
    enabled: bool = True  # harness enables; production paths stay default_off

    def __post_init__(self) -> None:
        if self.cold_trials < 1:
            raise ValueError("cold_trials must be >= 1")
        if self.claim_class != "fixture":
            raise ValueError("RESEARCH-08 only exposes fixture evidence")
        if self.max_wall_minutes <= 0 or self.max_wall_minutes > MAX_RUN_MINUTES:
            raise ValueError("max_wall_minutes must obey MAX_RUN_MINUTES")

    @property
    def fingerprint(self) -> str:
        return content_sha(asdict(self))

    def manifest(self) -> ExperimentCampaignV1:
        fingerprint = self.fingerprint
        return ExperimentCampaignV1(
            campaign_id=self.campaign_id,
            experiment_id=EXPERIMENT_KEY,
            hypothesis=(
                "Myhill-Nerode quotienting of the regular/static residual "
                "decoder graph reduces cold artifact size/load without "
                "changing accepted continuations."
            ),
            decision=(
                "Zero semantic disagreement and measurable cold-state "
                "improvement; else reject. Research-only / default_off — "
                "never production authority."
            ),
            endpoints=(
                CampaignEndpointV1(
                    endpoint_id="research08_cold_load",
                    metric=PRIMARY_METRIC,
                    role="primary",
                    direction="decrease",
                    minimum_effect=_NOISE_FLOOR_MS,
                ),
            ),
            arms=(
                CampaignArmV1(
                    arm_id="control_static_residual",
                    role="control",
                    config_sha256=content_sha(
                        {"campaign": fingerprint, "arm": "control"}
                    ),
                ),
                CampaignArmV1(
                    arm_id="treatment_myhill_nerode",
                    role="candidate",
                    config_sha256=content_sha(
                        {"campaign": fingerprint, "arm": "treatment"}
                    ),
                ),
            ),
            seeds=(self.seed,),
            budget=CampaignBudget(
                max_experiments=2,
                max_wall_minutes=self.max_wall_minutes,
            ),
            stopping_rules=(
                "Abort if language-equivalence or CompleteDomain parity fails.",
                "Reject when cold-load does not improve or state count is unchanged.",
            ),
            controls=(
                CampaignControlV1(
                    control_id="matched_static_adapter_control",
                    description=(
                        "Control and treatment derive from the same certified "
                        "StaticLalrAdapter; request-local overlays never enter."
                    ),
                    kind="quality",
                ),
                CampaignControlV1(
                    control_id="forged_coverage_complete_negative",
                    description=(
                        "A forged legacy coverageComplete bit alone must not "
                        "authorize CompleteDomain pruning (KERN-02)."
                    ),
                    kind="negative",
                ),
            ),
            negative_controls=("forged_coverage_complete_negative",),
            multiplicity_families=(
                MultiplicityFamilyV1(
                    family_id="research08_cold_family",
                    hypothesis_ids=("research08_cold_load",),
                    alpha=0.05,
                ),
            ),
            promotion_gates=(
                CampaignGateV1(
                    gate_id="fixture_only",
                    endpoint_id="research08_cold_load",
                    operator="ge",
                    threshold=0.0,
                ),
            ),
            rollback_gates=(
                CampaignGateV1(
                    gate_id="research08_semantic_disagreement",
                    endpoint_id="research08_cold_load",
                    operator="lt",
                    threshold=0.0,
                ),
            ),
            artifact_requirements=(
                ArtifactRequirementV1(kind="version_stamp"),
                ArtifactRequirementV1(kind="endpoint_result"),
            ),
            claim_class=self.claim_class,
            source_commit=self.source_commit,
            source_dirty=False,
            author="slm-training",
            created_at="1970-01-01T00:00:00Z",
        )


def plan_only_preview(campaign: Research08CampaignV1 | None = None) -> dict[str, Any]:
    campaign = campaign or Research08CampaignV1()
    manifest = campaign.manifest()
    row = experiment_by_key(EXPERIMENT_KEY)
    return {
        "schema": "research08_plan_only/v1",
        "experiment_key": EXPERIMENT_KEY,
        "linear_id": LINEAR_ID,
        "campaign_id": campaign.campaign_id,
        "claim_class": campaign.claim_class,
        "default_off": DEFAULT_OFF,
        "registry_disposition": row.get("disposition"),
        "registry_default_off": row.get("default_off"),
        "manifest_sha256": campaign_manifest_sha256(manifest),
        "primary_metric": PRIMARY_METRIC,
        "arms": [arm.arm_id for arm in manifest.arms],
        "production_authority": False,
    }


def run_campaign(
    campaign: Research08CampaignV1,
    *,
    root: Path,
) -> dict[str, Any]:
    """Lock ExperimentCampaignV1, run the pilot, return version-stampable payload."""

    row = experiment_by_key(EXPERIMENT_KEY)
    store = CampaignStore(campaign.campaign_id, root)
    store.initialize(
        CampaignSpec(
            campaign_id=campaign.campaign_id,
            objective="RESEARCH-08 Myhill–Nerode static residual quotient",
            primary_metric=PRIMARY_METRIC,
            budget=CampaignBudget(
                max_experiments=2,
                max_wall_minutes=campaign.max_wall_minutes,
            ),
            created_at="1970-01-01T00:00:00Z",
        )
    )
    lock = store.lock_experiment_campaign(campaign.manifest())
    if row.get("campaign_lock_sha256") == lock.manifest_sha256:
        assert_execution_allowed(row)
    elif row.get("campaign_lock_sha256") is None:
        pass
    else:
        raise RuntimeError(
            f"{EXPERIMENT_KEY}: registry campaign_lock_sha256 "
            f"{row.get('campaign_lock_sha256')!r} != live lock "
            f"{lock.manifest_sha256!r}"
        )

    adapter = static_lalr_adapter()
    pilot = run_research08_pilot(
        adapter, enabled=campaign.enabled, cold_trials=campaign.cold_trials
    )
    store.write_artifact("research08_myhill_nerode_result", pilot)
    store.append_event(
        "experiment_finished",
        experiment_id=EXPERIMENT_KEY,
        status=str(pilot.get("decision", "unknown")),
        detail={
            "decision": pilot.get("decision"),
            "zero_semantic_disagreement": pilot.get("zero_semantic_disagreement"),
            "cold_improved": (pilot.get("primary_metric") or {}).get("improved"),
        },
    )
    return {
        "schema": "research08_campaign_result/v1",
        "experiment_key": EXPERIMENT_KEY,
        "linear_id": LINEAR_ID,
        "campaign_id": campaign.campaign_id,
        "claim_class": campaign.claim_class,
        "default_off": DEFAULT_OFF,
        "production_authority": False,
        "manifest_sha256": lock.manifest_sha256,
        "pilot": pilot,
        "recommendation": pilot.get("decision"),
        "promotion": False,
    }
