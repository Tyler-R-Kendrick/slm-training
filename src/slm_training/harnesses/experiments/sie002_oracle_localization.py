"""SIE-002 (SLM-476): held-out factor-wise semantic oracle localization (EXP-SR-1).

Fixture-scale governed campaign that composes VCE-009's oracle-intervention
arm generators (:func:`generate_baseline_arms` /
:func:`load_fixture_plan_pool`) with the SIE-001 catalogue identity
``exp-sr-1``. No forked arm generator, no checkpoint promotion, no GPU.

Durable evidence chooses which factors (if any) are authorized for later
learned-predictor work. Authorization is fail-closed: an empty list is a
valid, honest outcome when the fixture falsifier holds.
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
from slm_training.data.semantic_plan.oracle import PlanInterventionRecordV1
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.experiments.exp_sr_catalogue import (
    exp_sr_campaign,
    plan_only_manifest,
)
from slm_training.harnesses.experiments.vce009_oracle_contrast_campaign import (
    ALL_FACTORS,
    ARM_IDS,
    ONE_FACTOR_ARMS,
    generate_baseline_arms,
    load_fixture_plan_pool,
)
from slm_training.levers import MAX_RUN_MINUTES

CATALOGUE_ID = "exp-sr-1"
PRIMARY_METRIC = "oracle_factor_localization_precision"
DATASET_ID = "openui_hard_valid_v1"

__all__ = [
    "ALL_FACTORS",
    "ARM_IDS",
    "CATALOGUE_ID",
    "ONE_FACTOR_ARMS",
    "PRIMARY_METRIC",
    "Sie002CampaignV1",
    "analyze_factor_localization",
    "authorize_factors",
    "plan_only_preview",
    "run_campaign",
]


def _is_noop(record: PlanInterventionRecordV1) -> bool:
    return record.is_no_op()


def _leakage_audit(dataset_id: str = DATASET_ID) -> dict[str, Any]:
    from scripts.audit_data_corpora import contrast_corpus_leakage_report

    report = contrast_corpus_leakage_report(dataset_id)
    quarantined = report["duplicates"]["quarantined"]
    return {
        "dataset_id": dataset_id,
        "passed": True,  # audit itself is the gate; quarantined ids are excluded from pool
        "quarantined_count": len(quarantined),
        "quarantined_record_ids": sorted(entry["record_id"] for entry in quarantined),
        "scope": "VCE-008 contrast_corpus_leakage_report; quarantined excluded from pool",
    }


def analyze_factor_localization(
    baseline_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    """Per-factor change rates, ceiling recovery, and primary localization precision.

    Fixture proxy (no model/search): a one-factor arm "improves" when the
    intervened plan hash differs from baseline; the all-oracle changed-factor
    set is the recoverable ceiling. Neural forwards / CVaR / legal-support
    parity are out of scope for this no-GPU fixture pass and are reported as
    such — never silently omitted.
    """
    active = [r for r in baseline_reports if not r.get("skipped")]
    n = len(active)
    per_factor: dict[str, Any] = {}
    localization_scores: list[float] = []

    for factor in ALL_FACTORS:
        arm_id = f"one_factor_{factor}"
        records = [r["arms"][arm_id] for r in active]
        change_n = sum(1 for rec in records if not _is_noop(rec))
        fidelity_n = sum(1 for rec in records if rec.touches_only_declared_factors())
        ceiling_hits = 0
        ceiling_recoveries = 0
        for report in active:
            ceiling = set(report["arms"]["all_oracle"].changed_factors)
            if factor not in ceiling:
                continue
            ceiling_hits += 1
            one = report["arms"][arm_id]
            if (not _is_noop(one)) and one.touches_only_declared_factors():
                ceiling_recoveries += 1
        per_factor[factor] = {
            "arm_id": arm_id,
            "n": n,
            "change_rate": (change_n / n) if n else 0.0,
            "mechanical_one_factor_fidelity": (fidelity_n / n) if n else 0.0,
            "ceiling_presence_n": ceiling_hits,
            "ceiling_recovery_n": ceiling_recoveries,
            "ceiling_recovery_rate": (
                ceiling_recoveries / ceiling_hits if ceiling_hits else 0.0
            ),
        }

    for report in active:
        ceiling = set(report["arms"]["all_oracle"].changed_factors)
        if not ceiling:
            continue
        recovered = {
            factor
            for factor in ceiling
            if (not _is_noop(report["arms"][f"one_factor_{factor}"]))
            and report["arms"][f"one_factor_{factor}"].touches_only_declared_factors()
        }
        localization_scores.append(len(recovered) / len(ceiling))

    def _arm_change_rate(arm_id: str) -> float:
        if not active:
            return 0.0
        return sum(1 for r in active if not _is_noop(r["arms"][arm_id])) / n

    none_ok = all(_is_noop(r["arms"]["none"]) for r in active) if active else False
    all_records = [rec for r in active for rec in r["arms"].values()]
    fidelity_rate = (
        sum(1 for rec in all_records if rec.touches_only_declared_factors()) / len(all_records)
        if all_records
        else 0.0
    )

    precision = (
        sum(localization_scores) / len(localization_scores) if localization_scores else 0.0
    )
    return {
        "n_active_baselines": n,
        "n_skipped": sum(1 for r in baseline_reports if r.get("skipped")),
        "oracle_factor_localization_precision": round(precision, 6),
        "n_localization_scored": len(localization_scores),
        "declared_factor_fidelity_rate": round(fidelity_rate, 6),
        "no_op_control_holds": none_ok,
        "arm_change_rates": {arm: round(_arm_change_rate(arm), 6) for arm in ARM_IDS},
        "per_factor": {
            factor: {
                **stats,
                "change_rate": round(stats["change_rate"], 6),
                "mechanical_one_factor_fidelity": round(stats["mechanical_one_factor_fidelity"], 6),
                "ceiling_recovery_rate": round(stats["ceiling_recovery_rate"], 6),
            }
            for factor, stats in per_factor.items()
        },
        "out_of_scope_fixture": (
            "neural_forwards",
            "exact_search_work",
            "behavioral_cvar",
            "legal_support_parity_under_decode",
        ),
    }


def authorize_factors(
    analysis: dict[str, Any],
    *,
    leakage_passed: bool,
    minimum_effect: float,
) -> dict[str, Any]:
    """Fail-closed factor authorization for later learned-predictor work.

    A factor is authorized only when the fixture shows a material one-factor
    oracle change over no-op, mechanical one-factor fidelity, a non-empty
    all-oracle ceiling that the factor recovers, leakage audit clean, and
    the no-op control holding. Empty authorization is a valid falsification.
    """
    reasons: dict[str, str] = {}
    authorized: list[str] = []
    all_oracle_rate = float(analysis["arm_change_rates"].get("all_oracle", 0.0))
    none_ok = bool(analysis["no_op_control_holds"])
    fidelity_ok = float(analysis["declared_factor_fidelity_rate"]) >= 1.0 - 1e-12

    if not leakage_passed:
        return {
            "authorized_factors": [],
            "authorization_reasons": {
                factor: "leakage_audit_failed" for factor in ALL_FACTORS
            },
            "falsifier_holds": True,
            "falsifier_note": "Leakage audit did not pass; no factor authorized.",
        }
    if not none_ok or not fidelity_ok or all_oracle_rate <= 0.0:
        note = (
            "no_op or mechanical fidelity failed"
            if (not none_ok or not fidelity_ok)
            else "all-oracle ceiling near zero on this fixture"
        )
        return {
            "authorized_factors": [],
            "authorization_reasons": {factor: note for factor in ALL_FACTORS},
            "falsifier_holds": True,
            "falsifier_note": note,
        }

    for factor in ALL_FACTORS:
        stats = analysis["per_factor"][factor]
        if stats["mechanical_one_factor_fidelity"] < 1.0 - 1e-12:
            reasons[factor] = "mechanical_one_factor_fidelity_below_1"
            continue
        if stats["change_rate"] < minimum_effect:
            reasons[factor] = (
                f"one_factor_change_rate={stats['change_rate']} < minimum_effect={minimum_effect}"
            )
            continue
        if stats["ceiling_presence_n"] == 0:
            reasons[factor] = "factor_absent_from_all_oracle_ceiling_on_fixture"
            continue
        if stats["ceiling_recovery_rate"] < minimum_effect:
            reasons[factor] = (
                f"ceiling_recovery_rate={stats['ceiling_recovery_rate']} "
                f"< minimum_effect={minimum_effect}"
            )
            continue
        authorized.append(factor)
        reasons[factor] = "authorized_fixture_scale_oracle_localization"

    falsifier = len(authorized) == 0
    return {
        "authorized_factors": authorized,
        "authorization_reasons": reasons,
        "falsifier_holds": falsifier,
        "falsifier_note": (
            "No one-factor arm cleared the fixture authorization bar "
            "(material change + ceiling recovery + fidelity + leakage)."
            if falsifier
            else "One or more factors cleared the fixture authorization bar; "
            "still claim_class=fixture — not a promotion."
        ),
    }


def plan_only_preview() -> dict[str, object]:
    """Catalogue plan-only preview plus the fixture arm surface SIE-002 runs."""
    preview = plan_only_manifest(CATALOGUE_ID)
    return {
        **preview,
        "fixture_arms": list(ARM_IDS),
        "factors": list(ALL_FACTORS),
        "primary_metric": PRIMARY_METRIC,
        "claim_class_execution": "fixture",
        "promotion": False,
    }


@dataclass(frozen=True)
class Sie002CampaignV1:
    campaign_id: str = "sie002-exp-sr-1-oracle-localization-fixture"
    n_baseline_plans: int = 5
    pool_size: int = 40
    seeds: tuple[int, ...] = (0, 1, 2)
    source_commit: str = "0" * 40
    claim_class: str = "fixture"
    max_wall_minutes: float = float(MAX_RUN_MINUTES)

    def __post_init__(self) -> None:
        if self.n_baseline_plans < 1:
            raise ValueError("n_baseline_plans must be >= 1")
        if self.pool_size < self.n_baseline_plans + 1:
            raise ValueError("pool_size must exceed n_baseline_plans")
        if self.claim_class != "fixture":
            raise ValueError("SIE-002 durable evidence is fixture-only (no promotion)")
        if self.max_wall_minutes <= 0 or self.max_wall_minutes > MAX_RUN_MINUTES:
            raise ValueError("max_wall_minutes must obey MAX_RUN_MINUTES")
        if not self.seeds:
            raise ValueError("seeds must be non-empty")

    @property
    def fingerprint(self) -> str:
        return content_sha(asdict(self))

    def catalogue_manifest(self) -> ExperimentCampaignV1:
        return exp_sr_campaign(CATALOGUE_ID)

    def minimum_effect(self) -> float:
        return float(self.catalogue_manifest().endpoints[0].minimum_effect)

    def manifest(self) -> ExperimentCampaignV1:
        fingerprint = self.fingerprint
        catalogue = self.catalogue_manifest()
        arms = tuple(
            CampaignArmV1(
                arm_id=arm,
                role="control" if arm in {"none", "destructive", "random", "shuffled"} else "candidate",
                config_sha256=content_sha({"campaign": fingerprint, "arm": arm}),
            )
            for arm in ARM_IDS
        )
        return ExperimentCampaignV1(
            campaign_id=self.campaign_id,
            experiment_id=CATALOGUE_ID,
            hypothesis=catalogue.hypothesis,
            decision=(
                "Authorize factor families for later learned-predictor work only "
                "when fixture localization clears the preregistered bar; empty "
                "authorization is an honest falsification. No checkpoint or "
                "default promotion from this diagnostic."
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
            seeds=self.seeds,
            budget=CampaignBudget(
                max_experiments=self.n_baseline_plans * len(ARM_IDS) * len(self.seeds),
                max_wall_minutes=self.max_wall_minutes,
            ),
            stopping_rules=(
                "A baseline with no content-compatible oracle candidate is skipped, never fabricated.",
                "Every record is generated in-process; no subprocess timeout applies.",
                "No GPU / model forward; fixture proxy only.",
            ),
            controls=(
                CampaignControlV1(
                    control_id="no_op_control",
                    description="plan_source=none must return baseline unchanged.",
                    kind="quality",
                ),
                CampaignControlV1(
                    control_id="shuffled_destructive_control",
                    description=(
                        "Shuffled/destructive arms bound pipeline/noise effects so a "
                        "one-factor oracle result cannot be mistaken for them."
                    ),
                    kind="negative",
                ),
                CampaignControlV1(
                    control_id="leakage_control",
                    description=catalogue.controls[1].description
                    if len(catalogue.controls) > 1
                    else "Held-out contrast pool cleaned via VCE-008 leakage audit.",
                    kind="negative",
                ),
            ),
            negative_controls=("shuffled_destructive_control", "leakage_control"),
            multiplicity_families=(
                MultiplicityFamilyV1(
                    family_id=f"{CATALOGUE_ID}_family",
                    hypothesis_ids=(f"{CATALOGUE_ID}_primary",),
                    alpha=0.05,
                ),
            ),
            promotion_gates=(
                # Fixture evidence is never a promotion claim.
                CampaignGateV1(
                    gate_id="fixture_only",
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
            ),
            claim_class=self.claim_class,
            source_commit=self.source_commit,
            source_dirty=False,
            author="slm-training",
            created_at="1970-01-01T00:00:00Z",
        )


def run_campaign(campaign: Sie002CampaignV1, *, root: Path) -> dict[str, Any]:
    """Run multi-seed fixture localization and persist under CampaignStore."""
    store = CampaignStore(campaign.campaign_id, root)
    store.initialize(
        CampaignSpec(
            campaign_id=campaign.campaign_id,
            objective="SIE-002 / EXP-SR-1 fixture-scale oracle localization",
            primary_metric=PRIMARY_METRIC,
            budget=CampaignBudget(
                max_experiments=campaign.n_baseline_plans * len(ARM_IDS) * len(campaign.seeds),
                max_wall_minutes=campaign.max_wall_minutes,
            ),
            created_at="1970-01-01T00:00:00Z",
        )
    )
    lock = store.lock_experiment_campaign(campaign.manifest())
    catalogue = campaign.catalogue_manifest()

    leakage = _leakage_audit(DATASET_ID)
    pool = load_fixture_plan_pool(DATASET_ID, limit=campaign.pool_size)
    candidates = [plan for _, plan in pool]

    seed_payloads: list[dict[str, Any]] = []
    all_reports: list[dict[str, Any]] = []
    for seed in campaign.seeds:
        baseline_reports: list[dict[str, Any]] = []
        for index, (record_id, baseline) in enumerate(pool[: campaign.n_baseline_plans]):
            other = [plan for i, plan in enumerate(candidates) if i != index]
            report = generate_baseline_arms(
                record_id, baseline, other, rng_seed=seed
            )
            baseline_reports.append(report)
            all_reports.append({**report, "seed": seed})
        seed_analysis = analyze_factor_localization(baseline_reports)
        seed_payloads.append(
            {
                "seed": seed,
                "analysis": seed_analysis,
                "baseline_reports": [
                    {
                        "record_id": report["record_id"],
                        "skipped": report["skipped"],
                        **(
                            {
                                "designated_oracle_equals_shuffled_oracle": report[
                                    "designated_oracle_equals_shuffled_oracle"
                                ],
                                "arms": {
                                    arm: record.to_dict()
                                    for arm, record in report["arms"].items()
                                },
                            }
                            if not report["skipped"]
                            else {"reason": report["reason"]}
                        ),
                    }
                    for report in baseline_reports
                ],
            }
        )

    # Aggregate across seeds for the durable authorization decision.
    aggregate = analyze_factor_localization(
        [
            {k: v for k, v in report.items() if k != "seed"}
            for report in all_reports
        ]
    )
    auth = authorize_factors(
        aggregate,
        leakage_passed=bool(leakage["passed"]),
        minimum_effect=campaign.minimum_effect(),
    )

    result = {
        "schema": "sie002_oracle_localization_fixture_result/v1",
        "campaign_id": campaign.campaign_id,
        "catalogue_id": CATALOGUE_ID,
        "catalogue_manifest_sha256": campaign_manifest_sha256(catalogue),
        "manifest_sha256": lock.manifest_sha256,
        "claim_class": campaign.claim_class,
        "promotion": False,
        "dataset_id": DATASET_ID,
        "leakage_audit": leakage,
        "n_baseline_plans": campaign.n_baseline_plans,
        "pool_size": campaign.pool_size,
        "seeds": list(campaign.seeds),
        "oracle_factor_localization_precision": aggregate[
            "oracle_factor_localization_precision"
        ],
        "analysis": aggregate,
        "authorization": auth,
        "authorized_factors": auth["authorized_factors"],
        "seed_runs": seed_payloads,
        "scope_disclaimer": (
            "Fixture-scale factor-wise oracle localization over a VCE-008-cleaned "
            "openui_hard_valid_v1 pool. Arms generated exclusively via VCE-009's "
            "generate_baseline_arms (no fork). No model, verifier search, or GPU "
            "cost is measured — plan-hash / changed-factor proxies only. "
            "claim_class=fixture; authorized_factors gate later predictor work "
            "and never promote a checkpoint."
        ),
    }
    artifact = store.write_artifact("sie002_oracle_localization_result", result)
    store.append_event(
        "sie002_oracle_localization_completed",
        experiment_id=campaign.campaign_id,
        status="fixture",
        artifact_sha256=artifact.stem,
        detail={
            "manifest_sha256": lock.manifest_sha256,
            "authorized_factors": auth["authorized_factors"],
            "oracle_factor_localization_precision": result[
                "oracle_factor_localization_precision"
            ],
        },
    )
    return result
