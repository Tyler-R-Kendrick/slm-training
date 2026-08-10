"""VCE-009 (SLM-468): fixture-scale oracle/contrast evidence campaign.

Runs every VCE-004/VCE-005 oracle-intervention arm (no-op, one-factor per
factor, all-oracle, shuffled, destructive, random) over real baseline/oracle
``SemanticPlanV1`` pairs drawn from the VCE-006/007 ``openui_hard_valid_v1``
contrast corpus, cleaned first through VCE-008's leakage/dedup audit
(quarantined records excluded from the pool), and persists the resulting
``PlanInterventionRecordV1`` evidence through the governed
``ExperimentCampaignV1``/``CampaignStore`` contract with
``claim_class="fixture"``.

No new oracle-arm generator (VCE-004/005's ``PlanOracleSubstitutor`` /
``apply_plan_intervention`` / ``select_shuffled_oracle`` /
``build_baseline_intervention`` already own that), no new evidence schema
(``ExperimentCampaignV1``/``CampaignResultV1`` own that --
``claim_class="fixture"`` already machine-exempts this campaign from every
promotion-only validation branch in ``validate_result_claim``), no new
writer (``CampaignStore`` owns that), no new leakage audit (VCE-008's
``contrast_corpus_leakage_report`` owns cleaning the pool). This module only
composes them, mirroring PCT-008's shape (:mod:`pct008_artifact_cold_warm`).

Every plan in this corpus is genuinely ``identity.provenance == "gold"``, so
the one-factor/all-oracle/shuffled arms use ``plan_source="gold"`` under
``honesty_mode="oracle_diagnostic"`` -- an honest oracle-diagnostic ceiling
experiment, not a mislabeled "predicted" arm. This also means those three
arm families are genuinely contamination-bannered (AC: "Oracle contamination
banner survives all exports" is exercised on real non-``None`` banners, not
just plumbing); ``none``/``destructive``/``random`` stay uncontaminated by
construction, and ``filter_manifest_safe`` demonstrably drops only the
former.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, replace
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
)
from slm_training.autoresearch.schemas import CampaignBudget, CampaignSpec
from slm_training.autoresearch.storage import CampaignStore
from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.data.semantic_plan.oracle import (
    InterventionIdentityV1,
    OracleFactor,
    PlanInterventionRecordV1,
    PlanOracleSubstitutor,
    apply_plan_intervention,
    build_baseline_intervention,
    filter_manifest_safe,
    select_shuffled_oracle,
)
from slm_training.data.store import DataStore
from slm_training.harness_core.lineage.records import content_sha
from slm_training.levers import MAX_RUN_MINUTES

ALL_FACTORS: tuple[OracleFactor, ...] = ("archetype", "roles", "topology", "bindings")
ONE_FACTOR_ARMS: tuple[str, ...] = tuple(f"one_factor_{factor}" for factor in ALL_FACTORS)
ARM_IDS: tuple[str, ...] = (
    "none",
    *ONE_FACTOR_ARMS,
    "all_oracle",
    "shuffled",
    "destructive",
    "random",
)
_CONTAMINATED_ARMS = frozenset({*ONE_FACTOR_ARMS, "all_oracle", "shuffled"})

__all__ = [
    "ALL_FACTORS",
    "ARM_IDS",
    "ONE_FACTOR_ARMS",
    "Vce009CampaignV1",
    "run_campaign",
]


def _load_fixture_plan_pool(
    dataset_id: str = "openui_hard_valid_v1", *, limit: int
) -> list[tuple[str, SemanticPlanV1]]:
    """Real baseline/oracle plans from the frozen contrast corpus, cleaned
    through VCE-008's leakage audit (quarantined records excluded)."""
    from scripts.audit_data_corpora import contrast_corpus_leakage_report

    ref = DataStore().verify("eval", dataset_id)
    records_path = ref.path / "records.jsonl"
    leakage = contrast_corpus_leakage_report(dataset_id)
    quarantined_ids = {entry["record_id"] for entry in leakage["duplicates"]["quarantined"]}

    pool: list[tuple[str, SemanticPlanV1]] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            record_id = payload["record"]["id"]
            if record_id in quarantined_ids:
                continue
            pool.append((record_id, SemanticPlanV1.from_dict(payload["source_plan"])))
            if len(pool) >= limit:
                break
    return pool


def _generate_baseline_arms(
    record_id: str,
    baseline: SemanticPlanV1,
    other_candidates: list[SemanticPlanV1],
    *,
    rng_seed: int,
) -> dict[str, Any]:
    """Every declared arm for one baseline plan, or an honest skip report
    when no content-compatible oracle candidate exists in the pool."""
    designated_oracle = select_shuffled_oracle(baseline, other_candidates, rng_seed=rng_seed)
    shuffled_oracle = select_shuffled_oracle(baseline, other_candidates, rng_seed=rng_seed + 1)
    if designated_oracle is None or shuffled_oracle is None:
        return {"record_id": record_id, "skipped": True, "reason": "no compatible oracle candidate in pool"}

    identity = InterventionIdentityV1(seed=rng_seed)

    def _timed(build: Any) -> PlanInterventionRecordV1:
        start = time.perf_counter()
        record = build()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return replace(record, compute={"apply_ms": round(elapsed_ms, 4)})

    arm_records: dict[str, PlanInterventionRecordV1] = {
        "none": _timed(lambda: build_baseline_intervention(baseline, identity=identity, compute={})),
        "destructive": _timed(
            lambda: apply_plan_intervention(
                PlanOracleSubstitutor(plan_source="destructive", oracle_factors=ALL_FACTORS),
                baseline,
                rng_seed=rng_seed,
                identity=identity,
                compute={},
            )
        ),
        "random": _timed(
            lambda: apply_plan_intervention(
                PlanOracleSubstitutor(plan_source="random", oracle_factors=ALL_FACTORS),
                baseline,
                rng_seed=rng_seed,
                identity=identity,
                compute={},
            )
        ),
        "all_oracle": _timed(
            lambda: apply_plan_intervention(
                PlanOracleSubstitutor(
                    plan_source="gold", oracle_factors=ALL_FACTORS, honesty_mode="oracle_diagnostic"
                ),
                baseline,
                designated_oracle,
                rng_seed=rng_seed,
                identity=identity,
                compute={},
            )
        ),
        "shuffled": _timed(
            lambda: apply_plan_intervention(
                PlanOracleSubstitutor(
                    plan_source="gold", oracle_factors=ALL_FACTORS, honesty_mode="oracle_diagnostic"
                ),
                baseline,
                shuffled_oracle,
                rng_seed=rng_seed,
                identity=identity,
                compute={},
            )
        ),
    }
    for factor in ALL_FACTORS:
        arm_records[f"one_factor_{factor}"] = _timed(
            lambda factor=factor: apply_plan_intervention(
                PlanOracleSubstitutor(
                    plan_source="gold", oracle_factors=(factor,), honesty_mode="oracle_diagnostic"
                ),
                baseline,
                designated_oracle,
                rng_seed=rng_seed,
                identity=identity,
                compute={},
            )
        )

    return {
        "record_id": record_id,
        "skipped": False,
        "designated_oracle_equals_shuffled_oracle": (
            designated_oracle.to_dict() == shuffled_oracle.to_dict()
        ),
        "arms": arm_records,
    }


@dataclass(frozen=True)
class Vce009CampaignV1:
    campaign_id: str = "vce009-oracle-contrast-fixture"
    n_baseline_plans: int = 5
    pool_size: int = 40
    seed: int = 0
    source_commit: str = "0" * 40
    claim_class: str = "fixture"
    max_wall_minutes: float = float(MAX_RUN_MINUTES)

    def __post_init__(self) -> None:
        if self.n_baseline_plans < 1:
            raise ValueError("n_baseline_plans must be >= 1")
        if self.pool_size < self.n_baseline_plans + 1:
            raise ValueError("pool_size must exceed n_baseline_plans (need distinct oracle candidates)")
        if self.claim_class != "fixture":
            raise ValueError("VCE-009 only exposes the fixture evidence campaign")
        if self.max_wall_minutes <= 0 or self.max_wall_minutes > MAX_RUN_MINUTES:
            raise ValueError("max_wall_minutes must obey MAX_RUN_MINUTES")

    @property
    def fingerprint(self) -> str:
        return content_sha(asdict(self))

    def manifest(self) -> ExperimentCampaignV1:
        fingerprint = self.fingerprint
        arms = tuple(
            CampaignArmV1(
                arm_id=arm,
                role="control" if arm in {"none", "destructive", "random"} else "candidate",
                config_sha256=content_sha({"campaign": fingerprint, "arm": arm}),
            )
            for arm in ARM_IDS
        )
        return ExperimentCampaignV1(
            campaign_id=self.campaign_id,
            experiment_id=self.campaign_id,
            hypothesis=(
                "Every oracle-intervention arm declares exactly the factor(s) "
                "it changed, the no-op arm is hash-identical to baseline, and "
                "only genuinely gold-sourced arms carry a contamination banner "
                "-- an oracle-localization result over this fixture pool "
                "cannot be mistaken for a generic extra-compute, leakage, or "
                "intervention-pipeline effect."
            ),
            decision=(
                "Report per-arm intervention evidence honestly; no "
                "model-quality or promotion claim is made here."
            ),
            endpoints=(
                CampaignEndpointV1(
                    endpoint_id="declared_factor_fidelity_rate",
                    metric="fraction_of_records_touching_only_declared_factors",
                    role="primary",
                    direction="increase",
                    minimum_effect=0.0,
                ),
            ),
            arms=arms,
            seeds=(self.seed,),
            budget=CampaignBudget(
                max_experiments=self.n_baseline_plans * len(ARM_IDS),
                max_wall_minutes=self.max_wall_minutes,
            ),
            stopping_rules=(
                "A baseline with no content-compatible oracle candidate in the "
                "pool is skipped and reported, never fabricated.",
                "Every record is generated in-process; no subprocess timeout applies.",
            ),
            controls=(
                CampaignControlV1(
                    control_id="no_op_control",
                    description="plan_source=none must return baseline unchanged, byte-identical.",
                    kind="quality",
                ),
                CampaignControlV1(
                    control_id="destructive_random_uncontaminated_control",
                    description=(
                        "destructive/random arms never read real oracle content, so "
                        "they must never carry a contamination banner."
                    ),
                    kind="negative",
                ),
            ),
            negative_controls=("destructive_random_uncontaminated_control",),
            multiplicity_families=(
                MultiplicityFamilyV1(
                    family_id="oracle_intervention_fidelity",
                    hypothesis_ids=("declared_factor_fidelity_rate",),
                    alpha=0.05,
                ),
            ),
            promotion_gates=(
                # Fixture evidence is never a promotion claim; deliberately
                # permissive, mirrors PCT-008's fixture_only gate.
                CampaignGateV1(
                    gate_id="fixture_only",
                    endpoint_id="declared_factor_fidelity_rate",
                    operator="ge",
                    threshold=0.0,
                ),
            ),
            rollback_gates=(
                CampaignGateV1(
                    gate_id="declared_factor_fidelity_regression",
                    endpoint_id="declared_factor_fidelity_rate",
                    operator="lt",
                    threshold=1.0,
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


def run_campaign(campaign: Vce009CampaignV1, *, root: Path) -> dict[str, Any]:
    """Run the fixture campaign in-process and persist a ``CampaignResultV1``
    payload under campaign governance. Never spawns a subprocess."""
    store = CampaignStore(campaign.campaign_id, root)
    store.initialize(
        CampaignSpec(
            campaign_id=campaign.campaign_id,
            objective="Fixture-scale oracle/contrast evidence (VCE-009)",
            primary_metric="declared_factor_fidelity_rate",
            budget=CampaignBudget(
                max_experiments=campaign.n_baseline_plans * len(ARM_IDS),
                max_wall_minutes=campaign.max_wall_minutes,
            ),
            created_at="1970-01-01T00:00:00Z",
        )
    )
    lock = store.lock_experiment_campaign(campaign.manifest())

    pool = _load_fixture_plan_pool(limit=campaign.pool_size)
    candidates = [plan for _, plan in pool]
    baseline_reports: list[dict[str, Any]] = []
    all_records: list[PlanInterventionRecordV1] = []
    for index, (record_id, baseline) in enumerate(pool[: campaign.n_baseline_plans]):
        other_candidates = [plan for i, plan in enumerate(candidates) if i != index]
        report = _generate_baseline_arms(record_id, baseline, other_candidates, rng_seed=campaign.seed)
        baseline_reports.append(report)
        if not report["skipped"]:
            all_records.extend(report["arms"].values())

    touched_only_declared = sum(1 for r in all_records if r.touches_only_declared_factors())
    fidelity_rate = touched_only_declared / len(all_records) if all_records else 0.0
    none_records = [
        r for report in baseline_reports if not report["skipped"] for arm, r in report["arms"].items() if arm == "none"
    ]
    no_op_holds = all(r.is_no_op() for r in none_records)
    contaminated = [r for r in all_records if r.is_contaminated]
    safe_records = filter_manifest_safe(all_records)
    contaminated_arm_ids = {
        arm
        for report in baseline_reports
        if not report["skipped"]
        for arm, r in report["arms"].items()
        if r.is_contaminated
    }

    result = {
        "schema": "vce009_oracle_contrast_fixture_result/v1",
        "campaign_id": campaign.campaign_id,
        "manifest_sha256": lock.manifest_sha256,
        "claim_class": campaign.claim_class,
        "baselines_attempted": campaign.n_baseline_plans,
        "baselines_skipped": sum(1 for r in baseline_reports if r["skipped"]),
        "declared_factor_fidelity_rate": round(fidelity_rate, 6),
        "no_op_control_holds": no_op_holds,
        "records_total": len(all_records),
        "records_contaminated": len(contaminated),
        "records_safe_for_export": len(safe_records),
        "contaminated_arm_ids": sorted(contaminated_arm_ids),
        "expected_contaminated_arm_ids": sorted(_CONTAMINATED_ARMS),
        "baseline_reports": [
            {
                "record_id": report["record_id"],
                "skipped": report["skipped"],
                **(
                    {
                        "designated_oracle_equals_shuffled_oracle": report[
                            "designated_oracle_equals_shuffled_oracle"
                        ],
                        "arms": {arm: record.to_dict() for arm, record in report["arms"].items()},
                    }
                    if not report["skipped"]
                    else {"reason": report["reason"]}
                ),
            }
            for report in baseline_reports
        ],
        "scope_disclaimer": (
            "Fixture-scale oracle-intervention evidence over a small pool of "
            "real, cleaned (VCE-008-audited) baseline/oracle plan pairs. No "
            "model, verifier, or search cost is measured -- 'compute' fields "
            "are pure in-process apply() wall-clock. No production-speedup or "
            "model-quality claim is made."
        ),
    }
    artifact = store.write_artifact("vce009_oracle_contrast_result", result)
    store.append_event(
        "vce009_oracle_contrast_completed",
        experiment_id=campaign.campaign_id,
        status="fixture",
        artifact_sha256=artifact.stem,
        detail={"manifest_sha256": lock.manifest_sha256},
    )
    return result
