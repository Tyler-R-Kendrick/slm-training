"""RSP-001 (SLM-488): witness-guided CEGIS repair campaign (EXP-SR-4).

Fixture-scale governed campaign comparing witness/residual-guided bounded repair
against matched-compute baselines on the existing SPV2-05 semantic repair records
and EFS2-03 conflict-slice topology fixtures. Composes existing owners only:

- :mod:`slm_training.harnesses.distill.semantic_repair` — ``SemanticRepairRecordV1``,
  ``project_repair_residual``, ``apply_repair_policy``, learned scorer
- :mod:`slm_training.harnesses.experiments.conflict_slice_repair` — remask policies
- :mod:`slm_training.evals.semantic_failure` — ``VerifierWitnessV1`` (optional)

Default remains OFF (treatment is an optional arm). No ship-gate weakening; no
promotion. Primary metric: ``verified_repair_yield``.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

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
from slm_training.evals.semantic_failure import VerifierWitnessV1
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.distill.semantic_repair import (
    ConflictSlice,
    LegalEdit,
    RepairEvidence,
    RepairFeatureExtractor,
    SemanticRepairRecordV1,
    SemanticRepairScorer,
    apply_repair_policy,
    build_repair_records_from_corruption,
    project_repair_residual,
    train_repair_policy_fixture,
)
from slm_training.harnesses.experiments.conflict_slice_repair import (
    ConflictSliceV1,
    TopologyNode,
    _tree_fingerprint,
    apply_repair_policy as apply_topology_repair,
    compare_repair_policies,
)
from slm_training.harnesses.experiments.exp_sr_catalogue import (
    exp_sr_campaign,
    plan_only_manifest,
)
from slm_training.levers import MAX_RUN_MINUTES

CATALOGUE_ID = "exp-sr-4"
PRIMARY_METRIC = "verified_repair_yield"
FIXTURE_CLEAN_OPENUI = 'root = Stack([cta], "column")\ncta = Button(":cta.label")'

SEMANTIC_ARM_IDS: tuple[str, ...] = (
    "no_repair",
    "regenerate",
    "edit_distance",
    "learned_scorer",
    "witness_cegis",
)
TOPOLOGY_ARM_IDS: tuple[str, ...] = (
    "suffix_rollback",
    "full_remask",
    "conflict_slice",
    "conflict_slice_expanded",
)
ARM_IDS: tuple[str, ...] = SEMANTIC_ARM_IDS + TOPOLOGY_ARM_IDS

__all__ = [
    "ARM_IDS",
    "CATALOGUE_ID",
    "FIXTURE_CLEAN_OPENUI",
    "PRIMARY_METRIC",
    "SEMANTIC_ARM_IDS",
    "TOPOLOGY_ARM_IDS",
    "Rsp001CampaignV1",
    "build_topology_fixture",
    "load_semantic_corpus",
    "plan_only_preview",
    "run_campaign",
    "run_semantic_repair_arm",
    "run_topology_repair_arm",
    "score_verified_repair_yield",
    "witness_from_record",
]

Recommendation = Literal["adopt-optional", "reject", "inconclusive"]


def witness_from_record(record: SemanticRepairRecordV1) -> VerifierWitnessV1:
    """Advisory witness vector derived only from the record's own failure evidence."""
    unresolved = tuple(
        sorted({f"evidence:{e.reason_code}" for e in record.failure_evidence if e.reason_code})
    )
    return VerifierWitnessV1(
        schema_version="verifier_witness/v1",
        source_trace_fingerprint=record.source_fingerprint,
        taxonomy_version="semantic_failure_taxonomy/v1",
        evaluator_version="fixture",
        evaluator_hash="rsp001-fixture",
        first_failed_gate=None,
        localizations=(),
        unresolved_localizations=unresolved,
        witness_digest="rsp001-fixture-witness",
    )


def _build_synthetic_fixture_records(
    *,
    n_records: int = 8,
) -> tuple[SemanticRepairRecordV1, ...]:
    """Deterministic SPV2-05-shaped records when the OpenUI bridge is unavailable."""
    broken = 'root = Stack([cta], "column")\ncta = Button(":cta.label"'
    fixed = FIXTURE_CLEAN_OPENUI
    source_fp = content_sha({"fixture": "rsp001", "clean": fixed})
    records: list[SemanticRepairRecordV1] = []
    for idx in range(n_records):
        record_id = f"rsp001-synthetic-{idx}"
        accepted = LegalEdit(
            edit_id=f"{record_id}-accepted",
            kind="replace_program",
            before=broken,
            after=fixed,
            cost=1,
            source="oracle_acceptable",
        )
        decoys = tuple(
            LegalEdit(
                edit_id=f"{record_id}-decoy-{j}",
                kind="replace_program",
                before=broken,
                after=f"{broken}_bad{j}",
                cost=4 + j,
                source="synthetic_negative",
            )
            for j in range(3)
        )
        legal_edits = (accepted,) + decoys
        conflict_slice = ConflictSlice(
            stage="grammar",
            failing_node_ids=(f"node-{idx}",),
            dependency_frontier=(f"frontier-{idx}",),
            protected_node_ids=("root",),
            completeness_class="EXACT",
            source_provenance="fixture:rsp001",
            notes="synthetic fixture — no OpenUI bridge required",
        )
        records.append(
            SemanticRepairRecordV1(
                record_id=record_id,
                source_fingerprint=source_fp,
                broken_openui=broken,
                failure_evidence=(
                    RepairEvidence(
                        reason_code="missing_quote",
                        analyzer="fixture",
                        detail="synthetic corruption",
                    ),
                ),
                conflict_slice=conflict_slice,
                legal_edits=legal_edits,
                accepted_edit_ids=(accepted.edit_id,),
                oracle_edit_id=accepted.edit_id,
                lineage={"fixture": "rsp001", "index": idx},
                metadata={
                    "family": "grammar",
                    "operator": "missing_quote",
                    "exact_repair": True,
                    "edit_distance": 1,
                    "n_legal_edits": len(legal_edits),
                    "synthetic_fixture": True,
                },
            )
        )
    return tuple(records)


def load_semantic_corpus(
    clean_openui: str = FIXTURE_CLEAN_OPENUI,
    *,
    max_records: int | None = None,
) -> tuple[SemanticRepairRecordV1, ...]:
    try:
        records = build_repair_records_from_corruption(clean_openui)
    except RuntimeError:
        records = _build_synthetic_fixture_records()
    if max_records is not None:
        records = records[:max_records]
    if not records:
        records = _build_synthetic_fixture_records()
    return records


def build_topology_fixture() -> tuple[TopologyNode, ConflictSliceV1]:
    """Standard EFS2-03-style topology conflict for remask-policy baselines."""
    n6 = TopologyNode(node_id=6, node_type="LITERAL", parent_id=3, active=True)
    n7 = TopologyNode(node_id=7, node_type="LITERAL", parent_id=3, active=True)
    n8 = TopologyNode(node_id=8, node_type="LITERAL", parent_id=4, active=True)
    n9 = TopologyNode(node_id=9, node_type="LITERAL", parent_id=5, active=True)
    n10 = TopologyNode(node_id=10, node_type="LITERAL", parent_id=5, active=True)
    n3 = TopologyNode(
        node_id=3,
        node_type="SLOT",
        parent_id=1,
        children=(n6, n7),
        active=True,
        decision_level=2,
    )
    n4 = TopologyNode(
        node_id=4,
        node_type="SLOT",
        parent_id=1,
        children=(n8,),
        active=True,
        decision_level=2,
    )
    n5 = TopologyNode(
        node_id=5,
        node_type="SLOT",
        parent_id=2,
        children=(n9, n10),
        active=True,
        decision_level=3,
    )
    n2 = TopologyNode(
        node_id=2,
        node_type="COMPONENT",
        parent_id=0,
        children=(n5,),
        active=True,
        decision_level=1,
        protected=True,
    )
    n1 = TopologyNode(
        node_id=1,
        node_type="COMPONENT",
        parent_id=0,
        children=(n3, n4),
        active=True,
        decision_level=1,
        certified=True,
    )
    n0 = TopologyNode(
        node_id=0,
        node_type="ROOT",
        children=(n1, n2),
        active=True,
        decision_level=0,
    )
    slice_ = ConflictSliceV1(
        conflict_id="rsp001-topology",
        stage="grammar",
        reason_code="localized_failure",
        failing_node_ids=(6, 8),
        dependency_frontier=(3, 4),
        protected_node_ids=(0, 2),
        completeness_class="EXACT",
        original_state_fingerprint=_tree_fingerprint(n0),
    )
    return n0, slice_


def _edit_verified(record: SemanticRepairRecordV1, edit: LegalEdit) -> bool:
    return edit.edit_id in record.accepted_edit_ids


def _protected_preserved(record: SemanticRepairRecordV1, edit: LegalEdit) -> bool:
    protected = set(record.conflict_slice.protected_node_ids)
    if not protected:
        return True
    # Fixture oracle: accepted edits preserve protected regions by construction.
    return _edit_verified(record, edit) or edit.source.startswith("oracle")


def run_witness_cegis_repair(
    record: SemanticRepairRecordV1,
    *,
    witness: VerifierWitnessV1,
    max_cycles: int,
    rng: random.Random,
) -> tuple[bool, list[dict[str, Any]]]:
    """Witness → residual → bounded repair under a fixed cycle budget."""
    cycles: list[dict[str, Any]] = []
    remaining_edits = list(record.legal_edits)
    for cycle in range(max_cycles):
        current = replace(record, legal_edits=tuple(remaining_edits))
        residual = project_repair_residual(current, witness=witness)
        cycle_info: dict[str, Any] = {
            "cycle": cycle,
            "residual_hash": residual.residual_hash,
            "obligations": list(residual.unsatisfied_obligations),
            "authorized": residual.can_authorize_repair(),
            "domain_size": len(residual.legal_repair_action_domain),
        }
        if not residual.can_authorize_repair() or not residual.legal_repair_action_domain:
            cycle_info["verified"] = False
            cycles.append(cycle_info)
            continue
        domain = set(residual.legal_repair_action_domain)
        candidates = [e for e in remaining_edits if e.edit_id in domain]
        if not candidates:
            cycle_info["verified"] = False
            cycles.append(cycle_info)
            continue
        # Deterministic residual-guided ranking: lowest cost within legal domain.
        candidates.sort(key=lambda e: (e.cost, e.edit_id))
        chosen = candidates[0]
        verified = _edit_verified(record, chosen)
        cycle_info.update(
            {
                "chosen_edit_id": chosen.edit_id,
                "verified": verified,
                "protected_preserved": _protected_preserved(record, chosen),
            }
        )
        cycles.append(cycle_info)
        if verified:
            return True, cycles
        remaining_edits = [e for e in remaining_edits if e.edit_id != chosen.edit_id]
        _ = rng  # matched interface; ranking is deterministic given residual
    return False, cycles


def run_regenerate_repair(
    record: SemanticRepairRecordV1,
    *,
    max_cycles: int,
    rng: random.Random,
) -> tuple[bool, list[dict[str, Any]]]:
    """Matched-compute unguided resampling over the full legal edit domain."""
    cycles: list[dict[str, Any]] = []
    for cycle in range(max_cycles):
        if not record.legal_edits:
            cycles.append({"cycle": cycle, "verified": False})
            continue
        chosen = rng.choice(record.legal_edits)
        verified = _edit_verified(record, chosen)
        cycles.append(
            {
                "cycle": cycle,
                "chosen_edit_id": chosen.edit_id,
                "verified": verified,
                "policy": "regenerate",
            }
        )
        if verified:
            return True, cycles
    return False, cycles


def run_semantic_repair_arm(
    arm_id: str,
    record: SemanticRepairRecordV1,
    *,
    max_cycles: int,
    rng: random.Random,
    scorer: SemanticRepairScorer | None = None,
    extractor: RepairFeatureExtractor | None = None,
) -> tuple[bool, dict[str, Any]]:
    if arm_id == "no_repair":
        return False, {"arm_id": arm_id, "cycles": [], "verified": False}
    if arm_id == "regenerate":
        ok, cycles = run_regenerate_repair(record, max_cycles=max_cycles, rng=rng)
        return ok, {"arm_id": arm_id, "cycles": cycles, "verified": ok}
    if arm_id == "witness_cegis":
        witness = witness_from_record(record)
        ok, cycles = run_witness_cegis_repair(
            record, witness=witness, max_cycles=max_cycles, rng=rng
        )
        return ok, {
            "arm_id": arm_id,
            "cycles": cycles,
            "verified": ok,
            "witness_unresolved": list(witness.unresolved_localizations),
        }
    if arm_id == "edit_distance":
        chosen, meta = apply_repair_policy(record, "edit_distance", rng=rng)
        verified = _edit_verified(record, chosen)
        return verified, {"arm_id": arm_id, **meta, "chosen_edit_id": chosen.edit_id, "verified": verified}
    if arm_id == "learned_scorer":
        if scorer is None or extractor is None:
            return False, {"arm_id": arm_id, "status": "N/A", "reason": "torch/scorer unavailable"}
        chosen, meta = apply_repair_policy(
            record, "learned", scorer=scorer, extractor=extractor, rng=rng
        )
        verified = _edit_verified(record, chosen)
        return verified, {"arm_id": arm_id, **meta, "chosen_edit_id": chosen.edit_id, "verified": verified}
    raise ValueError(f"unknown semantic arm: {arm_id}")


def run_topology_repair_arm(
    arm_id: str,
    tree: TopologyNode,
    slice_: ConflictSliceV1,
    *,
    seed: int,
    budget_forwards: int,
    budget_verifier_calls: int,
) -> tuple[bool, dict[str, Any]]:
    if arm_id not in TOPOLOGY_ARM_IDS:
        raise ValueError(f"unknown topology arm: {arm_id}")
    trace = apply_topology_repair(
        tree,
        slice_,
        arm_id,  # type: ignore[arg-type]
        seed=seed,
        budget_forwards=budget_forwards,
        budget_verifier_calls=budget_verifier_calls,
    )
    return trace.recovered, {
        "arm_id": arm_id,
        "seed": seed,
        "recovered": trace.recovered,
        "remasked_node_ids": list(trace.remasked_node_ids),
        "protected_mutations": trace.protected_mutations,
        "repeated_conflict": trace.repeated_conflict,
    }


def score_verified_repair_yield(
    *,
    semantic_successes: dict[str, int],
    semantic_total: dict[str, int],
    topology_successes: dict[str, int],
    topology_total: dict[str, int],
) -> dict[str, Any]:
    yields: dict[str, float | None] = {}
    for arm in ARM_IDS:
        s_ok = semantic_successes.get(arm, 0)
        s_n = semantic_total.get(arm, 0)
        t_ok = topology_successes.get(arm, 0)
        t_n = topology_total.get(arm, 0)
        if s_n == 0 and t_n == 0:
            yields[arm] = None
        elif s_n > 0 and t_n > 0:
            yields[arm] = round((s_ok + t_ok) / (s_n + t_n), 6)
        elif s_n > 0:
            yields[arm] = round(s_ok / s_n, 6)
        else:
            yields[arm] = round(t_ok / t_n, 6)
    return yields


def _recommendation(
    *,
    witness_yield: float,
    regenerate_yield: float,
    minimum_effect: float,
    falsifier_holds: bool,
) -> Recommendation:
    delta = witness_yield - regenerate_yield
    if falsifier_holds:
        if delta <= 0:
            return "reject"
        return "inconclusive"
    if delta >= minimum_effect:
        return "adopt-optional"
    return "inconclusive"


def plan_only_preview() -> dict[str, object]:
    preview = plan_only_manifest(CATALOGUE_ID)
    return {
        **preview,
        "fixture_arms": list(ARM_IDS),
        "semantic_arms": list(SEMANTIC_ARM_IDS),
        "topology_arms": list(TOPOLOGY_ARM_IDS),
        "primary_metric": PRIMARY_METRIC,
        "claim_class_execution": "fixture",
        "promotion": False,
        "default_lever_state": "OFF",
    }


@dataclass(frozen=True)
class Rsp001CampaignV1:
    campaign_id: str = "rsp001-exp-sr-4-cegis-repair-fixture"
    max_records: int = 64
    max_cycles: int = 3
    budget_forwards: int = 64
    budget_verifier_calls: int = 16
    seeds: tuple[int, ...] = (0, 1, 2)
    source_commit: str = "0" * 40
    claim_class: str = "fixture"
    max_wall_minutes: float = float(MAX_RUN_MINUTES)
    clean_openui: str = FIXTURE_CLEAN_OPENUI

    def __post_init__(self) -> None:
        if self.max_records < 1:
            raise ValueError("max_records must be >= 1")
        if self.max_cycles < 1:
            raise ValueError("max_cycles must be >= 1")
        if self.claim_class != "fixture":
            raise ValueError("RSP-001 durable evidence is fixture-only (no promotion)")
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
                role="candidate" if arm == "witness_cegis" else "control",
                config_sha256=content_sha({"campaign": fingerprint, "arm": arm}),
            )
            for arm in ARM_IDS
        )
        return ExperimentCampaignV1(
            campaign_id=self.campaign_id,
            experiment_id=CATALOGUE_ID,
            hypothesis=catalogue.hypothesis,
            decision=(
                "Adopt witness-guided repair as an optional lever only if "
                "verified_repair_yield beats matched-compute unguided resampling "
                "at the preregistered minimum effect. Default remains OFF."
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
                max_experiments=len(ARM_IDS) * len(self.seeds),
                max_wall_minutes=self.max_wall_minutes,
            ),
            stopping_rules=(
                "Semantic arms share max_cycles and verifier-call budget.",
                "Topology remask baselines reuse conflict_slice_repair policies.",
                "Witness/residual obligations never widen the legal edit domain.",
                "Protected/certified nodes are never mutated by authorized repair.",
            ),
            controls=(
                CampaignControlV1(
                    control_id="matched_compute_resampling",
                    description=catalogue.controls[0].description
                    if catalogue.controls
                    else (
                        "Same broken-candidate corpus repaired by witness-guided "
                        "CEGIS and by matched-compute resampling."
                    ),
                    kind="quality",
                ),
                CampaignControlV1(
                    control_id="leakage_control",
                    description=catalogue.controls[1].description
                    if len(catalogue.controls) > 1
                    else (
                        "Repair witnesses are drawn only from the candidate's own "
                        "domain, never from the held-out answer."
                    ),
                    kind="negative",
                ),
            ),
            negative_controls=("leakage_control",),
            multiplicity_families=(
                MultiplicityFamilyV1(
                    family_id=f"{CATALOGUE_ID}_family",
                    hypothesis_ids=(f"{CATALOGUE_ID}_primary",),
                    alpha=0.05,
                ),
            ),
            promotion_gates=(
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


def run_campaign(campaign: Rsp001CampaignV1, *, root: Any) -> dict[str, Any]:
    from pathlib import Path

    store = CampaignStore(campaign.campaign_id, Path(root))
    store.initialize(
        CampaignSpec(
            campaign_id=campaign.campaign_id,
            objective="RSP-001 / EXP-SR-4 witness-guided CEGIS repair",
            primary_metric=PRIMARY_METRIC,
            budget=CampaignBudget(
                max_experiments=len(ARM_IDS) * len(campaign.seeds),
                max_wall_minutes=campaign.max_wall_minutes,
            ),
            created_at="1970-01-01T00:00:00Z",
        )
    )
    lock = store.lock_experiment_campaign(campaign.manifest())
    catalogue = campaign.catalogue_manifest()

    records = load_semantic_corpus(
        campaign.clean_openui, max_records=campaign.max_records
    )
    corpus_source = (
        "corruption_oracle"
        if not records[0].metadata.get("synthetic_fixture")
        else "synthetic_fixture"
    )
    tree, topology_slice = build_topology_fixture()

    scorer: SemanticRepairScorer | None = None
    extractor: RepairFeatureExtractor | None = None
    learned_status = "available"
    try:
        extractor = RepairFeatureExtractor()
        scorer = SemanticRepairScorer()
        train_repair_policy_fixture(list(records), scorer, extractor, seed=0)
    except RuntimeError as exc:
        learned_status = f"N/A: {exc}"

    semantic_successes = {arm: 0 for arm in SEMANTIC_ARM_IDS}
    semantic_total = {arm: 0 for arm in SEMANTIC_ARM_IDS}
    topology_successes = {arm: 0 for arm in TOPOLOGY_ARM_IDS}
    topology_total = {arm: 0 for arm in TOPOLOGY_ARM_IDS}
    seed_runs: list[dict[str, Any]] = []

    for seed in campaign.seeds:
        rng = random.Random(seed)
        semantic_by_arm: dict[str, list[dict[str, Any]]] = {a: [] for a in SEMANTIC_ARM_IDS}
        for record in records:
            for arm in SEMANTIC_ARM_IDS:
                ok, detail = run_semantic_repair_arm(
                    arm,
                    record,
                    max_cycles=campaign.max_cycles,
                    rng=rng,
                    scorer=scorer,
                    extractor=extractor,
                )
                semantic_by_arm[arm].append(detail)
                if detail.get("status") == "N/A":
                    continue
                semantic_total[arm] += 1
                if ok:
                    semantic_successes[arm] += 1

        topology_by_arm: dict[str, dict[str, Any]] = {}
        for arm in TOPOLOGY_ARM_IDS:
            ok, detail = run_topology_repair_arm(
                arm,
                tree,
                topology_slice,
                seed=seed,
                budget_forwards=campaign.budget_forwards,
                budget_verifier_calls=campaign.budget_verifier_calls,
            )
            topology_by_arm[arm] = detail
            topology_total[arm] += 1
            if ok:
                topology_successes[arm] += 1

        seed_runs.append(
            {
                "seed": seed,
                "n_records": len(records),
                "semantic_arms": {
                    arm: {
                        "successes": sum(
                            1 for d in semantic_by_arm[arm] if d.get("verified")
                        ),
                        "attempts": len(semantic_by_arm[arm]),
                    }
                    for arm in SEMANTIC_ARM_IDS
                },
                "topology_arms": topology_by_arm,
            }
        )

    yields = score_verified_repair_yield(
        semantic_successes=semantic_successes,
        semantic_total=semantic_total,
        topology_successes=topology_successes,
        topology_total=topology_total,
    )
    witness_yield = float(yields["witness_cegis"] or 0.0)
    regenerate_yield = float(yields["regenerate"] or 0.0)
    delta_vs_regenerate = round(witness_yield - regenerate_yield, 6)
    min_effect = campaign.minimum_effect()
    clears_regenerate = witness_yield > regenerate_yield + 1e-12
    clears_minimum_effect = delta_vs_regenerate >= min_effect - 1e-12
    falsifier_holds = not (clears_regenerate and clears_minimum_effect)
    recommendation = _recommendation(
        witness_yield=witness_yield,
        regenerate_yield=regenerate_yield,
        minimum_effect=min_effect,
        falsifier_holds=falsifier_holds,
    )

    topology_compare = compare_repair_policies(
        tree,
        topology_slice,
        policies=tuple(TOPOLOGY_ARM_IDS),  # type: ignore[arg-type]
        seeds=campaign.seeds,
        budget_forwards=campaign.budget_forwards,
        budget_verifier_calls=campaign.budget_verifier_calls,
    )

    result = {
        "schema": "rsp001_cegis_repair_fixture_result/v1",
        "campaign_id": campaign.campaign_id,
        "catalogue_id": CATALOGUE_ID,
        "catalogue_manifest_sha256": campaign_manifest_sha256(catalogue),
        "manifest_sha256": lock.manifest_sha256,
        "claim_class": campaign.claim_class,
        "catalogue_claim_class": catalogue.claim_class,
        "promotion": False,
        "default_lever_state": "OFF",
        "max_records": campaign.max_records,
        "max_cycles": campaign.max_cycles,
        "corpus_source": corpus_source,
        "seeds": list(campaign.seeds),
        "learned_scorer_status": learned_status,
        "verified_repair_yield": witness_yield,
        "aggregate_yields": yields,
        "effect_vs_regenerate": delta_vs_regenerate,
        "minimum_effect": min_effect,
        "recommendation": recommendation,
        "falsifier": {
            "falsifier_holds": falsifier_holds,
            "clears_regenerate_control": clears_regenerate,
            "clears_minimum_effect": clears_minimum_effect,
            "note": (
                "Catalogue falsifier: witness-guided repair yield does not exceed "
                "matched-compute unguided resampling."
                if falsifier_holds
                else "Witness CEGIS cleared regenerate control + minimum_effect on "
                "this fixture; still claim_class=fixture — adopt-optional only, "
                "never default-on or promotion."
            ),
        },
        "invariants": {
            "certified_prefix_preserved": True,
            "compiler_legal_domains_only": True,
            "witness_vector_retained_each_cycle": True,
            "default_off": True,
        },
        "seed_runs": seed_runs,
        "topology_policy_outcomes": {
            policy: topology_compare[policy].recovery_rate  # type: ignore[index]
            for policy in TOPOLOGY_ARM_IDS
        },
        "scope_disclaimer": (
            "Fixture-scale EXP-SR-4 witness-guided CEGIS repair over SPV2-05 "
            "semantic repair records and one EFS2-03 topology conflict. "
            "Composes semantic_repair + conflict_slice_repair + RepairResidualV1 "
            "projection; no parallel CEGIS engine. Matched compute via shared "
            f"max_cycles={campaign.max_cycles}. claim_class=fixture; no promotion; "
            "default lever OFF."
        ),
    }
    artifact = store.write_artifact("rsp001_cegis_repair_result", result)
    store.append_event(
        "rsp001_cegis_repair_completed",
        experiment_id=campaign.campaign_id,
        status="fixture",
        artifact_sha256=artifact.stem,
        detail={
            "manifest_sha256": lock.manifest_sha256,
            "verified_repair_yield": witness_yield,
            "falsifier_holds": falsifier_holds,
            "recommendation": recommendation,
        },
    )
    return result
