"""SIE-008 (SLM-484): symbolic value-of-compute controller campaign (EXP-SR-5).

Fixture-scale governed campaign that runs the SIE-007 replay-only controller
substrate (:mod:`compute_state_controller`) through the SIE-001 catalogue
identity ``exp-sr-5`` under :class:`ExperimentCampaignV1`.

Arms (matched seeds, discovery/eval split on record identity — eval never
appears in linear fit):

* ``hand_threshold`` — fixed priority rule over observable counters (control)
* ``linear_scorer`` — NumPy least-squares one-vs-rest fit on discovery only
* ``symbolic_rule`` — preregistered :class:`ControllerRuleV1` IR expression
* ``gold_oracle`` — diagnostic ceiling from observed counterfactuals only

Safety (hard): exact singleton bypass (I2), controller cannot change legality
(I6), UNKNOWN features abstain, no production default change.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

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
from slm_training.harness_core.lineage.records import content_sha
from slm_training.harnesses.experiments.compute_state_controller import (
    ComputeStateFeaturesV1,
    ControllerRecommendationV1,
    ControllerRuleV1,
    controller_can_change_legality,
    features_from_decode_record,
    oracle_action_for_record,
    replay_rule,
)
from slm_training.harnesses.experiments.compute_state_controller import CONTROLLER_ACTIONS
from slm_training.harnesses.experiments.exp_sr_catalogue import (
    exp_sr_campaign,
    plan_only_manifest,
)
from slm_training.levers import MAX_RUN_MINUTES
from slm_training.models.decode_stats import DecodeIdentityV1, DecodeStats, build_decode_stats_record
from slm_training.models.decode_stats import DecodeStatsRecordV1

CATALOGUE_ID = "exp-sr-5"
PRIMARY_METRIC = "compute_value_regret"
DATASET_ID = "sie008_fixture_decode_traces_v1"

ARM_IDS: tuple[str, ...] = (
    "hand_threshold",
    "linear_scorer",
    "symbolic_rule",
    "gold_oracle",
)

_SCORABLE = tuple(a for a in CONTROLLER_ACTIONS if a != "abstain")

# Preregistered symbolic rule (safe IR only — no literals beyond v<i>/c<i>).
SYMBOLIC_RULE = ControllerRuleV1(
    rule_id="sie008-symbolic-v1",
    action_expressions={
        "repair": "v11",
        "bounded_search": "v9",
        "neural_forward": "v3",
        "deterministic_ranker": "(+ v5 v6)",
    },
    abstain_margin=0.25,
)

__all__ = [
    "ARM_IDS",
    "CATALOGUE_ID",
    "DATASET_ID",
    "PRIMARY_METRIC",
    "SYMBOLIC_RULE",
    "Sie008CampaignV1",
    "compute_recommendation",
    "compute_regret",
    "fixture_decode_records",
    "hand_threshold_recommend",
    "legal_support_parity",
    "linear_scorer_recommend",
    "plan_only_preview",
    "replay_arm",
    "run_campaign",
    "singleton_bypass_audit",
    "split_discovery_eval",
]


def fixture_decode_records(*, seed: int, n: int = 120) -> tuple[DecodeStatsRecordV1, ...]:
    """Deterministic synthetic steady-state decode traces for fixture replay."""
    if n < 24:
        raise ValueError("fixture needs >=24 records for discovery/eval split")
    rng = random.Random(seed)
    records: list[DecodeStatsRecordV1] = []
    prev_digest: str | None = None
    for idx in range(n):
        domain_size = 1 if rng.random() < 0.12 else rng.randint(2, 8)
        status = "complete" if domain_size == 1 or rng.random() < 0.92 else "incomplete"
        forwards = 0 if domain_size == 1 else rng.randint(0, 6)
        attempts = 1 if forwards == 0 and rng.random() < 0.35 else rng.randint(1, 4)
        retries = 1 if rng.random() < 0.08 and domain_size > 1 else 0
        ambiguous = rng.randint(0, max(1, forwards))
        stats = DecodeStats(
            forwards_count=forwards,
            ambiguous_rows_forwarded=ambiguous,
            denoiser_rows_evaluated=max(forwards, ambiguous),
            forced_row_tokens_without_forward=rng.randint(0, 3),
            semantic_singleton_bypasses=1 if domain_size == 1 else 0,
            compiler_candidates=rng.randint(0, 4),
            probes_count=rng.randint(0, 2),
            attempts=attempts,
            tokens_emitted=rng.randint(1, 24),
            unconstrained_retries=retries,
            total_ms=float(rng.randint(5, 400)),
        )
        identity = DecodeIdentityV1(
            artifact_sha256=hashlib.sha256(f"sie008-fixture-{seed}-{idx}".encode()).hexdigest(),
        )
        record = build_decode_stats_record(
            stats,
            measurement_stage="steady_state",
            completeness="complete",
            identity=identity,
            legal_domain_size=domain_size,
            legal_domain_status=status,
            witness_ids=(f"w-{seed}-{idx}",),
            prev_record_digest=prev_digest,
        )
        prev_digest = record.record_digest
        records.append(record)
    return tuple(records)


def split_discovery_eval(
    records: tuple[DecodeStatsRecordV1, ...],
    *,
    seed: int,
    discovery_fraction: float = 0.7,
) -> tuple[tuple[DecodeStatsRecordV1, ...], tuple[DecodeStatsRecordV1, ...]]:
    """Disjoint discovery/eval partition keyed by record digest (leakage-safe)."""
    if not 0.1 < discovery_fraction < 0.9:
        raise ValueError("discovery_fraction must be in (0.1, 0.9)")
    ordered = sorted(records, key=lambda r: r.record_digest)
    rng = random.Random(seed)
    indices = list(range(len(ordered)))
    rng.shuffle(indices)
    split_at = max(1, int(len(indices) * discovery_fraction))
    if split_at >= len(indices):
        split_at = len(indices) - 1
    discovery_idx = set(indices[:split_at])
    discovery = tuple(ordered[i] for i in sorted(discovery_idx))
    eval_records = tuple(ordered[i] for i in sorted(set(indices) - discovery_idx))
    if not discovery or not eval_records:
        raise ValueError("split produced empty discovery or eval set")
    if {r.record_digest for r in discovery} & {r.record_digest for r in eval_records}:
        raise ValueError("discovery/eval overlap")
    return discovery, eval_records


def hand_threshold_recommend(features: ComputeStateFeaturesV1) -> ControllerRecommendationV1:
    """Fixed priority baseline — never evaluates symbolic expressions."""
    if features.is_exact_singleton:
        return ControllerRecommendationV1(
            action="deterministic_ranker",
            score=None,
            reason="exact singleton: I2 bypass",
            expressions_evaluated=0,
        )
    vals = features.values
    if (vals.get("unconstrained_retries") or 0) > 0:
        return ControllerRecommendationV1(
            action="repair", score=1.0, reason="hand: unconstrained_retries>0", expressions_evaluated=0
        )
    if (vals.get("attempts") or 1) > 2:
        return ControllerRecommendationV1(
            action="bounded_search", score=1.0, reason="hand: attempts>2", expressions_evaluated=0
        )
    if (vals.get("forwards_count") or 0) > 0:
        return ControllerRecommendationV1(
            action="neural_forward", score=1.0, reason="hand: forwards_count>0", expressions_evaluated=0
        )
    return ControllerRecommendationV1(
        action="deterministic_ranker", score=1.0, reason="hand: default ranker", expressions_evaluated=0
    )


def _feature_matrix(records: tuple[DecodeStatsRecordV1, ...]) -> np.ndarray:
    rows: list[list[float]] = []
    for record in records:
        features = features_from_decode_record(record)
        if features.unknown_features():
            raise ValueError(
                f"fixture record {record.record_digest[:12]} has UNKNOWN features; "
                "linear fit requires complete measurements"
            )
        rows.append([float(v) for v in features.feature_vector()])
    return np.asarray(rows, dtype=np.float64)


def _fit_linear_weights(
    discovery: tuple[DecodeStatsRecordV1, ...],
) -> dict[str, tuple[np.ndarray, float]]:
    """One-vs-rest least squares on discovery traces only."""
    x = _feature_matrix(discovery)
    x_aug = np.hstack([x, np.ones((len(x), 1), dtype=np.float64)])
    weights: dict[str, tuple[np.ndarray, float]] = {}
    for action in _SCORABLE:
        y = np.asarray(
            [1.0 if oracle_action_for_record(r) == action else 0.0 for r in discovery],
            dtype=np.float64,
        )
        coef, _, _, _ = np.linalg.lstsq(x_aug, y, rcond=None)
        weights[action] = (coef[:-1], float(coef[-1]))
    return weights


def linear_scorer_recommend(
    features: ComputeStateFeaturesV1,
    *,
    weights: dict[str, tuple[np.ndarray, float]],
) -> ControllerRecommendationV1:
    if features.is_exact_singleton:
        return ControllerRecommendationV1(
            action="deterministic_ranker",
            score=None,
            reason="exact singleton: I2 bypass",
            expressions_evaluated=0,
        )
    unknown = features.unknown_features()
    if unknown:
        return ControllerRecommendationV1(
            action="abstain",
            score=None,
            reason=f"UNKNOWN feature(s): {list(unknown)}",
            expressions_evaluated=0,
        )
    vector = np.asarray([float(v) for v in features.feature_vector()], dtype=np.float64)
    scores = {
        action: float(np.dot(w, vector) + b) for action, (w, b) in weights.items()
    }
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    best_action, best_score = ranked[0]
    return ControllerRecommendationV1(
        action=best_action,
        score=best_score,
        reason="linear one-vs-rest argmax",
        expressions_evaluated=0,
    )


def compute_regret(
    records: tuple[DecodeStatsRecordV1, ...],
    recommend: Callable[[DecodeStatsRecordV1], ControllerRecommendationV1],
) -> dict[str, Any]:
    recommendations: list[ControllerRecommendationV1] = []
    oracles: list[str] = []
    agreements = 0
    abstentions = 0
    evaluated = 0
    for record in records:
        rec = recommend(record)
        oracle = oracle_action_for_record(record)
        recommendations.append(rec)
        oracles.append(oracle)
        evaluated += rec.expressions_evaluated
        if rec.action == "abstain":
            abstentions += 1
        elif rec.action == oracle:
            agreements += 1
    decided = len(records) - abstentions
    regret = 1.0 if decided == 0 else 1.0 - (agreements / decided)
    return {
        "n": len(records),
        "agreements": agreements,
        "abstentions": abstentions,
        "decided": decided,
        "compute_value_regret": round(regret, 6),
        "expressions_evaluated": evaluated,
        "coverage": round(decided / len(records), 6) if records else 0.0,
    }


def replay_arm(
    arm_id: str,
    records: tuple[DecodeStatsRecordV1, ...],
    *,
    discovery: tuple[DecodeStatsRecordV1, ...] = (),
) -> dict[str, Any]:
    if arm_id == "hand_threshold":
        fn: Callable[[DecodeStatsRecordV1], ControllerRecommendationV1] = (
            lambda r: hand_threshold_recommend(features_from_decode_record(r))
        )
    elif arm_id == "linear_scorer":
        weights = _fit_linear_weights(discovery)
        fn = lambda r: linear_scorer_recommend(  # noqa: E731
            features_from_decode_record(r), weights=weights
        )
    elif arm_id == "symbolic_rule":
        fn = lambda r: SYMBOLIC_RULE.recommend(features_from_decode_record(r))  # noqa: E731
    elif arm_id == "gold_oracle":
        fn = lambda r: (  # noqa: E731
            ControllerRecommendationV1(
                action=oracle_action_for_record(r),
                score=1.0,
                reason="gold oracle ceiling",
                expressions_evaluated=0,
            )
            if not features_from_decode_record(r).is_exact_singleton
            else ControllerRecommendationV1(
                action="deterministic_ranker",
                score=None,
                reason="exact singleton: I2 bypass",
                expressions_evaluated=0,
            )
        )
    else:
        raise ValueError(f"unknown arm: {arm_id}")
    metrics = compute_regret(records, fn)
    return {"arm_id": arm_id, **metrics}


def legal_support_parity() -> dict[str, Any]:
    parity = controller_can_change_legality() is False
    return {
        "legal_support_parity_exact": parity,
        "controller_can_change_legality": controller_can_change_legality(),
        "note": (
            "Controller recommends among authorized compute strategies only; "
            "legal domain is grammar-owned and unchanged (I6)."
        ),
    }


def singleton_bypass_audit(records: tuple[DecodeStatsRecordV1, ...]) -> dict[str, Any]:
    singletons = [
        r for r in records if features_from_decode_record(r).is_exact_singleton
    ]
    violations: list[str] = []
    for record in singletons:
        for arm in ("hand_threshold", "symbolic_rule"):
            if arm == "hand_threshold":
                rec = hand_threshold_recommend(features_from_decode_record(record))
            else:
                rec = SYMBOLIC_RULE.recommend(features_from_decode_record(record))
            if rec.expressions_evaluated != 0:
                violations.append(f"{arm}:{record.record_digest[:12]}")
    return {
        "n_singletons": len(singletons),
        "singleton_bypass_holds": not violations,
        "violations": violations,
        "note": "I2: complete singleton domains commit with zero expression evaluations.",
    }


def compute_recommendation(
    *,
    symbolic_regret: float,
    control_regret: float,
    minimum_effect: float,
    falsifier_holds: bool,
    safety_ok: bool,
) -> dict[str, Any]:
    delta = round(control_regret - symbolic_regret, 6)
    clears_minimum = delta >= minimum_effect - 1e-12
    if falsifier_holds or not safety_ok:
        disposition = "reject"
        rationale = (
            "Catalogue falsifier holds or safety audit failed; controller stays off."
            if falsifier_holds
            else "Safety audit failed (singleton bypass or legality parity)."
        )
    elif clears_minimum:
        disposition = "adopt_optional"
        rationale = (
            f"Symbolic rule beat hand-threshold control by {delta} regret "
            f"(minimum_effect={minimum_effect}); advisory replay only — no default."
        )
    elif abs(delta) < minimum_effect:
        disposition = "inconclusive"
        rationale = (
            f"Symbolic vs control regret delta {delta} below minimum_effect "
            f"{minimum_effect}; fixture inconclusive."
        )
    else:
        disposition = "reject"
        rationale = "Symbolic rule did not beat hand-threshold control on eval regret."
    return {
        "disposition": disposition,
        "rationale": rationale,
        "delta_vs_hand_threshold": delta,
        "clears_minimum_effect": clears_minimum,
    }


def plan_only_preview() -> dict[str, object]:
    preview = plan_only_manifest(CATALOGUE_ID)
    return {
        **preview,
        "fixture_arms": list(ARM_IDS),
        "primary_metric": PRIMARY_METRIC,
        "symbolic_rule_id": SYMBOLIC_RULE.rule_id,
        "claim_class_execution": "fixture",
        "promotion": False,
    }


@dataclass(frozen=True)
class Sie008CampaignV1:
    campaign_id: str = "sie008-exp-sr-5-voc-controller-fixture"
    n_records: int = 120
    discovery_fraction: float = 0.7
    seeds: tuple[int, ...] = (0, 1, 2)
    source_commit: str = "0" * 40
    claim_class: str = "fixture"
    max_wall_minutes: float = float(MAX_RUN_MINUTES)

    def __post_init__(self) -> None:
        if self.n_records < 24:
            raise ValueError("n_records must be >= 24")
        if not 0.1 < self.discovery_fraction < 0.9:
            raise ValueError("discovery_fraction must be in (0.1, 0.9)")
        if self.claim_class != "fixture":
            raise ValueError("SIE-008 durable evidence is fixture-only (no promotion)")
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
                role="candidate" if arm == "symbolic_rule" else "control",
                config_sha256=content_sha({"campaign": fingerprint, "arm": arm}),
            )
            for arm in ARM_IDS
        )
        return ExperimentCampaignV1(
            campaign_id=self.campaign_id,
            experiment_id=CATALOGUE_ID,
            hypothesis=catalogue.hypothesis,
            decision=(
                "Adopt the symbolic controller only when eval regret beats the "
                "hand-threshold control by the catalogue minimum_effect at matched "
                "coverage; gold oracle is diagnostic ceiling only. No production default."
            ),
            endpoints=(
                CampaignEndpointV1(
                    endpoint_id=f"{CATALOGUE_ID}_primary",
                    metric=PRIMARY_METRIC,
                    role="primary",
                    direction="decrease",
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
                "Discovery traces fit linear_scorer only; eval traces score every arm.",
                "Exact singleton domains bypass all controllers with zero expressions.",
                "Gold oracle is diagnostic only and never a promotion path.",
            ),
            controls=(
                CampaignControlV1(
                    control_id="hand_threshold",
                    description=(
                        "Fixed priority rule over observable decode counters "
                        "(matched fixed-budget control axis for this fixture)."
                    ),
                    kind="quality",
                ),
                CampaignControlV1(
                    control_id="leakage_control",
                    description=(
                        "Discovery/eval split is disjoint on record_digest; "
                        "linear_scorer never sees eval traces during fit."
                    ),
                    kind="negative",
                ),
                CampaignControlV1(
                    control_id="legal_support_parity",
                    description=(
                        "Controller recommends among authorized strategies only; "
                        "cannot widen or narrow legal support."
                    ),
                    kind="negative",
                ),
            ),
            negative_controls=("leakage_control", "legal_support_parity"),
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
                    operator="le",
                    threshold=1.0,
                ),
            ),
            rollback_gates=(
                CampaignGateV1(
                    gate_id=f"{CATALOGUE_ID}_kill",
                    endpoint_id=f"{CATALOGUE_ID}_primary",
                    operator="gt",
                    threshold=0.0,
                ),
            ),
            artifact_requirements=(
                ArtifactRequirementV1(kind="version_stamp"),
                ArtifactRequirementV1(kind="endpoint_result"),
                ArtifactRequirementV1(kind="paired_examples"),
            ),
            claim_class=self.claim_class,
            source_commit=self.source_commit,
            source_dirty=False,
            author="slm-training",
            created_at="1970-01-01T00:00:00Z",
        )


def run_campaign(campaign: Sie008CampaignV1, *, root: Path) -> dict[str, Any]:
    store = CampaignStore(campaign.campaign_id, root)
    store.initialize(
        CampaignSpec(
            campaign_id=campaign.campaign_id,
            objective="SIE-008 / EXP-SR-5 fixture value-of-compute controller",
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
    parity = legal_support_parity()
    min_effect = campaign.minimum_effect()

    seed_runs: list[dict[str, Any]] = []
    arm_regret: dict[str, list[float]] = {arm: [] for arm in ARM_IDS}

    for seed in campaign.seeds:
        records = fixture_decode_records(seed=seed, n=campaign.n_records)
        discovery, eval_records = split_discovery_eval(
            records, seed=seed, discovery_fraction=campaign.discovery_fraction
        )
        bypass = singleton_bypass_audit(records)
        arm_metrics: dict[str, Any] = {}
        for arm in ARM_IDS:
            metrics = replay_arm(arm, eval_records, discovery=discovery)
            arm_metrics[arm] = metrics
            arm_regret[arm].append(metrics["compute_value_regret"])

        # Symbolic rule full replay report for audit trail.
        symbolic_report = replay_rule(SYMBOLIC_RULE, eval_records)

        seed_runs.append(
            {
                "seed": seed,
                "n_records": len(records),
                "n_discovery": len(discovery),
                "n_eval": len(eval_records),
                "discovery_eval_disjoint": True,
                "singleton_bypass": bypass,
                "arms": arm_metrics,
                "symbolic_replay": {
                    "regret": symbolic_report.regret,
                    "agreements": symbolic_report.agreements,
                    "abstentions": symbolic_report.abstentions,
                },
            }
        )

    def _mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 6) if xs else 0.0

    aggregate_arms = {
        arm: {
            "compute_value_regret_mean": _mean(arm_regret[arm]),
            "compute_value_regret_per_seed": arm_regret[arm],
            "n_seeds": len(arm_regret[arm]),
        }
        for arm in ARM_IDS
    }
    symbolic_mean = aggregate_arms["symbolic_rule"]["compute_value_regret_mean"]
    control_mean = aggregate_arms["hand_threshold"]["compute_value_regret_mean"]
    oracle_mean = aggregate_arms["gold_oracle"]["compute_value_regret_mean"]
    delta_vs_control = round(control_mean - symbolic_mean, 6)
    ceiling_recovery = round(1.0 - (symbolic_mean / oracle_mean), 6) if oracle_mean > 0 else 0.0

    all_bypass_ok = all(run["singleton_bypass"]["singleton_bypass_holds"] for run in seed_runs)
    safety_ok = parity["legal_support_parity_exact"] and all_bypass_ok
    clears_control = symbolic_mean < control_mean - 1e-12
    clears_minimum = delta_vs_control >= min_effect - 1e-12
    falsifier_holds = not (clears_control and clears_minimum)

    paired_deltas = [
        arm_regret["hand_threshold"][i] - arm_regret["symbolic_rule"][i]
        for i in range(len(campaign.seeds))
    ]
    paired_mean = _mean(paired_deltas)
    paired_var = (
        sum((d - paired_mean) ** 2 for d in paired_deltas) / max(1, len(paired_deltas) - 1)
        if len(paired_deltas) > 1
        else 0.0
    )
    paired_se = round((paired_var**0.5) / (len(paired_deltas) ** 0.5), 6) if paired_deltas else 0.0

    recommendation = compute_recommendation(
        symbolic_regret=symbolic_mean,
        control_regret=control_mean,
        minimum_effect=min_effect,
        falsifier_holds=falsifier_holds,
        safety_ok=safety_ok,
    )

    result = {
        "schema": "sie008_voc_controller_fixture_result/v1",
        "campaign_id": campaign.campaign_id,
        "catalogue_id": CATALOGUE_ID,
        "catalogue_manifest_sha256": campaign_manifest_sha256(catalogue),
        "manifest_sha256": lock.manifest_sha256,
        "claim_class": campaign.claim_class,
        "catalogue_claim_class": catalogue.claim_class,
        "promotion": False,
        "dataset_id": DATASET_ID,
        "n_records": campaign.n_records,
        "discovery_fraction": campaign.discovery_fraction,
        "seeds": list(campaign.seeds),
        "symbolic_rule": SYMBOLIC_RULE.to_dict(),
        "legal_support_parity": parity,
        "compute_value_regret": symbolic_mean,
        "aggregate_arms": aggregate_arms,
        "effect_vs_hand_threshold": delta_vs_control,
        "ceiling_recovery_vs_gold": ceiling_recovery,
        "minimum_effect": min_effect,
        "paired_vs_hand_threshold": {
            "deltas": [round(d, 6) for d in paired_deltas],
            "mean": paired_mean,
            "se": paired_se,
            "n_seeds": len(paired_deltas),
        },
        "falsifier": {
            "falsifier_holds": falsifier_holds,
            "clears_hand_threshold_control": clears_control,
            "clears_minimum_effect": clears_minimum,
            "note": (
                "Catalogue falsifier: symbolic regret not below hand-threshold "
                "control by minimum_effect on eval traces."
                if falsifier_holds
                else "Symbolic rule cleared hand-threshold + minimum_effect on "
                "this fixture; still claim_class=fixture — no production default."
            ),
        },
        "recommendation": recommendation,
        "seed_runs": seed_runs,
        "scope_disclaimer": (
            "Fixture-scale EXP-SR-5 value-of-compute controller replay over "
            "deterministic synthetic DecodeStatsRecordV1 traces. Arms compose "
            "SIE-007 compute_state_controller (hand threshold, linear scorer fit "
            "on discovery only, preregistered symbolic IR, gold oracle ceiling). "
            "Primary metric is compute_value_regret on held-out eval traces "
            "(decrease is better). Discovery/eval are disjoint. I2 singleton "
            "bypass and I6 legality parity are audited. claim_class=fixture; "
            "no promotion or production default change."
        ),
    }
    artifact = store.write_artifact("sie008_voc_controller_result", result)
    store.append_event(
        "sie008_voc_controller_completed",
        experiment_id=campaign.campaign_id,
        status="fixture",
        artifact_sha256=artifact.stem,
        detail={
            "manifest_sha256": lock.manifest_sha256,
            "compute_value_regret": symbolic_mean,
            "falsifier_holds": falsifier_holds,
            "disposition": recommendation["disposition"],
            "legal_support_parity_exact": parity["legal_support_parity_exact"],
        },
    )
    return result
