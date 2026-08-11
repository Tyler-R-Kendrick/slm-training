"""SemanticPlanV1 extraction, canonicalization, oracle substitution, seed construction, and compilation."""

from __future__ import annotations

from slm_training.data.semantic_plan.canonicalize import canonicalize_plan, plan_factor_fingerprints
from slm_training.data.semantic_plan.compiler import (
    AuthorityProjectionV1,
    Evidence,
    EvidenceKind,
    EvidenceProjection,
    HardRemoval,
    OpenUISemanticPlanCompiler,
    PlanActionFeatures,
    PlanAssumption,
    PlanAssumptionTrail,
    PlanSeedResult,
    RestrictionResult,
    SemanticPlanCompiler,
    project_authority,
)
from slm_training.data.semantic_plan.extract import (
    OpenUISemanticPlanExtractor,
    SemanticPlanExtractor,
)
from slm_training.data.semantic_plan.oracle import (
    InterventionIdentityV1,
    PlanInterventionRecordV1,
    PlanOracleSubstitutor,
    apply_plan_intervention,
    build_baseline_intervention,
    filter_manifest_safe,
    intervention_record_integrity_ok,
    select_shuffled_oracle,
)
from slm_training.data.semantic_plan.seed import PlanSeedBuilder
from slm_training.data.semantic_plan.requirements_compile import (
    compile_goal_constraints,
    evaluate_goal_constraints,
)

__all__ = [
    "AuthorityProjectionV1",
    "Evidence",
    "EvidenceKind",
    "EvidenceProjection",
    "HardRemoval",
    "InterventionIdentityV1",
    "OpenUISemanticPlanExtractor",
    "OpenUISemanticPlanCompiler",
    "PlanActionFeatures",
    "PlanAssumption",
    "PlanAssumptionTrail",
    "PlanInterventionRecordV1",
    "PlanOracleSubstitutor",
    "PlanSeedBuilder",
    "PlanSeedResult",
    "RestrictionResult",
    "SemanticPlanCompiler",
    "SemanticPlanExtractor",
    "apply_plan_intervention",
    "build_baseline_intervention",
    "canonicalize_plan",
    "compile_goal_constraints",
    "evaluate_goal_constraints",
    "filter_manifest_safe",
    "intervention_record_integrity_ok",
    "plan_factor_fingerprints",
    "project_authority",
    "select_shuffled_oracle",
]
