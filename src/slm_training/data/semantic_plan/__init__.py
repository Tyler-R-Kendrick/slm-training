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
    filter_manifest_safe,
    intervention_record_integrity_ok,
)
from slm_training.data.semantic_plan.seed import PlanSeedBuilder

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
    "canonicalize_plan",
    "filter_manifest_safe",
    "intervention_record_integrity_ok",
    "plan_factor_fingerprints",
    "project_authority",
]
