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
from slm_training.data.semantic_plan.oracle import PlanOracleSubstitutor
from slm_training.data.semantic_plan.seed import PlanSeedBuilder

__all__ = [
    "AuthorityProjectionV1",
    "Evidence",
    "EvidenceKind",
    "EvidenceProjection",
    "HardRemoval",
    "OpenUISemanticPlanExtractor",
    "OpenUISemanticPlanCompiler",
    "PlanActionFeatures",
    "PlanAssumption",
    "PlanAssumptionTrail",
    "PlanOracleSubstitutor",
    "PlanSeedBuilder",
    "PlanSeedResult",
    "RestrictionResult",
    "SemanticPlanCompiler",
    "SemanticPlanExtractor",
    "canonicalize_plan",
    "plan_factor_fingerprints",
    "project_authority",
]
