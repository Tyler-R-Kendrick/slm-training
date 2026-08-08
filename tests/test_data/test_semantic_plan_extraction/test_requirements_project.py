"""Tests for SGS-005 authority projections."""

from __future__ import annotations

import pytest

from slm_training.data.progspec.prompt_requirements import (
    PromptSemanticRequirementsV1,
    RequirementFact,
)
from slm_training.data.semantic_plan.compiler import EvidenceKind
from slm_training.data.semantic_plan.requirements_canonicalize import (
    requirement_fact_fingerprints,
)
from slm_training.data.semantic_plan.requirements_project import (
    UnsafeAuthorityError,
    project_prompt_requirements,
    requirements_may_hard_prune,
)


def _fact(
    *,
    fact_id: str = "f1",
    provenance: str = "retrieved",
    authority: str = "advisory-learned",
) -> RequirementFact:
    return RequirementFact(
        fact_id=fact_id,
        factor_family="inventory",
        disposition="required",
        statement="requires Button",
        confidence=1.0,
        provenance=provenance,  # type: ignore[arg-type]
        authority=authority,  # type: ignore[arg-type]
    )


def test_production_rejects_oracle_override() -> None:
    requirements = PromptSemanticRequirementsV1(
        facts=(_fact(provenance="oracle_override", authority="oracle-diagnostic"),)
    )
    with pytest.raises(UnsafeAuthorityError, match="oracle_override"):
        project_prompt_requirements(requirements, mode="production")


def test_production_caps_authority_and_is_idempotent() -> None:
    requirements = PromptSemanticRequirementsV1(
        facts=(_fact(authority="oracle-diagnostic"),)
    )
    once = project_prompt_requirements(requirements, mode="production")
    assert once.requirements.facts[0].authority == "advisory-learned"
    assert once.downgrades
    twice = project_prompt_requirements(once.requirements, mode="production")
    assert (
        requirement_fact_fingerprints(once.requirements)["exact"]
        == requirement_fact_fingerprints(twice.requirements)["exact"]
    )


def test_evaluation_only_forces_weakest() -> None:
    requirements = PromptSemanticRequirementsV1(
        facts=(_fact(authority="advisory-learned"),)
    )
    result = project_prompt_requirements(requirements, mode="evaluation_only")
    assert all(f.authority == "evaluation-only" for f in result.requirements.facts)


def test_oracle_diagnostic_allows_oracle_override() -> None:
    requirements = PromptSemanticRequirementsV1(
        facts=(_fact(provenance="oracle_override", authority="oracle-diagnostic"),)
    )
    result = project_prompt_requirements(requirements, mode="oracle_diagnostic")
    assert result.requirements.facts[0].provenance == "oracle_override"


def test_requirements_alone_never_hard_prune() -> None:
    requirements = PromptSemanticRequirementsV1(facts=(_fact(),))
    assert requirements_may_hard_prune(requirements) is False
    assert (
        requirements_may_hard_prune(
            requirements,
            evidence_kinds=(EvidenceKind.PREDICTION_ONLY, EvidenceKind.ORACLE_GOLD),
        )
        is False
    )
    assert (
        requirements_may_hard_prune(
            requirements,
            evidence_kinds=(EvidenceKind.COMPILER_AUTHORED_CERTIFIED,),
        )
        is True
    )


def test_never_escalates_authority_on_merge_like_projection() -> None:
    requirements = PromptSemanticRequirementsV1(
        facts=(_fact(authority="evaluation-only"),)
    )
    result = project_prompt_requirements(requirements, mode="production")
    assert result.requirements.facts[0].authority == "evaluation-only"
