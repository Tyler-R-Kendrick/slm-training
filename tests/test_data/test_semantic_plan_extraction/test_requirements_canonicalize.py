"""Tests for PromptSemanticRequirementsV1 canonicalization and fingerprints."""

from __future__ import annotations

from slm_training.data.progspec.prompt_requirements import (
    AmbiguityGroup,
    PromptSemanticRequirementsV1,
    RequirementFact,
)
from slm_training.data.semantic_plan.requirements_canonicalize import (
    canonicalize_requirements,
    requirement_fact_fingerprints,
)


def _requirements(*, fact_id: str, group_id: str) -> PromptSemanticRequirementsV1:
    facts = (
        RequirementFact(
            fact_id=fact_id,
            factor_family="inventory",
            disposition="required",
            statement="has a Button",
            confidence=0.9,
            provenance="predicted",
            authority="advisory-learned",
        ),
        RequirementFact(
            fact_id=f"{fact_id}_other",
            factor_family="topology",
            disposition="unspecified",
            statement="nesting depth is not mentioned",
            confidence=0.0,
            provenance="predicted",
            authority="advisory-learned",
        ),
    )
    group = AmbiguityGroup(
        group_id=group_id,
        member_fact_ids=(fact_id, f"{fact_id}_other"),
    )
    return PromptSemanticRequirementsV1(facts=facts, ambiguity_groups=(group,))


def test_fingerprints_are_stable_across_alpha_renaming() -> None:
    requirements_a = _requirements(fact_id="fact_foo", group_id="group_foo")
    requirements_b = _requirements(fact_id="fact_bar", group_id="group_bar")

    assert requirements_a != requirements_b
    fingerprints_a = requirement_fact_fingerprints(requirements_a)
    fingerprints_b = requirement_fact_fingerprints(requirements_b)
    assert fingerprints_a == fingerprints_b


def test_canonicalization_is_idempotent() -> None:
    requirements = _requirements(fact_id="fact_x", group_id="group_x")
    first = canonicalize_requirements(requirements)
    second = canonicalize_requirements(first)
    assert first == second


def test_canonical_ids_follow_declaration_content_not_original_name() -> None:
    requirements = _requirements(fact_id="z_last", group_id="g")
    canonical = canonicalize_requirements(requirements)
    assert tuple(fact.fact_id for fact in canonical.facts) == ("fact_0000", "fact_0001")
    assert canonical.ambiguity_groups[0].group_id == "group_0000"


def test_fingerprints_cover_expected_parts() -> None:
    requirements = _requirements(fact_id="fact_x", group_id="group_x")
    fingerprints = requirement_fact_fingerprints(requirements)
    assert set(fingerprints) == {"exact", "fact_set", "authority", "ambiguity_groups"}
    for value in fingerprints.values():
        assert len(value) == 64  # SHA-256 hex length
