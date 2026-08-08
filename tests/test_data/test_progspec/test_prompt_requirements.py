"""Tests for PromptSemanticRequirementsV1 (SGS-003 / SLM-437)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from slm_training.data.progspec.prompt_requirements import (
    AmbiguityGroup,
    PromptSemanticRequirementsV1,
    RequirementFact,
    combine_authority,
)


def _fact(**overrides: object) -> RequirementFact:
    defaults: dict[str, object] = {
        "fact_id": "fact_button",
        "factor_family": "inventory",
        "disposition": "required",
        "statement": "the program must contain a Button",
        "confidence": 0.9,
        "provenance": "predicted",
        "authority": "advisory-learned",
    }
    defaults.update(overrides)
    return RequirementFact(**defaults)


def test_round_trip_preserves_all_fields() -> None:
    facts = (
        _fact(),
        _fact(
            fact_id="fact_style",
            disposition="unspecified",
            statement="layout style is not mentioned",
            confidence=0.0,
        ),
    )
    group = AmbiguityGroup(
        group_id="group_0",
        member_fact_ids=("fact_button", "fact_style"),
        note="mutually exclusive readings of 'card'",
    )
    requirements = PromptSemanticRequirementsV1(
        prompt_context_hash="ctx_abc",
        facts=facts,
        ambiguity_groups=(group,),
    )

    restored = PromptSemanticRequirementsV1.from_dict(requirements.to_dict())

    assert restored == requirements


def test_from_dict_rejects_unsupported_version() -> None:
    payload = PromptSemanticRequirementsV1(facts=(_fact(),)).to_dict()
    payload["requirements_version"] = "2"
    with pytest.raises(ValueError, match="unsupported PromptSemanticRequirementsV1 version"):
        PromptSemanticRequirementsV1.from_dict(payload)


def test_unspecified_is_distinct_from_forbidden() -> None:
    requirements = PromptSemanticRequirementsV1(
        facts=(
            _fact(fact_id="a", disposition="unspecified"),
            _fact(fact_id="b", disposition="forbidden"),
        )
    )
    assert requirements.facts_by_disposition("unspecified")[0].fact_id == "a"
    assert requirements.facts_by_disposition("forbidden")[0].fact_id == "b"
    # Unspecified never appears among forbidden facts, and vice versa.
    assert not any(
        fact.disposition == "forbidden"
        for fact in requirements.facts_by_disposition("unspecified")
    )


def test_ambiguity_group_preserves_conflicting_facts_without_resolution() -> None:
    conflicting = (
        _fact(fact_id="a", disposition="required", statement="uses a Card"),
        _fact(fact_id="b", disposition="forbidden", statement="must not use a Card"),
    )
    group = AmbiguityGroup(group_id="g", member_fact_ids=("a", "b"))
    requirements = PromptSemanticRequirementsV1(facts=conflicting, ambiguity_groups=(group,))

    # Both contradictory facts survive round-trip -- neither is silently dropped.
    restored = PromptSemanticRequirementsV1.from_dict(requirements.to_dict())
    assert {fact.fact_id for fact in restored.facts} == {"a", "b"}
    assert restored.ambiguity_groups[0].member_fact_ids == ("a", "b")


def test_ambiguity_group_rejects_unknown_fact_id() -> None:
    with pytest.raises(ValidationError, match="unknown fact ids"):
        PromptSemanticRequirementsV1(
            facts=(_fact(),),
            ambiguity_groups=(
                AmbiguityGroup(group_id="g", member_fact_ids=("fact_button", "missing")),
            ),
        )


def test_duplicate_fact_ids_rejected() -> None:
    with pytest.raises(ValidationError, match="fact_id values must be unique"):
        PromptSemanticRequirementsV1(facts=(_fact(), _fact()))


@pytest.mark.parametrize("authority", ["compiler-hard", "verifier-hard"])
def test_oracle_and_predicted_evidence_cannot_deserialize_as_hard_authority(
    authority: str,
) -> None:
    """Predicted/retrieved/oracle evidence can never claim compiler/verifier authority."""
    payload = PromptSemanticRequirementsV1(facts=(_fact(),)).to_dict()
    payload["facts"][0]["authority"] = authority
    with pytest.raises(ValidationError):
        PromptSemanticRequirementsV1.from_dict(payload)


def test_gold_provenance_cannot_masquerade_as_prompt_derived() -> None:
    """AST-derived gold plans cannot masquerade as prompt-derived requirements."""
    payload = PromptSemanticRequirementsV1(facts=(_fact(),)).to_dict()
    payload["facts"][0]["provenance"] = "gold"
    with pytest.raises(ValidationError):
        PromptSemanticRequirementsV1.from_dict(payload)


def test_combine_authority_never_escalates() -> None:
    assert combine_authority("advisory-learned", "evaluation-only") == "evaluation-only"
    assert combine_authority("oracle-diagnostic", "advisory-learned") == "oracle-diagnostic"
    assert combine_authority("evaluation-only", "evaluation-only") == "evaluation-only"


def test_merge_never_escalates_authority_or_confidence() -> None:
    strong_claim = PromptSemanticRequirementsV1(
        facts=(_fact(authority="advisory-learned", confidence=0.95),)
    )
    weak_claim = PromptSemanticRequirementsV1(
        facts=(
            _fact(
                authority="evaluation-only",
                provenance="retrieved",
                confidence=0.2,
                statement="contradicting weaker signal",
            ),
        )
    )

    merged = strong_claim.merged_with(weak_claim)

    assert len(merged.facts) == 1
    merged_fact = merged.facts[0]
    assert merged_fact.authority == "evaluation-only"  # weaker of the two, never escalated
    assert merged_fact.confidence == pytest.approx(0.2)
    assert merged_fact.provenance == "merged"


def test_merge_preserves_facts_unique_to_either_side() -> None:
    left = PromptSemanticRequirementsV1(facts=(_fact(fact_id="left_only"),))
    right = PromptSemanticRequirementsV1(facts=(_fact(fact_id="right_only"),))

    merged = left.merged_with(right)

    ids = {fact.fact_id: fact.provenance for fact in merged.facts}
    assert ids == {"left_only": "predicted", "right_only": "predicted"}


def test_facts_and_requirements_are_frozen() -> None:
    fact = _fact()
    requirements = PromptSemanticRequirementsV1(facts=(fact,))
    with pytest.raises(ValidationError):
        fact.confidence = 0.1  # type: ignore[misc]
    with pytest.raises(ValidationError):
        requirements.facts = ()  # type: ignore[misc]


def test_extra_fields_are_rejected() -> None:
    payload = PromptSemanticRequirementsV1(facts=(_fact(),)).to_dict()
    payload["facts"][0]["unexpected_field"] = "smuggled"
    with pytest.raises(ValidationError):
        PromptSemanticRequirementsV1.from_dict(payload)


def test_to_production_dict_rejects_invalid_honesty_mode() -> None:
    requirements = PromptSemanticRequirementsV1(facts=(_fact(),))
    with pytest.raises(ValueError, match="honesty_mode"):
        requirements.to_production_dict(honesty_mode="cheat")


def test_to_production_dict_drops_oracle_facts_and_records_the_reason() -> None:
    oracle_fact = _fact(
        fact_id="fact_oracle",
        provenance="oracle_override",
        authority="oracle-diagnostic",
        statement="oracle-substituted ground truth",
    )
    predicted_fact = _fact()
    requirements = PromptSemanticRequirementsV1(facts=(predicted_fact, oracle_fact))

    payload = requirements.to_production_dict()

    kept_ids = {fact["fact_id"] for fact in payload["facts"]}
    assert kept_ids == {"fact_button"}
    assert payload["dropped_facts"] == [
        {
            "fact_id": "fact_oracle",
            "provenance": "oracle_override",
            "authority": "oracle-diagnostic",
            "downgrade_reason": "oracle_override fact excluded from a production manifest",
        }
    ]


def test_to_production_dict_keeps_oracle_facts_in_diagnostic_mode() -> None:
    requirements = PromptSemanticRequirementsV1(
        facts=(_fact(fact_id="fact_oracle", provenance="oracle_override"),)
    )
    payload = requirements.to_production_dict(honesty_mode="oracle_diagnostic")
    assert {fact["fact_id"] for fact in payload["facts"]} == {"fact_oracle"}
    assert payload["dropped_facts"] == []


def test_to_production_dict_never_rewrites_kept_fact_authority() -> None:
    """Monotonic downgrade: a kept fact's authority is byte-identical, never escalated."""
    fact = _fact(authority="evaluation-only")
    requirements = PromptSemanticRequirementsV1(facts=(fact,))
    payload = requirements.to_production_dict()
    assert payload["facts"][0]["authority"] == "evaluation-only"


def test_to_production_dict_drops_ambiguity_group_with_an_oracle_member() -> None:
    """Nested provenance smuggling: a group referencing a dropped fact cannot survive."""
    oracle_fact = _fact(fact_id="fact_oracle", provenance="oracle_override")
    predicted_fact = _fact(fact_id="fact_predicted")
    group = AmbiguityGroup(group_id="g", member_fact_ids=("fact_oracle", "fact_predicted"))
    requirements = PromptSemanticRequirementsV1(
        facts=(oracle_fact, predicted_fact), ambiguity_groups=(group,)
    )

    payload = requirements.to_production_dict()

    assert payload["ambiguity_groups"] == []


def test_to_production_dict_round_trips_through_from_dict() -> None:
    requirements = PromptSemanticRequirementsV1(facts=(_fact(),))
    payload = requirements.to_production_dict()
    restored = PromptSemanticRequirementsV1.from_dict(
        {key: value for key, value in payload.items() if key != "dropped_facts"}
    )
    assert restored.to_dict() == {
        key: value for key, value in payload.items() if key != "dropped_facts"
    }


def test_to_production_dict_is_idempotent_once_oracle_facts_are_gone() -> None:
    """P(P(x)) == P(x): re-projecting an already-projected requirement set is a fixed point."""
    requirements = PromptSemanticRequirementsV1(
        facts=(_fact(), _fact(fact_id="fact_oracle", provenance="oracle_override"))
    )
    once = requirements.to_production_dict()
    reparsed = PromptSemanticRequirementsV1.from_dict(
        {key: value for key, value in once.items() if key != "dropped_facts"}
    )
    twice = reparsed.to_production_dict()

    assert twice["facts"] == once["facts"]
    assert twice["ambiguity_groups"] == once["ambiguity_groups"]
    assert twice["dropped_facts"] == []  # nothing oracle-tagged left to drop the second time
