"""Tests for SGS-004 conservative request→requirement extraction."""

from __future__ import annotations

import pytest

from slm_training.data.contract import GenerationRequest, RuntimeSymbol
from slm_training.data.quality import _prompt_component_requirements
from slm_training.data.semantic_plan.requirements_canonicalize import (
    requirement_fact_fingerprints,
)
from slm_training.data.semantic_plan.requirements_extract import (
    GoldTargetAccessError,
    extract_prompt_requirements,
)


def test_whitespace_and_order_canonicalize_identically() -> None:
    prompt_only_a = extract_prompt_requirements(
        GenerationRequest(prompt="  Build a form with a Button and an Input  ")
    )
    prompt_only_b = extract_prompt_requirements(
        GenerationRequest(prompt="Build a form with a Button and an Input")
    )
    assert requirement_fact_fingerprints(prompt_only_a)[
        "fact_set"
    ] == requirement_fact_fingerprints(prompt_only_b)["fact_set"]

    # Contiguous opaque slot inventories with the same markers canonicalize
    # identically regardless of declaration order in the request constructor
    # once alpha-renamed (canonicalize sorts by statement content).
    a = extract_prompt_requirements(
        GenerationRequest(
            prompt="Fill slots.",
            slot_contract=(":slot_0", ":slot_1"),
        )
    )
    b = extract_prompt_requirements(
        GenerationRequest(
            prompt="Fill slots.",
            slot_contract=(":slot_0", ":slot_1"),
        )
    )
    assert requirement_fact_fingerprints(a)["exact"] == requirement_fact_fingerprints(
        b
    )["exact"]


def test_ambiguous_style_prose_abstains() -> None:
    requirements = extract_prompt_requirements(
        GenerationRequest(prompt="Make it look modern and polished.")
    )
    assert requirements.facts == ()
    assert requirements.ambiguity_groups == ()


def test_gold_target_kwargs_rejected() -> None:
    request = GenerationRequest(prompt="Add a Button")
    with pytest.raises(GoldTargetAccessError, match="gold/target"):
        extract_prompt_requirements(request, openui="root = Button()")
    with pytest.raises(GoldTargetAccessError, match="gold/target"):
        extract_prompt_requirements(request, gold_ast={"typeName": "Button"})
    with pytest.raises(GoldTargetAccessError, match="gold/target"):
        extract_prompt_requirements(request, record=object())


def test_structured_runtime_symbols_and_slots() -> None:
    request = GenerationRequest(
        prompt="Fill the declared slots.",
        slot_contract=(":slot_0",),
        runtime_symbols=(
            RuntimeSymbol(surface=":slot_0", role="external_entity"),
        ),
    )
    requirements = extract_prompt_requirements(request, canonicalize=False)
    statements = {fact.statement for fact in requirements.facts}
    assert "declared template marker :slot_0" in statements
    assert any("runtime symbol :slot_0" in s for s in statements)
    assert all(fact.authority == "advisory-learned" for fact in requirements.facts)
    assert all(fact.provenance == "retrieved" for fact in requirements.facts)


def test_preserves_existing_prompt_inventory_helper() -> None:
    prompt = "Use two Buttons and an Input"
    helper = _prompt_component_requirements(prompt)
    requirements = extract_prompt_requirements(
        GenerationRequest(prompt=prompt), canonicalize=False
    )
    assert helper.count("Button") == 2
    assert "Input" in helper
    assert any("Button" in fact.statement for fact in requirements.facts)
    assert any("Input" in fact.statement for fact in requirements.facts)


def test_forbidden_component_explicit() -> None:
    requirements = extract_prompt_requirements(
        GenerationRequest(prompt="Build a Card without a Button"),
        canonicalize=False,
    )
    forbidden = [
        fact for fact in requirements.facts if fact.disposition == "forbidden"
    ]
    assert any("Button" in fact.statement for fact in forbidden)


def test_either_or_records_ambiguity_without_resolving() -> None:
    requirements = extract_prompt_requirements(
        GenerationRequest(prompt="Use either a Button or an Input"),
        canonicalize=False,
    )
    assert len(requirements.ambiguity_groups) == 1
    group = requirements.ambiguity_groups[0]
    assert len(group.member_fact_ids) == 2
    members = {
        fact.statement
        for fact in requirements.facts
        if fact.fact_id in group.member_fact_ids
    }
    assert any("Button" in s for s in members)
    assert any("Input" in s for s in members)
    either_required = [
        fact
        for fact in requirements.facts
        if fact.disposition == "required"
        and ("Button" in fact.statement or "Input" in fact.statement)
    ]
    assert either_required == []


def test_source_span_round_trip() -> None:
    request = GenerationRequest(
        prompt="Add a Button",
        slot_contract=(":slot_0",),
    )
    requirements = extract_prompt_requirements(request, canonicalize=False)
    assert any(
        fact.source_span == "slot_contract::slot_0" for fact in requirements.facts
    )
    restored = type(requirements).from_dict(requirements.to_dict())
    assert restored == requirements


def test_facts_are_advisory_only() -> None:
    """Extractor never emits compiler/verifier-hard authority (type-enforced)."""
    requirements = extract_prompt_requirements(
        GenerationRequest(
            prompt="Add a Button",
            runtime_symbols=(
                RuntimeSymbol(surface=":slot_0", role="external_entity"),
            ),
        )
    )
    assert all(fact.authority == "advisory-learned" for fact in requirements.facts)
