"""Tests for SLM-176 action-shortlist retrieval utilities."""

from __future__ import annotations

import pytest
import torch

from slm_training.dsl.action_descriptions import (
    ActionDescription,
    ActionDescriptionCatalog,
    FixtureDescriptionEncoder,
)
from slm_training.dsl.action_shortlist import (
    ActionShortlistPolicy,
    ActionShortlistTrace,
    build_query_vector,
    retrieve_then_rerank,
)


def _fixture_catalog() -> ActionDescriptionCatalog:
    entries = [
        ActionDescription(
            action_key="+Card",
            short_name="Card",
            signature="+Card()",
            description="Card container.",
            result_type="element",
            argument_roles=(),
            sibling_family="container",
            provenance="schema",
        ),
        ActionDescription(
            action_key="+Button",
            short_name="Button",
            signature="+Button()",
            description="Button action.",
            result_type="element",
            argument_roles=(),
            sibling_family="action",
            provenance="schema",
        ),
        ActionDescription(
            action_key="+Input",
            short_name="Input",
            signature="+Input()",
            description="Input field.",
            result_type="element",
            argument_roles=(),
            sibling_family="input",
            provenance="schema",
        ),
        ActionDescription(
            action_key="-",
            short_name="close",
            signature="-",
            description="Close token.",
            result_type=None,
            argument_roles=(),
            sibling_family=None,
            provenance="structural",
        ),
    ]
    return ActionDescriptionCatalog(entries=tuple(entries))


def test_policy_defaults_and_validation() -> None:
    policy = ActionShortlistPolicy()
    assert policy.mode == "off"
    assert policy.k == 8
    assert policy.min_legal_size == 16
    assert policy.score_margin == pytest.approx(0.0)
    assert policy.fallback_policy == "confidence_and_coverage"
    assert policy.shadow_full_score is False

    with pytest.raises(ValueError):
        ActionShortlistPolicy(mode="unknown")


def test_policy_round_trip() -> None:
    policy = ActionShortlistPolicy(
        mode="description_retrieval",
        k=4,
        min_legal_size=8,
        score_margin=0.1,
        shadow_full_score=True,
    )
    reconstructed = ActionShortlistPolicy.from_dict(policy.to_dict())
    assert reconstructed == policy


def test_trace_round_trip() -> None:
    trace = ActionShortlistTrace(
        legal_action_ids=("+Card", "+Button"),
        shortlist_ids=("+Card",),
        retrieval_scores={"+Card": 1.2, "+Button": 0.5},
        fallback_reason=None,
        shadow_full_selected_id="+Card",
    )
    data = trace.to_dict()
    reconstructed = ActionShortlistTrace.from_dict(data)
    assert reconstructed.legal_action_ids == trace.legal_action_ids
    assert reconstructed.shortlist_ids == trace.shortlist_ids
    assert reconstructed.retrieval_scores == pytest.approx(trace.retrieval_scores)
    assert reconstructed.fallback_reason == trace.fallback_reason
    assert reconstructed.shadow_full_selected_id == trace.shadow_full_selected_id


def test_retrieve_returns_subset_of_legal_actions() -> None:
    catalog = _fixture_catalog()
    vectors = catalog.fixture_vectors(32, source="schema_description")
    query = build_query_vector("container", catalog, FixtureDescriptionEncoder(32))
    legal = tuple(catalog.keys())
    policy = ActionShortlistPolicy(mode="description_retrieval", k=2, min_legal_size=1)
    shortlist, scores, fallback = retrieve_then_rerank(legal, query, vectors, policy)
    assert fallback is None
    assert set(shortlist).issubset(set(legal))
    assert len(shortlist) <= policy.k
    assert set(scores).issubset(set(legal))


def test_retrieve_fallback_when_legal_set_small() -> None:
    catalog = _fixture_catalog()
    vectors = catalog.fixture_vectors(32, source="schema_description")
    query = build_query_vector("container", catalog, FixtureDescriptionEncoder(32))
    legal = ("+Card", "+Button")
    policy = ActionShortlistPolicy(mode="description_retrieval", k=2, min_legal_size=8)
    shortlist, scores, fallback = retrieve_then_rerank(legal, query, vectors, policy)
    assert shortlist == legal
    assert fallback == "legal_set_below_min_legal_size"


def test_retrieve_includes_mandatory_ids() -> None:
    catalog = _fixture_catalog()
    vectors = catalog.fixture_vectors(32, source="schema_description")
    query = build_query_vector("container", catalog, FixtureDescriptionEncoder(32))
    legal = tuple(catalog.keys())
    policy = ActionShortlistPolicy(mode="description_retrieval", k=1, min_legal_size=1)
    shortlist, _, _ = retrieve_then_rerank(
        legal, query, vectors, policy, mandatory_ids=("+Input",)
    )
    assert "+Input" in shortlist


def test_retrieve_full_set_equivalence_when_k_large() -> None:
    catalog = _fixture_catalog()
    vectors = catalog.fixture_vectors(32, source="schema_description")
    query = build_query_vector("container", catalog, FixtureDescriptionEncoder(32))
    legal = tuple(catalog.keys())
    policy = ActionShortlistPolicy(
        mode="description_retrieval", k=len(legal) + 10, min_legal_size=1
    )
    shortlist, _, fallback = retrieve_then_rerank(legal, query, vectors, policy)
    assert fallback is None
    assert set(shortlist) == set(legal)


def test_retrieve_no_gold_injection() -> None:
    catalog = _fixture_catalog()
    vectors = catalog.fixture_vectors(32, source="schema_description")
    query = build_query_vector("container", catalog, FixtureDescriptionEncoder(32))
    legal = ("+Button", "-", "+Input")
    policy = ActionShortlistPolicy(mode="description_retrieval", k=5, min_legal_size=1)
    shortlist, _, _ = retrieve_then_rerank(legal, query, vectors, policy)
    for action_key in shortlist:
        assert action_key in legal


def test_retrieve_default_off_returns_full_set() -> None:
    catalog = _fixture_catalog()
    vectors = catalog.fixture_vectors(32, source="schema_description")
    query = build_query_vector("container", catalog, FixtureDescriptionEncoder(32))
    legal = tuple(catalog.keys())
    policy = ActionShortlistPolicy(mode="off")
    shortlist, _, fallback = retrieve_then_rerank(legal, query, vectors, policy)
    assert shortlist == legal
    assert fallback == "mode_off"


def test_build_query_vector_deterministic() -> None:
    catalog = _fixture_catalog()
    encoder = FixtureDescriptionEncoder(32)
    v1 = build_query_vector("container", catalog, encoder)
    v2 = build_query_vector("container", catalog, encoder)
    assert torch.allclose(v1, v2)
