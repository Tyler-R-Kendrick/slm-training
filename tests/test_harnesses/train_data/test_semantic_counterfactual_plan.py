"""Plan adapter: one-fact counterfactuals as relational derivatives, not new roots."""

from __future__ import annotations

from dataclasses import replace

import pytest

from slm_training.dsl.parser import validate as validate_source
from slm_training.dsl.semantic_frame import CAP1SchemaV1, derive_semantic_frame
from slm_training.harnesses.synthesis_plan import (
    CounterfactualDimension,
    DerivativePolicy,
    tiny_corpus_generation_policy,
)
from slm_training.harnesses.train_data.semantic_counterfactuals import (
    FACT_DIMENSIONS,
    PLAN_ADAPTER_SCHEMA,
    REJECTION_CODES,
    CounterfactualPairV1,
    derive_semantic_relations,
    mutate_program,
    one_fact_proof,
    pair_artifact_nodes,
)
from slm_training.harnesses.train_data.split_policy import RootFamilySplitPolicyV1
from tests.test_harnesses.train_data.test_semantic_counterfactuals import (
    DIMENSION_FIXTURES,
    STACK_SRC,
)


def _policy(
    *,
    count: int,
    dimensions: tuple[str, ...],
) -> DerivativePolicy:
    return replace(
        tiny_corpus_generation_policy().derivatives,
        semantic_counterfactuals_per_eligible_root=count,
        counterfactual_dimensions=tuple(
            CounterfactualDimension(name) for name in dimensions
        ),
    )


def _derive(source: str, *, count: int, dimensions: tuple[str, ...], family: str):
    return derive_semantic_relations(
        source, _policy(count=count, dimensions=dimensions), root_family_id=family
    )


@pytest.mark.parametrize("dimension", FACT_DIMENSIONS)
def test_each_declared_dimension_admits_or_rejects_with_code(dimension: str) -> None:
    result = _derive(
        DIMENSION_FIXTURES[dimension],
        count=1,
        dimensions=(dimension,),
        family="plan-family",
    )
    assert result.attempted_dimensions == (dimension,)
    assert result.independent_root_delta == 0
    assert result.eligible_roots == 1
    outcomes = result.admitted_pair_count + len(result.rejections)
    assert outcomes == 1
    if result.pairs:
        pair = result.pairs[0]
        assert isinstance(pair, CounterfactualPairV1)
        assert pair.dimension == dimension
        assert pair.proof.single_dimension_change is True
        assert pair.source_target and pair.counterfactual_target
        assert pair.root_family_id == "plan-family"
        assert not result.rejections
    else:
        rejection = result.rejections[0]
        assert rejection["code"] in REJECTION_CODES
        assert rejection["dimension"] == dimension


def test_two_dimension_mutation_rejects() -> None:
    schema = CAP1SchemaV1.from_pack("openui")
    source_frame = derive_semantic_frame(STACK_SRC, schema)
    program = validate_source(source_frame.canonical_source, dsl="openui")
    once, spec = mutate_program(program, "closed_value", schema=schema)
    twice, _ = mutate_program(
        validate_source(once, dsl="openui"), "order", schema=schema
    )
    proof = one_fact_proof(
        source_frame, derive_semantic_frame(twice, schema), spec, schema=schema
    )
    assert proof.single_dimension_change is False
    assert proof.dimension_diffs["closed_value"] > 0
    assert proof.dimension_diffs["order"] > 0

    result = _derive(
        STACK_SRC, count=1, dimensions=("closed_value",), family="plan-family"
    )
    assert result.admitted_pair_count == 1
    assert result.pairs[0].proof.single_dimension_change is True
    assert result.independent_root_delta == 0


def test_no_site_mutation_rejects() -> None:
    result = _derive(STACK_SRC, count=1, dimensions=("relation",), family="plan-family")
    assert result.admitted_pair_count == 0
    assert result.attempted_dimensions == ("relation",)
    assert result.rejections[0]["code"] == "no_mutation_site"
    assert result.independent_root_delta == 0


def test_source_and_counterfactual_share_split_family() -> None:
    family = "plan-root-family"
    result = _derive(
        STACK_SRC, count=1, dimensions=("closed_value",), family=family
    )
    assert result.admitted_pair_count == 1
    pair = result.pairs[0]
    expected_split = RootFamilySplitPolicyV1().assign(family)
    assert pair.root_family_id == family
    assert pair.split == expected_split
    source_node, cf_node = pair_artifact_nodes(pair)
    assert source_node.root_family_id == cf_node.root_family_id == family
    assert source_node.split == cf_node.split == expected_split


def test_independent_root_delta_is_always_zero() -> None:
    admitted = _derive(
        STACK_SRC, count=1, dimensions=("closed_value",), family="plan-family"
    )
    rejected = _derive(
        STACK_SRC, count=1, dimensions=("relation",), family="plan-family"
    )
    empty = _derive(STACK_SRC, count=0, dimensions=(), family="plan-family")
    for result in (admitted, rejected, empty):
        assert result.independent_root_delta == 0
        assert result.to_dict()["independent_root_delta"] == 0
        assert result.to_dict()["schema"] == PLAN_ADAPTER_SCHEMA


def test_zero_count_attempts_nothing() -> None:
    result = _derive(STACK_SRC, count=0, dimensions=(), family="plan-family")
    assert result.pairs == ()
    assert result.rejections == ()
    assert result.attempted_dimensions == ()
    assert result.admitted_pair_count == 0
    assert result.eligible_roots == 0
    assert result.independent_root_delta == 0
