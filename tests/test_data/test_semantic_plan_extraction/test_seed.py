"""Tests for seed construction from semantic plans."""

from __future__ import annotations

import pytest

from slm_training.data.progspec.schema import ProgramSpec
from slm_training.data.semantic_plan.extract import OpenUISemanticPlanExtractor
from slm_training.data.semantic_plan.seed import PlanSeedBuilder, SeedResult
from slm_training.dsl.lang_core import bridge_available
from slm_training.dsl.pack import DslPack
from slm_training.dsl.parser import validate


SKIP_REASON = "OpenUI bridge deps missing"


@pytest.mark.skipif(not bridge_available(), reason=SKIP_REASON)
def test_seed_builder_produces_valid_seed(
    sample_spec: ProgramSpec, pack: DslPack
) -> None:
    plan = OpenUISemanticPlanExtractor().extract(sample_spec, pack)
    result = PlanSeedBuilder(pack).build(plan)
    assert isinstance(result, SeedResult)
    assert result.ok is True
    assert result.reason is None
    assert result.seed is not None
    validate(result.seed)


@pytest.mark.skipif(not bridge_available(), reason=SKIP_REASON)
def test_single_component_seed(single_spec: ProgramSpec, pack: DslPack) -> None:
    plan = OpenUISemanticPlanExtractor().extract(single_spec, pack)
    result = PlanSeedBuilder(pack).build(plan)
    assert result.ok is True
    assert result.seed is not None
    assert 'root = TextContent("' in result.seed
    validate(result.seed)


@pytest.mark.skipif(not bridge_available(), reason=SKIP_REASON)
def test_seed_builder_preserves_structure(
    sample_spec: ProgramSpec, pack: DslPack
) -> None:
    plan = OpenUISemanticPlanExtractor().extract(sample_spec, pack)
    result = PlanSeedBuilder(pack).build(plan)
    seed = result.seed or ""
    # The sample has a Stack root containing a Card containing two TextContents,
    # plus a sibling Button.
    assert "Stack([" in seed
    assert "Card([" in seed
    assert seed.count("TextContent(") == 2
    assert "Button(" in seed


@pytest.mark.skipif(not bridge_available(), reason=SKIP_REASON)
def test_seed_builder_uses_opaque_marker_identity(
    single_spec: ProgramSpec, pack: DslPack
) -> None:
    plan = OpenUISemanticPlanExtractor().extract(single_spec, pack)
    result = PlanSeedBuilder(pack).build(plan)
    seed = result.seed or ""
    assert ":slot_0." in seed
    assert all(symbol.symbol_id not in seed for symbol in plan.symbols)


@pytest.mark.skipif(not bridge_available(), reason=SKIP_REASON)
def test_seed_builder_is_fail_closed_for_multiple_roots(
    sample_spec: ProgramSpec, pack: DslPack
) -> None:
    plan = OpenUISemanticPlanExtractor().extract(sample_spec, pack)
    # Create a plan with no topology so every role appears as a root.
    broken = plan.model_copy(
        update={
            "topology": plan.topology.model_copy(
                update={"parent_relation_candidates": None}
            )
        }
    )
    result = PlanSeedBuilder(pack).build(broken)
    assert result.ok is False
    assert result.seed is None
    assert "root role" in (result.reason or "").lower()


@pytest.mark.skipif(not bridge_available(), reason=SKIP_REASON)
def test_seed_builder_fails_closed_for_unknown_content_symbol(
    single_spec: ProgramSpec, pack: DslPack
) -> None:
    plan = OpenUISemanticPlanExtractor().extract(single_spec, pack)
    broken = plan.model_copy(
        update={
            "bindings": tuple(
                binding.model_copy(update={"candidate_symbols": ("caller_name",)})
                for binding in plan.bindings
            )
        }
    )
    result = PlanSeedBuilder(pack).build(broken)
    assert result.ok is False
    assert result.seed is None
    assert "unknown content symbol" in (result.reason or "")
