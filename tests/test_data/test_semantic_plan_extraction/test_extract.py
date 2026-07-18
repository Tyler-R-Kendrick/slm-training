"""Tests for semantic plan extraction."""

from __future__ import annotations

import pytest

from slm_training.data.progspec.schema import ProgramSpec
from slm_training.data.semantic_plan.extract import OpenUISemanticPlanExtractor
from slm_training.dsl.lang_core import bridge_available
from slm_training.dsl.pack import DslPack


SKIP_REASON = "OpenUI bridge deps missing"


@pytest.mark.skipif(not bridge_available(), reason=SKIP_REASON)
def test_extractor_produces_archetype_roles_topology_symbols_bindings(
    sample_spec: ProgramSpec, pack: DslPack
) -> None:
    plan = OpenUISemanticPlanExtractor().extract(sample_spec, pack)

    assert plan.identity.pack_id == "openui"
    assert plan.identity.provenance == "gold"
    assert plan.archetype.id == "stack_column"
    assert plan.archetype.confidence == 1.0

    families = {slot.component_family for slot in plan.role_slots}
    assert families == {"Stack", "Card", "TextContent", "Button"}

    assert plan.topology.parent_relation_candidates
    edges = plan.topology.parent_relation_candidates
    parents = {edge["parent_role_id"] for edge in edges}
    children = {edge["child_role_id"] for edge in edges}
    assert len(parents) == 2  # Stack and Card
    assert len(children) == 4  # Card, two TextContents, Button

    assert len(plan.symbols) == 3
    roles = {sym.semantic_role for sym in plan.symbols}
    assert roles == {"text", "label"}

    assert len(plan.bindings) == 3
    binding_roles = {b.role_slot_id for b in plan.bindings}
    slot_ids = {slot.role_id for slot in plan.role_slots}
    assert binding_roles <= slot_ids


@pytest.mark.skipif(not bridge_available(), reason=SKIP_REASON)
def test_extractor_abstains_on_unsupported_ast(pack: DslPack) -> None:
    extractor = OpenUISemanticPlanExtractor()
    with pytest.raises(ValueError, match="element dict"):
        extractor.extract(
            ProgramSpec(
                id="bad",
                ast={"not": "element"},
                canonical_openui='root = TextContent(":x.text")',
                facts={},
                contract_id="a" * 16,
                program_family_id="pf1",
                lineage_id="ln1",
                split_group_id="sg1",
            ),
            pack,
        )
