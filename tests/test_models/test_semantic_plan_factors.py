"""Regression tests for CAP2-05 SemanticPlanV1 program-factor tensorization."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.data.progspec.semantic_plan import (
    PlanArchetype,
    PlanBinding,
    PlanIdentity,
    PlanSymbol,
    PlanTopology,
    RoleSlot,
    SemanticPlanV1,
)
from slm_training.models.semantic_plan_factors import (
    PROGRAM_FACTOR_FAMILIES,
    ProgramFactorTensorizerConfig,
    concat_factor_tensors,
    factor_dims,
    tensorize_semantic_plan,
    total_feature_dim,
)


def _plan(
    *,
    archetype_id: str = "stack_column",
    component_families: tuple[str, ...] = ("Stack", "TextContent"),
    required_flags: tuple[bool | None, ...] = (True, False),
    with_binding: bool = True,
) -> SemanticPlanV1:
    role_slots = tuple(
        RoleSlot(
            role_id=f"role_{i:04d}",
            component_family=family,
            min_cardinality=0,
            max_cardinality=1,
            required=required_flags[i] if i < len(required_flags) else None,
        )
        for i, family in enumerate(component_families)
    )
    symbols = (PlanSymbol(symbol_id="sym_0000", semantic_role="text"),)
    bindings = (
        (
            PlanBinding(
                role_slot_id=role_slots[-1].role_id,
                candidate_symbols=("sym_0000",),
                placeholder_fallback=True,
            ),
        )
        if with_binding
        else ()
    )
    return SemanticPlanV1(
        identity=PlanIdentity(pack_id="openui-v1", provenance="predicted"),
        archetype=PlanArchetype(id=archetype_id, confidence=1.0),
        role_slots=role_slots,
        topology=PlanTopology(
            parent_relation_candidates=(
                {
                    "parent_role_id": role_slots[0].role_id,
                    "child_role_id": role_slots[-1].role_id,
                    "relation": "contains",
                },
            ),
            depth_bounds=(0, 1),
        ),
        symbols=symbols,
        bindings=bindings,
    )


def test_factor_dims_sum_to_total() -> None:
    config = ProgramFactorTensorizerConfig()
    dims = factor_dims(config)
    assert set(dims) == set(PROGRAM_FACTOR_FAMILIES)
    assert sum(dims.values()) == total_feature_dim(config)


def test_tensorize_is_deterministic() -> None:
    plan = _plan()
    first = tensorize_semantic_plan(plan)
    second = tensorize_semantic_plan(plan)
    for name in PROGRAM_FACTOR_FAMILIES:
        assert torch.equal(first[name], second[name])


def test_concat_matches_family_order_and_width() -> None:
    plan = _plan()
    config = ProgramFactorTensorizerConfig()
    factors = tensorize_semantic_plan(plan, config)
    concatenated = concat_factor_tensors(factors)
    assert concatenated.shape == (total_feature_dim(config),)
    offset = 0
    dims = factor_dims(config)
    for name in PROGRAM_FACTOR_FAMILIES:
        width = dims[name]
        assert torch.equal(concatenated[offset : offset + width], factors[name])
        offset += width


def test_inventory_sensitive_to_component_family() -> None:
    plan_a = _plan(component_families=("Stack", "TextContent"))
    plan_b = _plan(component_families=("Card", "TextContent"))
    factors_a = tensorize_semantic_plan(plan_a)
    factors_b = tensorize_semantic_plan(plan_b)
    assert not torch.equal(factors_a["inventory"], factors_b["inventory"])
    # Topology/binder factors are unaffected by swapping only the root family.
    assert torch.equal(factors_a["topology"], factors_b["topology"])


def test_unknown_component_family_falls_into_overflow_bucket() -> None:
    plan = _plan(component_families=("TotallyUnknownWidget", "TextContent"))
    config = ProgramFactorTensorizerConfig()
    tensor = tensorize_semantic_plan(plan, config)["inventory"]
    # Last coordinate is the unknown/other bucket.
    assert tensor[-1] > 0.0


def test_style_layout_derived_from_archetype_id() -> None:
    column_plan = _plan(archetype_id="stack_column")
    row_plan = _plan(archetype_id="stack_row")
    no_direction_plan = _plan(archetype_id="card")
    column_style = tensorize_semantic_plan(column_plan)["style_layout"]
    row_style = tensorize_semantic_plan(row_plan)["style_layout"]
    card_style = tensorize_semantic_plan(no_direction_plan)["style_layout"]
    assert not torch.equal(column_style, row_style)
    config = ProgramFactorTensorizerConfig()
    # "card" matches no style token -> falls into the trailing unknown bucket.
    assert card_style[-1] == 1.0
    assert card_style.shape == (len(config.style_vocab) + 1,)


def test_binder_reference_reflects_bindings_and_symbols() -> None:
    with_binding = tensorize_semantic_plan(_plan(with_binding=True))["binder_reference"]
    without_binding = tensorize_semantic_plan(_plan(with_binding=False))["binder_reference"]
    assert not torch.equal(with_binding, without_binding)


def test_binder_reference_retains_opaque_declared_binding_target_ordinal() -> None:
    base = _plan()
    first = base.model_copy(update={
        "symbols": (
            PlanSymbol(symbol_id="sym_0000", semantic_role="text"),
            PlanSymbol(symbol_id="sym_0001", semantic_role="text"),
        ),
        "bindings": (
            PlanBinding(
                role_slot_id=base.bindings[0].role_slot_id,
                candidate_symbols=("sym_0000",),
                placeholder_fallback=True,
            ),
        ),
    })
    second = first.model_copy(update={
        "bindings": (
            PlanBinding(
                role_slot_id=first.bindings[0].role_slot_id,
                candidate_symbols=("sym_0001",),
                placeholder_fallback=True,
            ),
        ),
    })

    first_tensor = tensorize_semantic_plan(first)["binder_reference"]
    second_tensor = tensorize_semantic_plan(second)["binder_reference"]

    assert not torch.equal(first_tensor, second_tensor)


def test_cardinality_and_property_role_value_truncate_by_max_role_slots() -> None:
    config = ProgramFactorTensorizerConfig(max_role_slots=1)
    plan = _plan(component_families=("Stack", "TextContent", "Button"))
    cardinality = tensorize_semantic_plan(plan, config)["cardinality"]
    role_value = tensorize_semantic_plan(plan, config)["property_role_value"]
    assert cardinality.shape == (2,)
    assert role_value.shape == (2,)
