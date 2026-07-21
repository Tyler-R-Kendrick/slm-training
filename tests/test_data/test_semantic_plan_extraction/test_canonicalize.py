"""Tests for semantic plan canonicalization and fingerprints."""

from __future__ import annotations

import pytest

from slm_training.data.progspec.semantic_plan import (
    PlanArchetype,
    PlanBinding,
    PlanIdentity,
    PlanSymbol,
    PlanTopology,
    RoleSlot,
    SemanticPlanV1,
)
from slm_training.data.semantic_plan.canonicalize import (
    canonicalize_plan,
    plan_factor_fingerprints,
)
from slm_training.data.semantic_plan.extract import OpenUISemanticPlanExtractor
from slm_training.dsl.lang_core import bridge_available
from slm_training.dsl.pack import DslPack


SKIP_REASON = "OpenUI bridge deps missing"


def _minimal_plan(
    *, role_id: str = "role_a", symbol_id: str = "sym_x"
) -> SemanticPlanV1:
    return SemanticPlanV1(
        identity=PlanIdentity(pack_id="openui", provenance="predicted"),
        archetype=PlanArchetype(id="card", confidence=0.9),
        role_slots=(
            RoleSlot(role_id=role_id, component_family="Card"),
            RoleSlot(role_id="child", component_family="TextContent"),
        ),
        topology=PlanTopology(
            parent_relation_candidates=(
                {
                    "parent_role_id": role_id,
                    "child_role_id": "child",
                    "relation": "contains",
                },
            )
        ),
        symbols=(PlanSymbol(symbol_id=symbol_id, semantic_role="text"),),
        bindings=(
            PlanBinding(
                role_slot_id="child",
                candidate_symbols=(symbol_id,),
                placeholder_fallback=True,
            ),
        ),
    )


@pytest.mark.skipif(not bridge_available(), reason=SKIP_REASON)
def test_canonicalization_is_idempotent(
    sample_spec,
    pack: DslPack,  # noqa: ANN001
) -> None:
    plan = OpenUISemanticPlanExtractor().extract(sample_spec, pack)
    first = canonicalize_plan(plan)
    second = canonicalize_plan(first)
    assert first == second


def test_fingerprints_are_stable_across_alpha_renaming() -> None:
    plan_a = _minimal_plan(role_id="role_foo", symbol_id="sym_0001")
    plan_b = _minimal_plan(role_id="role_bar", symbol_id="sym_0002")

    assert plan_a != plan_b
    fingerprints_a = plan_factor_fingerprints(plan_a)
    fingerprints_b = plan_factor_fingerprints(plan_b)
    assert fingerprints_a == fingerprints_b


def test_symbol_identity_follows_declaration_order_not_name_or_role() -> None:
    first = _minimal_plan(symbol_id="meaningful_title").model_copy(
        update={
            "symbols": (
                PlanSymbol(symbol_id="meaningful_title", semantic_role="title"),
                PlanSymbol(symbol_id="cta", semantic_role="label"),
            )
        }
    )
    renamed = first.model_copy(
        update={
            "symbols": (
                PlanSymbol(symbol_id="z", semantic_role="title"),
                PlanSymbol(symbol_id="a", semantic_role="label"),
            ),
            "bindings": (
                PlanBinding(
                    role_slot_id="child",
                    candidate_symbols=("z",),
                    placeholder_fallback=True,
                ),
            ),
        }
    )
    canonical_first = canonicalize_plan(first)
    canonical_renamed = canonicalize_plan(renamed)
    assert tuple(symbol.symbol_id for symbol in canonical_first.symbols) == (
        "sym_0000",
        "sym_0001",
    )
    assert canonical_first == canonical_renamed


def test_plan_rejects_duplicate_or_dangling_symbol_identity() -> None:
    plan = _minimal_plan()
    duplicate = plan.model_dump()
    duplicate["symbols"] = list(duplicate["symbols"])
    duplicate["symbols"].append(dict(duplicate["symbols"][0]))
    with pytest.raises(ValueError, match="symbol_id values must be unique"):
        SemanticPlanV1.model_validate(duplicate)
    payload = plan.model_dump()
    payload["bindings"][0]["candidate_symbols"] = ("missing",)
    with pytest.raises(ValueError, match="unknown symbol ids"):
        SemanticPlanV1.model_validate(payload)


@pytest.mark.skipif(not bridge_available(), reason=SKIP_REASON)
def test_fingerprints_cover_all_factors(
    sample_spec,
    pack: DslPack,  # noqa: ANN001
) -> None:
    plan = OpenUISemanticPlanExtractor().extract(sample_spec, pack)
    fingerprints = plan_factor_fingerprints(plan)
    assert set(fingerprints) == {
        "exact",
        "archetype",
        "role_set",
        "topology",
        "bindings",
    }
    for value in fingerprints.values():
        assert len(value) == 64  # SHA-256 hex length
