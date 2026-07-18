"""Tests for oracle substitution fail-closed behavior."""

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
from slm_training.data.semantic_plan.oracle import PlanOracleSubstitutor


def _baseline_plan() -> SemanticPlanV1:
    return SemanticPlanV1(
        identity=PlanIdentity(pack_id="openui", provenance="predicted"),
        archetype=PlanArchetype(id="stack", confidence=0.6),
        role_slots=(RoleSlot(role_id="r1", component_family="Stack"),),
        topology=PlanTopology(),
    )


def _oracle_plan() -> SemanticPlanV1:
    return SemanticPlanV1(
        identity=PlanIdentity(pack_id="openui", provenance="gold"),
        archetype=PlanArchetype(id="card", confidence=1.0),
        role_slots=(
            RoleSlot(role_id="r1", component_family="Card"),
            RoleSlot(role_id="r2", component_family="TextContent"),
        ),
        topology=PlanTopology(
            parent_relation_candidates=(
                {"parent_role_id": "r1", "child_role_id": "r2", "relation": "contains"},
            )
        ),
        symbols=(PlanSymbol(symbol_id="s1", semantic_role="text"),),
        bindings=(PlanBinding(role_slot_id="r2", candidate_symbols=("s1",)),),
    )


def test_none_source_returns_baseline() -> None:
    baseline = _baseline_plan()
    oracle = _oracle_plan()
    subst = PlanOracleSubstitutor(plan_source="none", oracle_factors=("archetype",))
    result = subst.apply(baseline, oracle)
    assert result == baseline


def test_gold_source_rejected_in_production() -> None:
    subst = PlanOracleSubstitutor(plan_source="gold", oracle_factors=("archetype",))
    with pytest.raises(ValueError, match="gold plan source requires"):
        subst.apply(_baseline_plan(), _oracle_plan())


def test_gold_source_allowed_in_diagnostic_mode() -> None:
    subst = PlanOracleSubstitutor(
        plan_source="gold",
        oracle_factors=("archetype",),
        honesty_mode="oracle_diagnostic",
    )
    result = subst.apply(_baseline_plan(), _oracle_plan())
    assert result.archetype.id == "card"


def test_predicted_source_cannot_consume_gold_oracle() -> None:
    subst = PlanOracleSubstitutor(plan_source="predicted", oracle_factors=("archetype",))
    with pytest.raises(ValueError, match="cannot consume a gold oracle"):
        subst.apply(_baseline_plan(), _oracle_plan())


def test_factor_wise_substitution() -> None:
    subst = PlanOracleSubstitutor(
        plan_source="gold",
        oracle_factors=("archetype", "topology"),
        honesty_mode="oracle_diagnostic",
    )
    result = subst.apply(_baseline_plan(), _oracle_plan())
    assert result.archetype.id == "card"
    assert result.topology.parent_relation_candidates
    assert len(result.role_slots) == 1  # roles not substituted
    assert not result.bindings  # bindings not substituted


def test_contamination_banner_for_gold() -> None:
    subst = PlanOracleSubstitutor(
        plan_source="gold",
        oracle_factors=("archetype",),
        use_mode="features",
        honesty_mode="oracle_diagnostic",
    )
    banner = subst.contamination_banner()
    assert banner is not None
    assert "ORACLE_DIAGNOSTIC" in banner
    assert "features" in banner


def test_no_banner_for_predicted() -> None:
    subst = PlanOracleSubstitutor(plan_source="predicted", oracle_factors=())
    assert subst.contamination_banner() is None
