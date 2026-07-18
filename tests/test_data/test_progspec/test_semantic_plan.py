from __future__ import annotations

import json
from typing import Any

import pytest

from slm_training.data.progspec.semantic_plan import (
    PlanArchetype,
    PlanIdentity,
    RoleSlot,
    SemanticPlanV1,
)


def _predicted_plan() -> SemanticPlanV1:
    return SemanticPlanV1(
        identity=PlanIdentity(
            pack_id="openui-v1",
            contract_hash="a" * 16,
            prompt_context_hash="ctx_123",
            provenance="predicted",
        ),
        archetype=PlanArchetype(id="profile_card", confidence=0.85),
        role_slots=(
            RoleSlot(
                role_id="header",
                component_family="Card",
                min_cardinality=1,
                max_cardinality=1,
                required=True,
            ),
            RoleSlot(
                role_id="title",
                component_family="Text",
                min_cardinality=0,
                max_cardinality=1,
                required=False,
            ),
        ),
    )


def test_round_trip_preserves_values() -> None:
    plan = _predicted_plan()
    rebuilt = SemanticPlanV1.from_dict(plan.to_dict())
    assert rebuilt == plan


def test_json_round_trip() -> None:
    plan = _predicted_plan()
    serialized = json.dumps(plan.to_dict())
    rebuilt = SemanticPlanV1.from_dict(json.loads(serialized))
    assert rebuilt == plan


def test_unknown_version_rejected() -> None:
    data = _predicted_plan().to_dict()
    data["plan_version"] = "2"
    with pytest.raises(ValueError, match="unsupported"):
        SemanticPlanV1.from_dict(data)


def test_unknown_provenance_rejected() -> None:
    data = _predicted_plan().to_dict()
    data["identity"]["provenance"] = "stolen"
    with pytest.raises(ValueError):
        SemanticPlanV1.model_validate(data)


def test_cardinality_bounds_validated() -> None:
    with pytest.raises(ValueError, match="min_cardinality"):
        RoleSlot(role_id="x", min_cardinality=2, max_cardinality=1)


def test_gold_plan_rejected_from_production() -> None:
    plan = SemanticPlanV1(
        identity=PlanIdentity(
            pack_id="openui-v1",
            contract_hash="a" * 16,
            provenance="gold",
        )
    )
    assert plan.is_oracle_only is True
    with pytest.raises(ValueError, match="cannot enter a production manifest"):
        plan.to_production_dict()


def test_oracle_plan_allowed_in_diagnostic_mode() -> None:
    plan = SemanticPlanV1(
        identity=PlanIdentity(
            pack_id="openui-v1",
            contract_hash="a" * 16,
            source_program_fingerprint="fp_123",
            provenance="oracle_override",
        )
    )
    payload = plan.to_production_dict(honesty_mode="oracle_diagnostic")
    assert payload["identity"]["provenance"] == "oracle_override"
    assert "source_program_fingerprint" not in payload["identity"]


def test_abstained_plan_compiles_to_baseline() -> None:
    plan = SemanticPlanV1(
        identity=PlanIdentity(
            pack_id="openui-v1",
            contract_hash="a" * 16,
            provenance="predicted",
        ),
        archetype=PlanArchetype(id="unknown"),
        role_slots=(RoleSlot(role_id="header", component_family="Card"),),
        confidence_calibration={"abstention_reason": "ambiguous"},
    )
    assert plan.compile_to_baseline() is True


def test_empty_predicted_plan_compiles_to_baseline() -> None:
    plan = SemanticPlanV1(
        identity=PlanIdentity(
            pack_id="openui-v1",
            contract_hash="a" * 16,
            provenance="predicted",
        )
    )
    assert plan.compile_to_baseline() is True


def test_actionable_predicted_plan_does_not_compile_to_baseline() -> None:
    plan = _predicted_plan()
    assert plan.compile_to_baseline() is False


def test_invalid_honesty_mode_rejected() -> None:
    plan = _predicted_plan()
    with pytest.raises(ValueError, match="honesty_mode"):
        plan.to_production_dict(honesty_mode="cheat")


def test_confidence_out_of_range_rejected() -> None:
    data: dict[str, Any] = _predicted_plan().to_dict()
    data["archetype"] = {"confidence": 1.5}
    with pytest.raises(ValueError):
        SemanticPlanV1.model_validate(data)
