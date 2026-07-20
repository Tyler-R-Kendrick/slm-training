"""Tests for slm_training.data.semantic_plan.compiler."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.data.progspec.schema import ProgramSpec
from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.data.semantic_plan import (
    Evidence,
    EvidenceKind,
    OpenUISemanticPlanCompiler,
    OpenUISemanticPlanExtractor,
    PlanAssumption,
    PlanAssumptionTrail,
    PlanSeedResult,
)
from slm_training.dsl.pack import get_pack
from slm_training.dsl.parser import validate


SIMPLE_SOURCE = 'root = Stack([cta])\ncta = Button(":cta.label")'


def _make_spec(source: str, spec_id: str = "test") -> ProgramSpec:
    return ProgramSpec.from_openui(
        id=spec_id,
        openui=source,
        facts={},
        program_family_id="openui",
        lineage_id="test",
        split_group_id="sg0",
        split="train",
    )


def _make_gold_plan(source: str = SIMPLE_SOURCE) -> SemanticPlanV1:
    spec = _make_spec(source)
    pack = get_pack("openui")
    return OpenUISemanticPlanExtractor().extract(spec, pack)


@pytest.fixture
def compiler() -> OpenUISemanticPlanCompiler:
    return OpenUISemanticPlanCompiler(honesty_mode="oracle_diagnostic")


def test_deterministic_valid_seed_for_canonical_plan(compiler: OpenUISemanticPlanCompiler) -> None:
    plan = _make_gold_plan()
    pack = get_pack("openui")
    result = compiler.build_valid_seed(None, plan, pack)
    assert result.ok is True
    assert result.seed is not None
    validate(result.seed)
    assert result.verifier_outcome == {"validated": True}
    assert result.provenance == "gold"
    assert result.plan_coverage["role_count"] >= 2


def test_partial_unknown_plan_returns_failure_or_baseline(compiler: OpenUISemanticPlanCompiler) -> None:
    pack = get_pack("openui")
    plan = SemanticPlanV1(
        identity={"pack_id": "openui", "provenance": "predicted"},  # type: ignore[arg-type]
    )
    result = compiler.build_valid_seed(None, plan, pack)
    assert result.ok is True
    assert result.seed is None
    assert "no actionable plan" in (result.reason or "")


def test_multiple_roots_fail_closed(compiler: OpenUISemanticPlanCompiler) -> None:
    pack = get_pack("openui")
    plan = SemanticPlanV1(
        identity={"pack_id": "openui", "provenance": "predicted"},  # type: ignore[arg-type]
        role_slots=[
            {"role_id": "r1", "component_family": "Stack"},  # type: ignore[list-item]
            {"role_id": "r2", "component_family": "Stack"},  # type: ignore[list-item]
        ],
    )
    result = compiler.build_valid_seed(None, plan, pack)
    assert result.ok is False
    assert result.seed is None
    assert "root" in (result.reason or "").lower()
    assert result.fail_closed_reason is not None


def test_no_predicted_field_enters_certified_restriction(compiler: OpenUISemanticPlanCompiler) -> None:
    plan = _make_gold_plan()
    evidence = [
        Evidence(
            evidence_id="forbidden_button",
            kind=EvidenceKind.PREDICTION_ONLY,
        )
    ]
    result = compiler.certified_restrictions(None, None, plan, evidence)
    assert result.hard_removals == ()
    assert len(result.soft_removals) == 1
    assert result.soft_removals[0][0] == "forbidden_button"
    assert result.false_hard_prune_count == 0


def test_hard_removal_requires_certificate(compiler: OpenUISemanticPlanCompiler) -> None:
    plan = _make_gold_plan()
    evidence = [
        Evidence(
            evidence_id="required_text",
            kind=EvidenceKind.COMPILER_AUTHORED_CERTIFIED,
            certificate=None,
        )
    ]
    result = compiler.certified_restrictions(None, None, plan, evidence)
    assert result.hard_removals == ()
    assert len(result.unknown_preserved) == 1


def test_certified_hard_removal_with_certificate(compiler: OpenUISemanticPlanCompiler) -> None:
    plan = _make_gold_plan()
    evidence = [
        Evidence(
            evidence_id="forbidden_card",
            kind=EvidenceKind.COMPILER_AUTHORED_CERTIFIED,
            certificate="pack_schema:card_forbidden",
        )
    ]
    result = compiler.certified_restrictions(None, None, plan, evidence)
    assert len(result.hard_removals) == 1
    removal = result.hard_removals[0]
    assert removal.action_id == "forbidden_card"
    assert "pack_schema:card_forbidden" in removal.reason
    assert result.false_hard_prune_count == 0


def test_supported_candidates_survive_eligible_arms(compiler: OpenUISemanticPlanCompiler) -> None:
    plan = _make_gold_plan()
    actions = ["Stack", "Button", "Card", ":cta.label"]
    features = compiler.annotate_actions(None, actions, plan)
    assert len(features) == len(actions)
    # Supported candidates (Stack, Button) must not be flagged conflict.
    supported = {f.action_id for f in features if not f.conflict_or_unknown}
    assert "Stack" in supported
    assert "Button" in supported
    assert any(f.matches_predicted_role or f.component_family_compatible for f in features)
    # No action is ever hard-removed by soft features.
    assert all(f.plan_confidence >= 0.0 for f in features)


def test_unsafe_predicted_hard_control_is_non_promotable() -> None:
    compiler = OpenUISemanticPlanCompiler(
        honesty_mode="oracle_diagnostic",
        allow_unsafe_predicted_hard_control=True,
    )
    plan = _make_gold_plan()
    evidence = [
        Evidence(
            evidence_id="predicted_bad",
            kind=EvidenceKind.PREDICTION_ONLY,
        )
    ]
    result = compiler.certified_restrictions(None, None, plan, evidence)
    removals = [r for r in result.hard_removals if "UNSAFE" in r.reason]
    assert len(removals) == 1
    assert removals[0].action_id == "predicted_bad"


def test_reversible_assumptions_retract_on_rollback() -> None:
    trail = PlanAssumptionTrail()
    a1 = PlanAssumption("a1", "role=Stack")
    a2 = PlanAssumption("a2", "child=Button", depends_on=("a1",))
    trail.push([a1, a2])
    assert len(trail.active) == 2
    removed = trail.rollback()
    assert len(removed) == 2
    assert trail.active == []
    assert not trail._frames


def test_no_plan_path_is_baseline(compiler: OpenUISemanticPlanCompiler) -> None:
    pack = get_pack("openui")
    actions = ["Stack", "Button"]
    seed_result = compiler.build_valid_seed(None, None, pack)
    assert seed_result.seed is None
    assert seed_result.ok is True
    assert "no actionable plan" in (seed_result.reason or "")

    features = compiler.annotate_actions(None, actions, None)
    assert len(features) == 2
    assert all(f.plan_confidence == 0.0 for f in features)
    assert all(f.provenance == "none" for f in features)
    assert all(not f.matches_predicted_role for f in features)


def test_seed_result_schema_round_trip(tmp_path: Path) -> None:
    plan = _make_gold_plan()
    pack = get_pack("openui")
    compiler = OpenUISemanticPlanCompiler()
    result = compiler.build_valid_seed(None, plan, pack)
    data = result.to_dict()
    restored = PlanSeedResult(**data)
    assert restored.ok == result.ok
    assert restored.seed == result.seed


def test_unknown_plan_version_fails_closed() -> None:
    data = {
        "plan_version": "99",
        "identity": {"pack_id": "openui", "provenance": "predicted"},
    }
    with pytest.raises(ValueError, match="unsupported SemanticPlanV1 version"):
        SemanticPlanV1.from_dict(data)


def test_seed_canonicalizer_parity(compiler: OpenUISemanticPlanCompiler) -> None:
    plan = _make_gold_plan()
    pack = get_pack("openui")
    r1 = compiler.build_valid_seed(None, plan, pack)
    r2 = compiler.build_valid_seed(None, plan, pack)
    assert r1.seed == r2.seed
    assert r1.ok == r2.ok


def test_feature_schema_round_trip() -> None:
    from slm_training.data.semantic_plan import PlanActionFeatures

    f = PlanActionFeatures(action_id="a", plan_confidence=0.75)
    data = f.to_dict()
    restored = PlanActionFeatures(**data)
    assert restored.action_id == "a"
    assert restored.plan_confidence == pytest.approx(0.75)


def test_plan_seed_builder_reused_for_openui(compiler: OpenUISemanticPlanCompiler) -> None:
    plan = _make_gold_plan()
    pack = get_pack("openui")
    result = compiler.build_valid_seed(None, plan, pack)
    assert result.seed is not None
    # The seed should contain the root component and at least one child.
    assert "root" in result.seed
    assert "Stack" in result.seed
