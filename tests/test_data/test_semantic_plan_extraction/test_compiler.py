"""Tests for slm_training.data.semantic_plan.compiler."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.data.progspec.schema import ProgramSpec
from slm_training.data.progspec.semantic_plan import SemanticPlanV1
from slm_training.data.semantic_plan import (
    AuthorityProjectionV1,
    Evidence,
    EvidenceKind,
    OpenUISemanticPlanCompiler,
    OpenUISemanticPlanExtractor,
    PlanAssumption,
    PlanAssumptionTrail,
    PlanSeedResult,
    project_authority,
)
from slm_training.data.semantic_plan.compiler import RestrictionResult
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


# --- project_authority (SGS-005 / SLM-443) ---------------------------------


def test_project_authority_certified_hard_removal_becomes_compiler_hard(
    compiler: OpenUISemanticPlanCompiler,
) -> None:
    evidence = [
        Evidence(
            evidence_id="forbidden_card",
            kind=EvidenceKind.COMPILER_AUTHORED_CERTIFIED,
            certificate="pack_schema:card_forbidden",
        )
    ]
    restriction = compiler.certified_restrictions(None, None, _make_gold_plan(), evidence)
    projection = project_authority(None, restriction, honesty_mode="production")

    assert [e.evidence_id for e in projection.compiler_certified] == ["forbidden_card"]
    assert projection.compiler_certified[0].downgrade_reason is None
    assert projection.production_safe_advisory == ()
    assert projection.evaluation_only == ()


def test_project_authority_downgrades_uncertified_compiler_claim(
    compiler: OpenUISemanticPlanCompiler,
) -> None:
    evidence = [
        Evidence(
            evidence_id="uncertified",
            kind=EvidenceKind.COMPILER_AUTHORED_CERTIFIED,
            certificate=None,
        )
    ]
    restriction = compiler.certified_restrictions(None, None, _make_gold_plan(), evidence)
    projection = project_authority(None, restriction)

    assert projection.compiler_certified == ()
    downgraded = projection.evaluation_only[0]
    assert downgraded.evidence_id == "uncertified"
    assert downgraded.hard_removal is False
    assert "no certified hard removal" in downgraded.downgrade_reason


def test_project_authority_downgrades_unsafe_hard_removal_never_promotable() -> None:
    """A prediction-only hard removal (only reachable via the unsafe escape
    hatch) can never surface as advisory or compiler-hard in a manifest."""
    unsafe_compiler = OpenUISemanticPlanCompiler(
        honesty_mode="oracle_diagnostic", allow_unsafe_predicted_hard_control=True
    )
    evidence = [Evidence(evidence_id="unsafe_prune", kind=EvidenceKind.PREDICTION_ONLY)]
    restriction = unsafe_compiler.certified_restrictions(None, None, _make_gold_plan(), evidence)
    assert any(r.action_id == "unsafe_prune" for r in restriction.hard_removals)  # sanity

    projection = project_authority(None, restriction, honesty_mode="oracle_diagnostic")

    assert projection.compiler_certified == ()
    assert projection.production_safe_advisory == ()
    downgraded = projection.evaluation_only[0]
    assert downgraded.evidence_id == "unsafe_prune"
    assert downgraded.hard_removal is True
    assert "unsafe diagnostic control" in downgraded.downgrade_reason


def test_project_authority_excludes_oracle_gold_from_production() -> None:
    evidence = [Evidence(evidence_id="oracle_hint", kind=EvidenceKind.ORACLE_GOLD)]
    restriction = RestrictionResult(evidence_log=tuple(evidence))
    projection = project_authority(None, restriction, honesty_mode="production")

    assert projection.oracle_diagnostic == ()
    downgraded = projection.evaluation_only[0]
    assert downgraded.evidence_id == "oracle_hint"
    assert "excluded from a production manifest" in downgraded.downgrade_reason


def test_project_authority_keeps_oracle_gold_in_diagnostic_mode() -> None:
    evidence = [Evidence(evidence_id="oracle_hint", kind=EvidenceKind.ORACLE_GOLD)]
    restriction = RestrictionResult(evidence_log=tuple(evidence))
    projection = project_authority(None, restriction, honesty_mode="oracle_diagnostic")

    assert [e.evidence_id for e in projection.oracle_diagnostic] == ["oracle_hint"]
    assert projection.oracle_diagnostic[0].downgrade_reason is None


def test_project_authority_downgrades_tainted_dependency(
    compiler: OpenUISemanticPlanCompiler,
) -> None:
    """Nested provenance smuggling: a certified removal cannot launder an
    oracle-derived dependency into compiler-hard authority."""
    oracle_dep = Evidence(evidence_id="oracle_source", kind=EvidenceKind.ORACLE_GOLD)
    tainted_certified = Evidence(
        evidence_id="laundered",
        kind=EvidenceKind.COMPILER_AUTHORED_CERTIFIED,
        certificate="cert:laundered",
        depends_on=("oracle_source",),
    )
    restriction = compiler.certified_restrictions(
        None, None, _make_gold_plan(), [oracle_dep, tainted_certified]
    )
    assert any(r.action_id == "laundered" for r in restriction.hard_removals)  # sanity

    projection = project_authority(None, restriction, honesty_mode="oracle_diagnostic")

    laundered = next(
        e for e in projection.evaluation_only if e.evidence_id == "laundered"
    )
    assert "oracle" in laundered.downgrade_reason
    assert all(e.evidence_id != "laundered" for e in projection.compiler_certified)


def test_project_authority_never_changes_the_restriction_it_projects(
    compiler: OpenUISemanticPlanCompiler,
) -> None:
    """No projection changes the exact candidate domain by itself."""
    evidence = [
        Evidence(
            evidence_id="forbidden_card",
            kind=EvidenceKind.COMPILER_AUTHORED_CERTIFIED,
            certificate="pack_schema:card_forbidden",
        )
    ]
    restriction = compiler.certified_restrictions(None, None, _make_gold_plan(), evidence)
    before = restriction.to_dict()
    project_authority(None, restriction)
    assert restriction.to_dict() == before


def test_project_authority_advisory_and_certified_buckets_never_overlap(
    compiler: OpenUISemanticPlanCompiler,
) -> None:
    evidence = [
        Evidence(evidence_id="predicted_only", kind=EvidenceKind.PREDICTION_ONLY),
        Evidence(
            evidence_id="certified",
            kind=EvidenceKind.COMPILER_AUTHORED_CERTIFIED,
            certificate="cert:x",
        ),
    ]
    restriction = compiler.certified_restrictions(None, None, _make_gold_plan(), evidence)
    projection = project_authority(None, restriction)

    advisory_ids = {e.evidence_id for e in projection.production_safe_advisory}
    certified_ids = {e.evidence_id for e in projection.compiler_certified}
    assert advisory_ids.isdisjoint(certified_ids)
    assert advisory_ids == {"predicted_only"}
    assert certified_ids == {"certified"}


def test_project_authority_is_idempotent(compiler: OpenUISemanticPlanCompiler) -> None:
    evidence = [
        Evidence(
            evidence_id="forbidden_card",
            kind=EvidenceKind.COMPILER_AUTHORED_CERTIFIED,
            certificate="pack_schema:card_forbidden",
        ),
        Evidence(evidence_id="retrieved_hint", kind=EvidenceKind.RETRIEVAL),
    ]
    restriction = compiler.certified_restrictions(None, None, _make_gold_plan(), evidence)
    once = project_authority(None, restriction)
    twice = project_authority(None, restriction)
    assert once == twice


def test_project_authority_round_trips_through_from_dict(
    compiler: OpenUISemanticPlanCompiler,
) -> None:
    evidence = [
        Evidence(
            evidence_id="forbidden_card",
            kind=EvidenceKind.COMPILER_AUTHORED_CERTIFIED,
            certificate="pack_schema:card_forbidden",
        )
    ]
    restriction = compiler.certified_restrictions(None, None, _make_gold_plan(), evidence)
    projection = project_authority(None, restriction)
    restored = AuthorityProjectionV1.from_dict(projection.to_dict())
    assert restored == projection


def test_project_authority_rejects_invalid_honesty_mode(
    compiler: OpenUISemanticPlanCompiler,
) -> None:
    restriction = compiler.certified_restrictions(None, None, _make_gold_plan(), [])
    with pytest.raises(ValueError, match="honesty_mode"):
        project_authority(None, restriction, honesty_mode="cheat")


def test_project_authority_projects_plan_via_to_production_dict() -> None:
    predicted_plan = SemanticPlanV1(
        identity={"pack_id": "openui", "provenance": "predicted"},  # type: ignore[arg-type]
    )
    restriction = RestrictionResult()
    projection = project_authority(predicted_plan, restriction, honesty_mode="production")
    assert projection.plan == predicted_plan.to_production_dict(honesty_mode="production")
