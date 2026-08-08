"""Tests for SGS-006 prompt-requirement integration adapters."""

from __future__ import annotations

from slm_training.data.progspec.prompt_requirements import (
    PromptSemanticRequirementsV1,
    RequirementFact,
)
from slm_training.data.progspec.semantic_plan import (
    PlanArchetype,
    PlanConfidenceCalibration,
    PlanCoverage,
    PlanIdentity,
    PlanTopology,
    RoleSlot,
    SemanticPlanV1,
)
from slm_training.data.semantic_plan.compiler import EvidenceKind
from slm_training.data.semantic_plan.requirements_integrate import (
    annotate_actions_with_requirements,
    requirements_hard_prune_allowed,
    score_plan_against_requirements,
)


def _plan(*families: str) -> SemanticPlanV1:
    slots = tuple(
        RoleSlot(role_id=f"r{i}", component_family=name, min_cardinality=1)
        for i, name in enumerate(families)
    )
    return SemanticPlanV1(
        identity=PlanIdentity(
            pack_id="openui",
            contract_hash="0123456789abcdef",
            provenance="predicted",
        ),
        archetype=PlanArchetype(id="demo", confidence=1.0),
        topology=PlanTopology(),
        coverage=PlanCoverage(),
        confidence_calibration=PlanConfidenceCalibration(),
        role_slots=slots,
    )


def _req(*statements: tuple[str, str, str]) -> PromptSemanticRequirementsV1:
    facts = []
    for index, (disposition, statement, fact_id) in enumerate(statements):
        facts.append(
            RequirementFact(
                fact_id=fact_id or f"f{index}",
                factor_family="inventory",
                disposition=disposition,  # type: ignore[arg-type]
                statement=statement,
                confidence=1.0,
                provenance="retrieved",
                authority="advisory-learned",
            )
        )
    return PromptSemanticRequirementsV1(facts=tuple(facts))


def test_baseline_when_requirements_absent() -> None:
    report = score_plan_against_requirements(_plan("Button"), None)
    assert report.baseline is True
    assert report.facts == ()


def test_satisfied_and_violated_inventory() -> None:
    plan = _plan("Button")
    reqs = _req(
        ("required", "prompt requires component Button", "need_button"),
        ("required", "prompt requires component Input", "need_input"),
        ("forbidden", "prompt forbids component Card", "no_card"),
    )
    report = score_plan_against_requirements(plan, reqs)
    by_id = {fact.fact_id: fact.status for fact in report.facts}
    assert by_id["need_button"] == "satisfied"
    assert by_id["need_input"] == "violated"
    assert by_id["no_card"] == "satisfied"


def test_soft_annotate_preserves_action_count() -> None:
    reqs = _req(("required", "prompt requires component Button", "need_button"))
    actions = ["emit:Button", "emit:Card", "emit:Input"]
    features = annotate_actions_with_requirements(actions, reqs)
    assert len(features) == 3
    assert features[0].component_family_compatible is True
    assert features[1].component_family_compatible is False


def test_no_hard_prune_without_certified_evidence() -> None:
    reqs = _req(("required", "prompt requires component Button", "need_button"))
    assert requirements_hard_prune_allowed(reqs) is False
    assert (
        requirements_hard_prune_allowed(
            reqs, evidence_kinds=(EvidenceKind.COMPILER_AUTHORED_CERTIFIED,)
        )
        is True
    )
