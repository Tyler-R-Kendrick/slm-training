"""SGS-008 (SLM-455): SyGuS/SemGuS capability report tests.

Round-trip fixtures cover a finite symbolic-regression fragment and a
restricted OpenUI fragment, per the issue's acceptance criteria.
"""

from __future__ import annotations

import dataclasses

import pytest

from slm_training.data.progspec.prompt_requirements import (
    AmbiguityGroup,
    PromptSemanticRequirementsV1,
    RequirementFact,
)
from slm_training.data.progspec.synthesis_capability_report import (
    CAPABILITY_REPORT_SCHEMA,
    SyGuSCapabilityReportV1,
    build_sygus_capability_report,
    capability_report_integrity_ok,
)
from slm_training.data.progspec.synthesis_problem import (
    EvidenceProvenanceV1,
    ObjectiveTermV1,
    PackIdentityV1,
    RuntimeSymbolDeclarationV1,
    SearchBudgetV1,
    VerificationRequirementV1,
    VerifiedSynthesisProblemV1,
)


def _symbolic_regression_problem() -> VerifiedSynthesisProblemV1:
    return VerifiedSynthesisProblemV1(
        problem_id="p_sr",
        pack_identity=PackIdentityV1(pack_id="symbolic_regression"),
        runtime_symbols=(
            RuntimeSymbolDeclarationV1(surface="x", role="input_variable"),
            RuntimeSymbolDeclarationV1(surface="y", role="input_variable"),
        ),
        verification_requirements=(
            VerificationRequirementV1(
                requirement_id="r1", kind="evaluator", evaluator_version="v1"
            ),
        ),
        objective=(
            ObjectiveTermV1(name="fit_error", weight=1.0, direction="minimize"),
        ),
    )


def _openui_problem(**overrides: object) -> VerifiedSynthesisProblemV1:
    fields: dict[str, object] = dict(
        problem_id="p_openui",
        pack_identity=PackIdentityV1(pack_id="openui", contract_version=5),
        authored_context="Build a card with a title",
        runtime_symbols=(
            RuntimeSymbolDeclarationV1(surface=":hero.title", role="external_entity"),
        ),
        requirements=PromptSemanticRequirementsV1(
            facts=(
                RequirementFact(
                    fact_id="f1",
                    factor_family="inventory",
                    disposition="required",
                    statement="has a title",
                    confidence=0.9,
                    provenance="predicted",
                    authority="advisory-learned",
                ),
            )
        ),
        verification_requirements=(
            VerificationRequirementV1(requirement_id="r1", kind="gate", gate="G2"),
        ),
        evidence_provenance=(
            EvidenceProvenanceV1(source="oracle", authority="oracle-diagnostic"),
        ),
        search_budget=SearchBudgetV1(max_forwards=64, max_wall_seconds=5.0),
        objective=(ObjectiveTermV1(name="quality", weight=1.0, direction="maximize"),),
    )
    fields.update(overrides)
    return VerifiedSynthesisProblemV1(**fields)


# ---------------------------------------------------------------------------
# Symbolic-regression fragment: a finite, faithfully representable grammar
# ---------------------------------------------------------------------------


def test_symbolic_regression_fixture_declares_representable_variables_and_grammar() -> None:
    report = build_sygus_capability_report(_symbolic_regression_problem())

    representable = {f.element: f for f in report.findings if f.status == "representable"}
    assert "runtime_symbols[x]" in representable
    assert "runtime_symbols[y]" in representable
    assert representable["runtime_symbols[x]"].sygus_fragment == "(declare-var x Real)"
    assert "operator_vocabulary" in representable
    assert "synth-fun" in representable["operator_vocabulary"].sygus_fragment

    assert report.sygus_program is not None
    assert "(set-logic NRA)" in report.sygus_program
    assert "(declare-var x Real)" in report.sygus_program
    assert "(synth-fun expr" in report.sygus_program
    # No (check-synth) -- inspired interoperability, not a runnable problem.
    assert "check-synth" not in report.sygus_program


def test_symbolic_regression_grammar_mirrors_the_typed_ir_operator_vocabulary() -> None:
    from slm_training.dsl.symbolic_expr_ir import BINARY_OPERATORS, UNARY_OPERATORS

    report = build_sygus_capability_report(_symbolic_regression_problem())
    grammar_finding = next(f for f in report.findings if f.element == "operator_vocabulary")
    for op in BINARY_OPERATORS | UNARY_OPERATORS:
        assert f"{op} " in grammar_finding.sygus_fragment or f"({op} " in grammar_finding.sygus_fragment


def test_verification_requirements_are_always_extension_required_even_for_symbolic_regression() -> None:
    report = build_sygus_capability_report(_symbolic_regression_problem())
    finding = next(f for f in report.findings if f.element == "verification_requirements[r1]")
    assert finding.status == "extension_required"


# ---------------------------------------------------------------------------
# Restricted OpenUI fragment: opaque symbol-existence declarations only
# ---------------------------------------------------------------------------


def test_openui_fixture_declares_only_opaque_symbol_existence() -> None:
    report = build_sygus_capability_report(_openui_problem())

    symbol_finding = next(
        f for f in report.findings if f.element == "runtime_symbols[:hero.title]"
    )
    assert symbol_finding.status == "representable"
    assert symbol_finding.sygus_fragment == "(declare-var |:hero.title| String)"

    # No operator-vocabulary grammar for a non-symbolic-regression pack.
    assert not any(f.element == "operator_vocabulary" for f in report.findings)
    assert report.sygus_program is not None
    assert "synth-fun" not in report.sygus_program
    assert "set-logic" not in report.sygus_program


def test_openui_verification_facts_and_objectives_are_marked_unsupported() -> None:
    report = build_sygus_capability_report(_openui_problem())
    by_element = {f.element: f for f in report.findings}

    assert by_element["verification_requirements[r1]"].status == "extension_required"
    assert by_element["requirements.facts[f1]"].status == "non_representable"
    assert by_element["objective[quality]"].status == "non_representable" or (
        by_element["objective[quality]"].status == "extension_required"
    )
    assert by_element["search_budget"].status == "extension_required"
    assert by_element["evidence_provenance[oracle]"].status == "non_representable"


# ---------------------------------------------------------------------------
# Nothing is silently dropped (AC1)
# ---------------------------------------------------------------------------


def test_ambiguity_groups_are_reported_non_representable() -> None:
    problem = _openui_problem(
        requirements=PromptSemanticRequirementsV1(
            facts=(
                RequirementFact(
                    fact_id="f1",
                    factor_family="inventory",
                    disposition="required",
                    statement="a",
                    confidence=0.5,
                    provenance="predicted",
                    authority="advisory-learned",
                ),
                RequirementFact(
                    fact_id="f2",
                    factor_family="inventory",
                    disposition="forbidden",
                    statement="b",
                    confidence=0.5,
                    provenance="predicted",
                    authority="advisory-learned",
                ),
            ),
            ambiguity_groups=(
                AmbiguityGroup(group_id="g1", member_fact_ids=("f1", "f2")),
            ),
        )
    )
    report = build_sygus_capability_report(problem)
    finding = next(f for f in report.findings if f.element == "requirements.ambiguity_groups[g1]")
    assert finding.status == "non_representable"


def test_no_requirement_bearing_element_is_silently_dropped() -> None:
    problem = _openui_problem()
    report = build_sygus_capability_report(problem)

    expected_count = (
        len(problem.runtime_symbols)
        + len(problem.verification_requirements)
        + len(problem.requirements.facts)
        + len(problem.requirements.ambiguity_groups)
        + len(problem.objective)
        + 1  # search_budget is declared in this fixture
        + len(problem.evidence_provenance)
    )
    assert len(report.findings) == expected_count


def test_undeclared_search_budget_produces_no_finding() -> None:
    problem = _openui_problem(search_budget=SearchBudgetV1())
    report = build_sygus_capability_report(problem)
    assert not any(f.element == "search_budget" for f in report.findings)


# ---------------------------------------------------------------------------
# Immutability, determinism, tamper detection
# ---------------------------------------------------------------------------


def test_report_is_frozen_deterministic_and_replayable() -> None:
    problem = _symbolic_regression_problem()
    first = build_sygus_capability_report(problem)
    second = build_sygus_capability_report(problem)
    assert first == second
    assert first.report_hash == second.report_hash
    assert capability_report_integrity_ok(first)

    with pytest.raises(dataclasses.FrozenInstanceError):
        first.sygus_program = None  # type: ignore[misc]

    assert SyGuSCapabilityReportV1.from_dict(first.to_dict()) == first


def test_report_tamper_and_schema_rejection() -> None:
    report = build_sygus_capability_report(_symbolic_regression_problem())

    tampered = report.to_dict()
    tampered["sygus_program"] = "(tampered)"
    with pytest.raises(ValueError, match="hash mismatch"):
        SyGuSCapabilityReportV1.from_dict(tampered)

    wrong_schema = report.to_dict()
    wrong_schema["schema_version"] = "sygus_capability_report/v0"
    with pytest.raises(ValueError, match="unsupported SyGuS capability report schema"):
        SyGuSCapabilityReportV1.from_dict(wrong_schema)


def test_report_is_bound_to_the_exact_problem_digest() -> None:
    problem = _symbolic_regression_problem()
    report = build_sygus_capability_report(problem)
    assert report.problem_digest == problem.digest
    assert report.problem_id == problem.problem_id


# ---------------------------------------------------------------------------
# AC5: documented, machine-readable distinction from standards compliance
# ---------------------------------------------------------------------------


def test_conformance_fields_document_inspired_interoperability() -> None:
    report = build_sygus_capability_report(_symbolic_regression_problem())
    assert report.conformance == "inspired_interoperability"
    assert "not SyGuS-IF standards conformance" in report.conformance_note


def test_schema_version_constant_matches_default() -> None:
    report = build_sygus_capability_report(_symbolic_regression_problem())
    assert report.schema_version == CAPABILITY_REPORT_SCHEMA
