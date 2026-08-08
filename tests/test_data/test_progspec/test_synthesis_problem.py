"""Tests for VerifiedSynthesisProblemV1 (SGS-007 / SLM-444)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from slm_training.data.progspec.prompt_requirements import (
    PromptSemanticRequirementsV1,
    RequirementFact,
)
from slm_training.data.progspec.synthesis_problem import (
    EvidenceProvenanceV1,
    ObjectiveTermV1,
    PackIdentityV1,
    RuntimeSymbolDeclarationV1,
    SearchBudgetV1,
    SYNTHESIS_PROBLEM_SCHEMA_VERSION,
    VerificationRequirementV1,
    VerifiedSynthesisProblemV1,
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
            ObjectiveTermV1(name="complexity", weight=0.1, direction="minimize"),
        ),
    )


# ---------------------------------------------------------------------------
# Round trip / hash stability
# ---------------------------------------------------------------------------


def test_round_trip_preserves_all_fields() -> None:
    problem = _openui_problem()
    restored = VerifiedSynthesisProblemV1.from_dict(problem.to_dict())
    assert restored == problem
    assert restored.digest == problem.digest


def test_digest_is_deterministic_and_content_sensitive() -> None:
    a = _openui_problem()
    b = _openui_problem()
    assert a.digest == b.digest
    c = _openui_problem(authored_context="different context")
    assert c.digest != a.digest


def test_from_dict_rejects_unsupported_version() -> None:
    payload = _openui_problem().to_dict()
    payload["schema_version"] = "2"
    with pytest.raises(ValueError, match="unsupported VerifiedSynthesisProblemV1 version"):
        VerifiedSynthesisProblemV1.from_dict(payload)


# ---------------------------------------------------------------------------
# OpenUI and symbolic-regression fixtures share one top-level contract
# ---------------------------------------------------------------------------


def test_openui_and_symbolic_regression_round_trip_through_the_same_contract() -> None:
    for problem in (_openui_problem(), _symbolic_regression_problem()):
        restored = VerifiedSynthesisProblemV1.from_dict(problem.to_dict())
        assert restored == problem
        assert restored.pack_identity.pack_id in {"openui", "symbolic_regression"}


def test_real_pack_registry_identities_construct_a_valid_envelope() -> None:
    """The two packs landed by SLM-440/441 populate this envelope's identity
    fields without needing any OpenUI-specific runtime-symbol shape."""
    from slm_training.dsl.pack import get_pack
    from slm_training.dsl.symbolic_regression_pack import (
        SymbolicRegressionProblemV1,
    )

    openui_pack = get_pack("openui")
    problem = VerifiedSynthesisProblemV1(
        problem_id="p_registry_openui",
        pack_identity=PackIdentityV1(pack_id=openui_pack.pack_id),
    )
    assert problem.pack_identity.pack_id == "openui"

    sr_pack = get_pack("symbolic_regression")
    sr_problem_contract = SymbolicRegressionProblemV1(
        variables=("x",),
        table=((1.0,),),
        targets=(2.0,),
        allowed_operators=frozenset({"+"}),
        coefficient_hole_policy="fixed_constants_only",
        complexity_budget=5,
        numeric_domain_policy="reject_nonfinite",
        train_indices=(0,),
        validation_indices=(),
    )
    envelope = VerifiedSynthesisProblemV1(
        problem_id="p_registry_sr",
        pack_identity=PackIdentityV1(pack_id=sr_pack.pack_id),
        runtime_symbols=tuple(
            RuntimeSymbolDeclarationV1(surface=name, role="input_variable")
            for name in sr_problem_contract.variables
        ),
        provenance={"symbolic_regression_problem_fingerprint": sr_problem_contract.fingerprint},
    )
    assert envelope.pack_identity.pack_id == "symbolic_regression"
    assert envelope.runtime_symbols[0].surface == "x"


# ---------------------------------------------------------------------------
# No request can declare itself compiler-certified merely through this envelope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("authority", ["compiler-hard", "verifier-hard"])
def test_evidence_provenance_rejects_hard_authority(authority: str) -> None:
    with pytest.raises(ValidationError):
        EvidenceProvenanceV1(source="x", authority=authority)


# ---------------------------------------------------------------------------
# Malformed / authority-escalation and identifier-collision fixtures
# ---------------------------------------------------------------------------


def test_duplicate_runtime_symbol_surfaces_are_rejected() -> None:
    with pytest.raises(ValidationError, match="runtime_symbols surfaces must be unique"):
        _openui_problem(
            runtime_symbols=(
                RuntimeSymbolDeclarationV1(surface="x", role="input_variable"),
                RuntimeSymbolDeclarationV1(surface="x", role="input_variable"),
            )
        )


def test_duplicate_verification_requirement_ids_are_rejected() -> None:
    with pytest.raises(
        ValidationError, match="verification_requirements requirement_id values must be unique"
    ):
        _openui_problem(
            verification_requirements=(
                VerificationRequirementV1(requirement_id="r1", kind="gate", gate="G0"),
                VerificationRequirementV1(requirement_id="r1", kind="gate", gate="G1"),
            )
        )


def test_duplicate_objective_term_names_are_rejected() -> None:
    with pytest.raises(ValidationError, match="objective term names must be unique"):
        _openui_problem(
            objective=(
                ObjectiveTermV1(name="quality", weight=1.0, direction="maximize"),
                ObjectiveTermV1(name="quality", weight=0.5, direction="minimize"),
            )
        )


def test_gate_requirement_without_gate_name_is_rejected() -> None:
    with pytest.raises(ValidationError, match="gate.*requirement must set"):
        VerificationRequirementV1(requirement_id="r1", kind="gate")


def test_evaluator_requirement_without_identity_is_rejected() -> None:
    with pytest.raises(ValidationError, match="evaluator.*requirement must set"):
        VerificationRequirementV1(requirement_id="r1", kind="evaluator")


def test_extra_fields_are_rejected() -> None:
    payload = _openui_problem().to_dict()
    payload["not_a_real_field"] = True
    with pytest.raises(ValidationError):
        VerifiedSynthesisProblemV1.from_dict(payload)


def test_search_budget_rejects_negative_values() -> None:
    with pytest.raises(ValidationError):
        SearchBudgetV1(max_forwards=-1)


# ---------------------------------------------------------------------------
# Generated JSON Schema is committed and current (not hand-maintained)
# ---------------------------------------------------------------------------


def test_generated_json_schema_matches_the_committed_snapshot() -> None:
    from slm_training.bridge_utils import repo_root

    schema_path = (
        repo_root()
        / "src"
        / "slm_training"
        / "resources"
        / "verified_synthesis_problem.schema.json"
    )
    committed = json.loads(schema_path.read_text(encoding="utf-8"))
    generated = VerifiedSynthesisProblemV1.model_json_schema()
    assert committed == generated


def test_schema_version_constant_matches_default() -> None:
    problem = _openui_problem()
    assert problem.schema_version == SYNTHESIS_PROBLEM_SCHEMA_VERSION
