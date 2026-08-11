"""KERN-12 / SLM-532: practical computability classification vocabulary."""

from __future__ import annotations

import json

import pytest

from slm_training.formal import computability_classification as cc


def test_every_class_has_contract() -> None:
    assert set(cc.CLASS_CONTRACTS) == cc.PRACTICAL_CLASSES
    for class_id, contract in cc.CLASS_CONTRACTS.items():
        assert contract.class_id == class_id
        assert contract.meaning.strip()
        assert contract.required_evidence
        assert "RCA0" in contract.non_implications or class_id == "unclassified"
        # unclassified lists other practical classes as non-implications
        if class_id == "unclassified":
            assert "finite_decidable" in contract.non_implications


def test_schema_and_fixtures_exist() -> None:
    assert cc.fixture_path().is_file()
    assert cc.schema_path().is_file()
    schema = json.loads(cc.schema_path().read_text(encoding="utf-8"))
    assert schema["properties"]["schema"]["const"] == cc.SCHEMA


@pytest.mark.parametrize(
    "case",
    cc.load_fixtures()["cases"],
    ids=lambda c: c["id"],
)
def test_fixture_case_parity(case: dict) -> None:
    result = cc.evaluate_fixture_case(case)
    assert result["case_id"] == case["id"]
    if case["expect"].get("accepted"):
        assert result["class_id"] == cc.normalize_class_id(case["expect"]["class_id"])
        if case.get("lean_label"):
            assert cc.lean_label_string(result["class_id"]) == case["lean_label"] or (
                case["lean_label"].startswith("rm_research:")
            )


def test_evaluate_all_stable() -> None:
    first = cc.evaluate_all()
    second = cc.evaluate_all()
    assert first["results_sha256"] == second["results_sha256"]
    assert first["contracts_sha256"] == second["contracts_sha256"]


def test_print_axioms_cannot_yield_rca0() -> None:
    claim = cc.ClassificationClaimV1(
        schema_version=cc.SCHEMA,
        claim_id="unit.print_axioms",
        class_id="rm_research:RCA0",
        evidence_kinds=("print_axioms_only",),
        source_family="unit",
    )
    verdict = cc.validate_classification_claim(claim)
    assert not verdict.accepted
    blob = " ".join(verdict.rejection_reasons).lower()
    assert "print axioms" in blob
    assert "interpretation_package" in blob


def test_legacy_alias_normalizes() -> None:
    assert (
        cc.normalize_class_id("classical_noncomputable_existence")
        == "noncomputable_existence"
    )
    assert cc.validate_practical_class_id("classical_noncomputable_existence") == (
        "noncomputable_existence"
    )


def test_unknown_label_no_fallthrough() -> None:
    with pytest.raises(cc.ComputabilityClassificationError, match="unknown"):
        cc.validate_practical_class_id("quantum_hyperarithmetical")


def test_lean_label_strings_match_inductive() -> None:
    """Python strings must match LeverProofLean.ComputabilityClassification labels."""

    expected = {
        "finite_decidable",
        "primitive_recursive",
        "bounded_search",
        "total_recursive",
        "semidecidable",
        "co_semidecidable",
        "oracle_relative",
        "classical_propositional",
        "noncomputable_existence",
        "unclassified",
    }
    assert set(cc.PRACTICAL_CLASSES) == expected
    for label in expected:
        assert cc.lean_label_string(label) == label


def test_implications_do_not_smuggle_rm() -> None:
    for contract in cc.CLASS_CONTRACTS.values():
        for implied in contract.permitted_implications:
            assert not implied.startswith("rm_research")
            assert implied not in {"RCA0", "WKL0", "ACA0", "ATR0", "Pi1CA0"}
