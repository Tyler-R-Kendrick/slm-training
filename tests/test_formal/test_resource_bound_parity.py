"""KERN-10 / SLM-538: Python↔Lean resource-bound + trigger parity."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from slm_training.formal import resource_bound_parity as rbp
from slm_training.formal.finite_search_bounds import complete_assignment_count
from slm_training.harness_core.lineage.records import canonical_json, content_sha


def _cases() -> list[dict]:
    return list(rbp.load_fixtures()["cases"])


def test_schema_and_fixture_paths_exist() -> None:
    assert rbp.fixture_path().is_file()
    assert rbp.schema_path().is_file()
    schema = json.loads(rbp.schema_path().read_text(encoding="utf-8"))
    assert schema["properties"]["schema"]["const"] == rbp.SCHEMA


def test_fixture_document_schema() -> None:
    doc = rbp.load_fixtures()
    assert doc["schema"] == rbp.SCHEMA
    assert doc["issue"] == rbp.ISSUE
    assert doc["kern"] == rbp.KERN
    assert doc["wall_clock_claim"] is False
    ids = [c["id"] for c in doc["cases"]]
    assert len(ids) == len(set(ids))
    families = {c["family"] for c in doc["cases"]}
    assert "finite_search" in families
    assert "mechanism_trigger" in families
    assert "adversarial" in families


@pytest.mark.parametrize("case", _cases(), ids=lambda c: c["id"])
def test_each_fixture_case_parity(case: dict) -> None:
    result = rbp.compare_case(case)
    assert result["case_id"] == case["id"]
    assert result["status"] == case.get("status", "supported")
    assert "digest" in result


def test_evaluate_all_canonical_stable() -> None:
    first = rbp.evaluate_all()
    second = rbp.evaluate_all()
    assert first["results_sha256"] == second["results_sha256"]
    assert canonical_json(first) == canonical_json(second)
    assert content_sha(first["results"]) == first["results_sha256"]


def test_stale_fixture_schema_fails_closed() -> None:
    doc = rbp.load_fixtures()
    doc = copy.deepcopy(doc)
    doc["schema"] = "resource_bound_trigger_parity/v0"
    with pytest.raises(rbp.UnsupportedConstructError, match="stale_schema"):
        rbp.evaluate_case(
            {
                "id": "forged_stale",
                "family": "adversarial",
                "status": "unsupported",
                "unsupported_reason": "stale_schema_version",
                "inputs": {
                    "probe": "stale_schema_version",
                    "claimed_schema": "resource_bound_trigger_parity/v0",
                },
                "expect": {},
            }
        )
    soft = rbp.compare_case(
        {
            "id": "forged_stale",
            "family": "adversarial",
            "status": "unsupported",
            "unsupported_reason": "stale_schema_version",
            "inputs": {
                "probe": "stale_schema_version",
                "claimed_schema": "resource_bound_trigger_parity/v0",
            },
            "expect": {},
        }
    )
    assert soft["status"] == "unsupported"
    # Document-level load also rejects.
    path = rbp.fixture_path()
    original = path.read_text(encoding="utf-8")
    try:
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(rbp.UnsupportedConstructError, match="stale_schema"):
            rbp.load_fixtures()
    finally:
        path.write_text(original, encoding="utf-8")


def test_parity_mismatch_names_field_theorem_model() -> None:
    case = copy.deepcopy(next(c for c in _cases() if c["id"] == "fs_two_three"))
    case["expect"]["assignment_count"] = 999
    with pytest.raises(rbp.ParityMismatchError) as excinfo:
        rbp.compare_case(case)
    err = excinfo.value
    assert err.case_id == "fs_two_three"
    assert err.field == "assignment_count"
    assert err.lean_module == "LeverProofLean.FiniteSearchBounds"
    assert err.model_id == "ExpansionVerificationQueryModel"
    assert "assignment_count" in str(err)


def test_mutation_of_finite_search_formula_detected() -> None:
    """CI catch: intentional formula drift must disagree with frozen expect."""
    assert complete_assignment_count([2, 3]) == 6
    # Mutating the product formula in-place is simulated by comparing a forged expect.
    case = copy.deepcopy(next(c for c in _cases() if c["id"] == "fs_two_three"))
    case["expect"]["assignment_count"] = complete_assignment_count([2, 3]) + 1
    with pytest.raises(rbp.ParityMismatchError, match="assignment_count"):
        rbp.compare_case(case)


def test_mutation_of_computability_label_detected() -> None:
    case = copy.deepcopy(next(c for c in _cases() if c["id"] == "bb_two_three"))
    case["expect"]["computability_classification"] = "finite_decidable"
    with pytest.raises(rbp.ParityMismatchError, match="computability_classification"):
        rbp.compare_case(case)


def test_unknown_computability_label_fails_closed() -> None:
    with pytest.raises(rbp.UnsupportedConstructError, match="unknown_computability"):
        rbp.evaluate_case(
            {
                "id": "forged_label",
                "family": "adversarial",
                "status": "unsupported",
                "unsupported_reason": "unknown_computability_label",
                "inputs": {
                    "probe": "unknown_computability_label",
                    "label": "rm_research:fake",
                },
                "expect": {},
            }
        )


def test_boundary_coverage_present() -> None:
    ids = {c["id"] for c in _cases()}
    required = {
        "fs_empty",
        "fs_one",
        "fs_two_three",
        "fs_zero_mid",
        "fs_many_duplicates",
        "fs_huge_product",
        "adv_incomplete_domain",
        "adv_unknown_trigger",
        "adv_stale_schema",
        "jd_unknown_timeout",
        "jd_invalid_malformed",
        "adv_cost_model_mismatch",
    }
    assert required <= ids


def test_lean_bindings_cover_supported_numeric_families() -> None:
    lean_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "leverproof_lean"
        / "LeverProofLean"
        / "ResourceBoundParity.lean"
    )
    assert lean_path.is_file()
    text = lean_path.read_text(encoding="utf-8")
    for case in _cases():
        if case.get("status") != "supported":
            continue
        if case["family"] in {"judgment", "singleton", "mechanism_trigger", "bound_ast"}:
            # Bound via owner modules; ResourceBoundParity mirrors numeric cores.
            continue
        if case["family"] in {"finite_search", "black_box", "closure", "event_trace", "computability"}:
            assert case["id"] in text, case["id"]


def test_mechanism_owner_cost_models_surface_identity() -> None:
    models = rbp.mechanism_owner_cost_models()
    assert models["_neural_forward_only"].startswith("NeuralForwardOnlyModel")
    assert "singleton_bypass_absent_no_effect" in models
