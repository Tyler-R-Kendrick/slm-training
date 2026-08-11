"""HARN-08 / SLM-537: conservative RM labeling + anti-overclaim tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.harnesses.reasoning.revmath.labeling import (
    assert_label_claim_accepted,
    default_analysis_kind_for_task,
    has_big_five_equivalence_package,
    parse_label_claim,
    report_labeling_disclaimer,
    validate_label_claim,
)
from slm_training.harnesses.reasoning.revmath.schemas import RevmathSchemaError

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "src" / "slm_training" / "resources" / "revmath" / "fixtures"

POSITIVE = (
    "labeling_reversed_equivalent",
    "labeling_interpreted",
    "labeling_practical_computability",
    "labeling_counterexample_known",
    "labeling_rm_inspired_minimization",
)

NEGATIVE = (
    ("labeling_print_axioms_only_rca0", r"print axioms|#print axioms"),
    ("labeling_lean_compile_rca0", r"Lean compile|project-specific axioms|insufficient"),
    ("labeling_finite_bounded_as_rca0", r"finite bounded|interpreted state"),
    ("labeling_classical_existence_as_rca0", r"classical existence"),
    ("labeling_empirical_as_wkl0", r"empirical"),
    ("labeling_ablation_as_equivalence", r"RM-inspired|assumption minimization"),
)


def _load(stem: str):
    return parse_label_claim(
        json.loads((FIXTURES / f"{stem}.claim.json").read_text(encoding="utf-8"))
    )


@pytest.mark.parametrize("stem", POSITIVE)
def test_positive_claims_accepted(stem: str) -> None:
    claim = _load(stem)
    verdict = assert_label_claim_accepted(claim)
    assert verdict.accepted
    assert verdict.labeling_state == claim.labeling_state
    assert not verdict.rejection_reasons


def test_reversed_equivalent_requires_full_package() -> None:
    claim = _load("labeling_reversed_equivalent")
    assert has_big_five_equivalence_package(claim)
    assert claim.proposed_subsystem_id == "WKL0"
    assert claim.coding_id
    assert claim.forward_evidence_sha256
    assert claim.reversal_evidence_sha256


def test_interpreted_is_not_equivalence() -> None:
    claim = _load("labeling_interpreted")
    assert claim.labeling_state == "interpreted"
    assert not has_big_five_equivalence_package(claim)
    assert claim.reversal_evidence_sha256 is None


@pytest.mark.parametrize("stem,pattern", NEGATIVE)
def test_negative_overclaims_rejected(stem: str, pattern: str) -> None:
    claim = _load(stem)
    verdict = validate_label_claim(claim)
    assert not verdict.accepted
    assert verdict.rejection_reasons
    blob = " | ".join(verdict.rejection_reasons)
    assert __import__("re").search(pattern, blob, __import__("re").I)
    with pytest.raises(RevmathSchemaError, match="overclaim rejected"):
        assert_label_claim_accepted(claim)


def test_compile_success_cannot_be_rca0() -> None:
    claim = _load("labeling_lean_compile_rca0")
    assert claim.proposed_subsystem_id == "RCA0"
    reasons = " ".join(validate_label_claim(claim).rejection_reasons)
    assert "Lean compile" in reasons or "project-specific" in reasons


def test_ablation_task_defaults_to_rm_inspired() -> None:
    assert (
        default_analysis_kind_for_task("assumption_ablation")
        == "rm_inspired_assumption_minimization"
    )
    assert default_analysis_kind_for_task("reversal") == "genuine_reverse_mathematics"
    assert (
        default_analysis_kind_for_task("computability_classification")
        == "practical_computability"
    )


def test_report_disclaimer_distinguishes_minimization() -> None:
    text = report_labeling_disclaimer(
        analysis_kind="rm_inspired_assumption_minimization",
        labeling_state="candidate_upper_bound",
    )
    assert "not a Big-Five" in text
    genuine = report_labeling_disclaimer(
        analysis_kind="genuine_reverse_mathematics",
        labeling_state="reversed_equivalent",
    )
    assert "reversal" in genuine.lower()


def test_practical_finite_is_not_rca0() -> None:
    claim = _load("labeling_practical_computability")
    assert claim.proposed_subsystem_id is None
    assert claim.practical_class == "finite_decidable"
    assert claim.labeling_state == "practical_computability_only"
