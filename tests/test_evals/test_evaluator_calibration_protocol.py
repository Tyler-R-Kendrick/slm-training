"""VCE-010 (SLM-469): evaluator calibration / blinded adjudication /
risk-coverage protocol tests.

Drives the shipped ``evaluator_calibration_protocol`` entry points -- and,
through them, the real semantic-contrast builder, judge-independence
statistics, blinded-annotation machinery, and campaign contract -- rather
than reimplementing any of them.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from slm_training.autoresearch.experiment_campaign import ExperimentCampaignV1
from slm_training.autoresearch.storage import CampaignStore
from slm_training.evals import evaluator_calibration_protocol as mod
from slm_training.evals.evaluator_calibration_protocol import (
    ARM_IDS,
    EvaluatorCalibrationProtocolV1,
    _human_rows_from_aggregate,
    reason_family_confusion_matrix,
    run_protocol,
)
from slm_training.evals.judge_independence import ExternalJudgeConfig

_VALID_DISPOSITIONS = {"match", "mismatch", "blocked", "inconclusive"}


def test_protocol_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="source_count"):
        EvaluatorCalibrationProtocolV1(source_count=1)
    with pytest.raises(ValueError, match="min_raters"):
        EvaluatorCalibrationProtocolV1(min_raters=1)
    with pytest.raises(ValueError, match="fixture/diagnostic"):
        EvaluatorCalibrationProtocolV1(claim_class="promotion_candidate")
    with pytest.raises(ValueError, match="calibration_error_max"):
        EvaluatorCalibrationProtocolV1(calibration_error_max=0.0)
    with pytest.raises(ValueError, match="MAX_RUN_MINUTES"):
        EvaluatorCalibrationProtocolV1(max_wall_minutes=1000.0)


def test_manifest_is_a_real_valid_campaign_with_four_arms() -> None:
    manifest = EvaluatorCalibrationProtocolV1().manifest()
    assert isinstance(manifest, ExperimentCampaignV1)
    assert {arm.arm_id for arm in manifest.arms} == set(ARM_IDS)
    assert manifest.claim_class == "diagnostic"
    assert manifest.claim_class not in {"promotion_candidate", "ship_gate"}
    control_arms = {arm.arm_id for arm in manifest.arms if arm.role == "control"}
    assert control_arms == {"deterministic_only"}


def test_manifest_rollback_gate_is_not_vacuous_and_is_preregistered() -> None:
    """AC: promotion thresholds are preregistered before external run --
    the gate threshold comes from the protocol config, fixed at manifest
    construction time, before any external-judge or human-label evidence
    exists."""
    protocol = EvaluatorCalibrationProtocolV1(calibration_error_max=0.2)
    gate = next(
        g
        for g in protocol.manifest().rollback_gates
        if g.gate_id == "calibration_error_exceeds_preregistered_ceiling"
    )
    assert gate.operator == "gt"
    assert gate.threshold == 0.2
    assert not (0.0 > gate.threshold)  # a perfect result never fires
    assert 1.0 > gate.threshold  # a broken result does fire


def test_manifest_negative_control_matches_declared_negative_controls() -> None:
    manifest = EvaluatorCalibrationProtocolV1().manifest()
    negative_control_ids = {c.control_id for c in manifest.controls if c.kind == "negative"}
    assert negative_control_ids == set(manifest.negative_controls)
    assert negative_control_ids


def test_module_never_imports_the_compiler_gate_stack() -> None:
    """AC: human/independent evidence cannot become compiler legality --
    enforced structurally: this module must never import the G0-G12
    verifier gate stack."""
    source = inspect.getsource(mod)
    tree = ast.parse(source)
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    forbidden = {"slm_training.data.verify", "slm_training.data.verify.stack"}
    assert not (imported_modules & forbidden)
    assert not any(name.startswith("slm_training.data.verify") for name in imported_modules)


def test_run_protocol_end_to_end_with_real_arms(tmp_path: Path) -> None:
    protocol = EvaluatorCalibrationProtocolV1(source_count=4, seed=3)
    result = run_protocol(protocol, root=tmp_path)

    assert result["claim_class"] == "diagnostic"
    assert result["disposition"] in {"complete", "rollback_gate_fired", "inconclusive"}
    assert len(result["arms"]) == len(ARM_IDS)
    arm_ids = {row["arm"] for row in result["arms"]}
    assert arm_ids == set(ARM_IDS)
    for row in result["arms"]:
        assert row["disposition"] in _VALID_DISPOSITIONS
        assert "compute" in row
        assert set(row["compute"]) == {"forwards", "verifier_calls", "wall_ms"}
    assert "scope_disclaimer" in result and result["scope_disclaimer"]

    store = CampaignStore(protocol.protocol_id, tmp_path)
    locked = store.load_experiment_campaign(protocol.protocol_id)
    assert locked.manifest_sha256 == result["manifest_sha256"]


def test_missing_evidence_arms_are_blocked_not_assumed_valid(tmp_path: Path) -> None:
    """AC: missing external evidence yields blocked/inconclusive, not
    assumed validity."""
    result = run_protocol(EvaluatorCalibrationProtocolV1(source_count=4, seed=3), root=tmp_path)
    by_arm = {row["arm"]: row for row in result["arms"]}

    external_only = by_arm["external_only_missing_human"]
    assert external_only["disposition"] == "blocked"
    assert external_only["reason"] == "missing_human_evidence"
    assert external_only["disposition"] != "match"

    human_only = by_arm["human_only_missing_external"]
    assert human_only["disposition"] == "blocked"
    assert human_only["reason"] == "missing_external_evidence"
    assert human_only["disposition"] != "match"


def test_run_protocol_is_deterministic_on_repeat(tmp_path: Path) -> None:
    protocol = EvaluatorCalibrationProtocolV1(source_count=4, seed=3)
    first = run_protocol(protocol, root=tmp_path / "run1")
    second = run_protocol(protocol, root=tmp_path / "run2")

    assert first["manifest_sha256"] == second["manifest_sha256"]
    assert first["external_human_calibration_error"] == second["external_human_calibration_error"]
    first_dispositions = [(row["arm"], row["disposition"]) for row in first["arms"]]
    second_dispositions = [(row["arm"], row["disposition"]) for row in second["arms"]]
    assert first_dispositions == second_dispositions


def test_independence_is_machine_checked_and_contamination_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC: independence provenance is machine-checked -- a misconfigured
    external judge whose declared model_family overlaps a candidate family
    must fail JudgeEvidenceV1.independent, and the protocol must surface
    that as independence_ok=False / arm disposition="mismatch", never
    silently accept it."""

    def _contaminated_config() -> ExternalJudgeConfig:
        return ExternalJudgeConfig(
            provider="mock_external_judge_provider",
            provider_version="fixture-1",
            model_family="semantic_contrast_positive",  # overlaps candidate_model_families
            model_version="fixture-1",
            rubric_id="vce010_pairwise_acceptability",
            rubric_version="1",
            rubric="x",
            temperature=0.0,
            seed=0,
            max_attempts=2,
            max_tokens=256,
            max_cost_usd=1.0,
        )

    monkeypatch.setattr(mod, "_external_config", _contaminated_config)
    result = run_protocol(EvaluatorCalibrationProtocolV1(source_count=4, seed=3), root=tmp_path)

    assert result["independence_ok"] is False
    full = next(row for row in result["arms"] if row["arm"] == "full_triple_judge")
    assert full["independence_ok"] is False
    assert full["disposition"] == "mismatch"


def test_ambiguous_human_evidence_is_retained_separately_not_defaulted() -> None:
    """AC: ambiguous/UNKNOWN cases retained separately -- an incomplete
    aggregate (disagreement without an adjudicator, or missing raters) must
    never be defaulted to a winner."""
    aggregate = {
        "pairs": [
            {
                "pair_id": "p_incomplete",
                "complete": False,
                "consensus_winner": None,
                "mean_confidence": None,
                "reason_counts": {},
            },
            {
                "pair_id": "p_complete",
                "complete": True,
                "consensus_winner": "left",
                "mean_confidence": 0.8,
                "reason_counts": {"content": 2},
            },
        ]
    }
    rows = _human_rows_from_aggregate(aggregate, {"p_incomplete": "left", "p_complete": "left"})
    by_pair = {row["pair_id"]: row for row in rows}

    incomplete = by_pair["p_incomplete"]
    assert incomplete["human_ambiguous"] is True
    assert incomplete["human_verdict"] == "error"
    assert incomplete["human_pass"] is False

    complete = by_pair["p_complete"]
    assert complete["human_ambiguous"] is False
    assert complete["human_verdict"] == "left"
    assert complete["human_pass"] is True


def test_reason_family_confusion_matrix_is_a_real_crosstab() -> None:
    rows = [
        {
            "stratum": "topology",
            "deterministic_verdict": "left",
            "external_verdict": "right",
            "external_reason_codes": ["binding"],
            "human_verdict": "left",
            "human_reason_codes": ["content"],
        },
        {
            "stratum": "topology",
            "deterministic_verdict": "left",
            "external_verdict": "left",
            "external_reason_codes": ["topology"],
            "human_verdict": "left",
            "human_reason_codes": ["topology"],
        },
    ]
    matrix = reason_family_confusion_matrix(rows)
    assert matrix == {"topology": {"binding": 1}}


def test_run_protocol_with_tiny_sample_is_inconclusive_not_fabricated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC: null/negative/incomplete results are first-class outcomes -- a
    frozen sample too small to analyze must be reported inconclusive, never
    silently padded or skipped."""
    monkeypatch.setattr(mod, "_frozen_stratified_sample", lambda protocol, *, root: [])
    result = run_protocol(EvaluatorCalibrationProtocolV1(source_count=4, seed=3), root=tmp_path)
    assert result["disposition"] == "inconclusive"
    assert result["reason"] == "frozen_stratified_sample_too_small"
    assert result["pair_n"] == 0
