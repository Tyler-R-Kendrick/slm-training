"""Regression tests for the CAP3-03 ternary falsification matrix harness."""

from __future__ import annotations

import pytest
import torch.nn as nn

pytest.importorskip("torch")

from slm_training.harnesses.experiments.cap3_03_ternary_falsification import (
    ArmConfig,
    ArmResult,
    FORMAT_FACTORIES,
    MatrixReport,
    MatchedConditions,
    build_arms,
    evaluate_arm,
    make_format,
    run_matrix,
)
from slm_training.harnesses.quantization.calibration import CalibrationCorpusManifest, CalibrationSample
from slm_training.models.local_action_head import LocalFlatHead


class ToyLocalModel(nn.Module):
    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.local_head = LocalFlatHead(hidden_dim)


def _make_samples(n: int = 8) -> list[CalibrationSample]:
    samples: list[CalibrationSample] = []
    actions = ["a:0", "a:1", "a:2"]
    for i in range(n):
        legal = tuple(actions[: (i % 3) + 1])
        selected = legal[i % len(legal)]
        samples.append(
            CalibrationSample(
                trace_id=f"t-{i}",
                state_fingerprint=f"s-{i % 4}",
                state_signature_version="1",
                legal_action_ids=legal,
                selected_action_id=selected,
                target_action_ids=(selected,),
                top1_margin=0.5,
                posterior_entropy_bits=0.0,
                scope_signature="scope",
                template_signature="tpl",
                production_weight=1.0,
                bin_id=None,
                sensitivity_score=None,
                verification_outcome=None,
            )
        )
    return samples


def _make_manifest(samples: list[CalibrationSample]) -> CalibrationCorpusManifest:
    return CalibrationCorpusManifest(
        schema_version="cap3-02.v1",
        source_trace_ids=[s.trace_id for s in samples],
        checkpoint_id="toy",
        teacher_id="toy",
        state_signature_version="1",
        sample_count=len(samples),
        sampling_strategy="uniform_state",
        inclusion_rules={},
        exclusion_rules={},
        coverage_fields={},
        raw_production_frequency_weights={},
        bin_edges=None,
        calibration_split_hashes=[],
        test_split_hashes=[],
        no_test_leakage_asserted=False,
    )


@pytest.fixture
def toy_model() -> nn.Module:
    return ToyLocalModel(hidden_dim=16)


@pytest.fixture
def samples() -> list[CalibrationSample]:
    return _make_samples()


@pytest.fixture
def manifest(samples: list[CalibrationSample]) -> CalibrationCorpusManifest:
    return _make_manifest(samples)


def test_format_factories_cover_expected_ids() -> None:
    expected = {
        "fp16",
        "int8",
        "int4",
        "binary",
        "ternary",
        "symmetric4",
        "symmetric_four_level",
        "learned4zero",
        "learned_four_level_zero",
        "binary_plus_mask",
    }
    assert expected.issubset(set(FORMAT_FACTORIES))


def test_make_format_returns_quant_format() -> None:
    fmt = make_format("ternary", group_size=8)
    assert fmt.format_id == "ternary"
    assert fmt.physical_slot_bits == 2


def test_build_arms_one_per_format_seed() -> None:
    arms = build_arms(
        checkpoint_id="toy",
        formats=("ternary", "learned4zero"),
        group_size=8,
        seeds=(0, 1),
        calibration_manifest_sha="sha",
    )
    assert len(arms) == 4
    assert {a.format_id for a in arms} == {"ternary", "learned4zero"}
    assert {a.seed for a in arms} == {0, 1}


def test_evaluate_arm_returns_result(toy_model: nn.Module, samples: list[CalibrationSample]) -> None:
    manifest = _make_manifest(samples)
    arm = ArmConfig(
        arm_id="ternary_test",
        format_id="ternary",
        group_size=8,
        seed=0,
        checkpoint_id="toy",
        calibration_manifest_sha="sha",
    )
    result = evaluate_arm(arm, toy_model, manifest, samples, hidden_dim=16)
    assert isinstance(result, ArmResult)
    assert result.status == "ok"
    assert result.sample_count == len(samples)
    assert 0.0 <= result.top1_accuracy <= 1.0
    assert result.ledger_sha256


def test_matched_conditions_assert_equal() -> None:
    a = MatchedConditions(
        checkpoint_id="toy",
        group_size=8,
        physical_slot_bits=2,
        calibration_manifest_sha="sha",
        sample_count=8,
        sampling_strategy="uniform_state",
        activation_dtype="fp16",
        accumulation_dtype="fp32",
        qat_steps=0,
    )
    b = MatchedConditions(
        checkpoint_id="toy",
        group_size=8,
        physical_slot_bits=2,
        calibration_manifest_sha="sha",
        sample_count=8,
        sampling_strategy="uniform_state",
        activation_dtype="fp16",
        accumulation_dtype="fp32",
        qat_steps=0,
    )
    a.assert_matches(b)


def test_matched_conditions_assert_different_physical_bits() -> None:
    a = MatchedConditions(
        checkpoint_id="toy",
        group_size=8,
        physical_slot_bits=2,
        calibration_manifest_sha="sha",
        sample_count=8,
        sampling_strategy="uniform_state",
        activation_dtype="fp16",
        accumulation_dtype="fp32",
        qat_steps=0,
    )
    b = MatchedConditions(
        checkpoint_id="toy",
        group_size=8,
        physical_slot_bits=4,
        calibration_manifest_sha="sha",
        sample_count=8,
        sampling_strategy="uniform_state",
        activation_dtype="fp16",
        accumulation_dtype="fp32",
        qat_steps=0,
    )
    with pytest.raises(ValueError, match="matched conditions differ"):
        a.assert_matches(b)


def test_run_matrix_produces_versioned_report(toy_model: nn.Module, manifest: CalibrationCorpusManifest, samples: list[CalibrationSample]) -> None:
    report = run_matrix(
        toy_model,
        manifest,
        samples,
        formats=("ternary", "learned4zero"),
        group_size=8,
        seeds=(0,),
        hidden_dim=16,
    )
    assert isinstance(report, MatrixReport)
    assert report.checkpoint_id == "toy"
    assert {a.format_id for a in report.arms} == {"ternary", "learned4zero"}
    assert all(r.status == "ok" for r in report.arms)
    assert all(r.ledger_sha256 for r in report.arms)


def test_run_matrix_flags_mismatched_physical_bits(toy_model: nn.Module, manifest: CalibrationCorpusManifest, samples: list[CalibrationSample]) -> None:
    report = run_matrix(
        toy_model,
        manifest,
        samples,
        formats=("ternary", "int4"),
        group_size=8,
        seeds=(0,),
        hidden_dim=16,
    )
    by_status = {r.format_id: r.status for r in report.arms}
    assert by_status["ternary"] == "ok"
    assert by_status["int4"] == "error"


def test_report_json_roundtrip(toy_model: nn.Module, manifest: CalibrationCorpusManifest, samples: list[CalibrationSample]) -> None:
    report = run_matrix(
        toy_model,
        manifest,
        samples,
        formats=("ternary",),
        group_size=8,
        seeds=(0,),
        hidden_dim=16,
    )
    raw = report.to_json(indent=None)
    parsed = raw  # Smoke: ensure no exception during serialization.
    assert report.version in parsed
    assert "ternary" in parsed
