"""Regression tests for SDE4-04 pretrained-denoiser activation manifest (SLM-182)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.harnesses.experiments.pretrained_denoiser_activation import (
    ACTIVATION_VERDICTS,
    CAMPAIGN_VERDICTS,
    DEFAULT_ACTIVATION_GATES,
    DEFAULT_ARMS,
    HYPOTHESIS_ID,
    MANIFEST_SCHEMA,
    ActivationGate,
    BudgetCap,
    LicenseTerms,
    PretrainedDenoiserActivationManifest,
    PretrainedDenoiserArm,
    PretrainedDenoiserCandidate,
    build_pretrained_denoiser_activation_manifest,
    validate_pretrained_denoiser_activation_manifest,
)


@pytest.fixture
def license_compatible() -> LicenseTerms:
    return LicenseTerms(
        spdx_id="MIT",
        commercial_use_allowed=True,
        redistribution_allowed=True,
        modification_allowed=True,
        attribution_required=True,
        notes="",
    )


@pytest.fixture
def example_candidate(license_compatible: LicenseTerms) -> PretrainedDenoiserCandidate:
    return PretrainedDenoiserCandidate(
        candidate_id="candidate_1",
        provider="huggingface",
        repository="bert-base-uncased",
        model="bert-base-uncased",
        revision="main",
        file_hashes={"model.safetensors": "abc123"},
        license=license_compatible,
        architecture="bert",
        pretraining_objective="masked_lm",
        parameter_count=110_000_000,
        hidden_width=768,
        num_layers=12,
        context_length=512,
        tokenizer_id="bert-base-uncased",
        conversion_method="hf_to_safetensors",
        supported_formats=("safetensors",),
        estimated_train_memory_bytes=4_000_000_000,
        estimated_inference_memory_bytes=1_000_000_000,
        estimated_flops_per_forward=500_000_000,
        expected_serialized_bytes=440_000_000,
        expected_deployed_bytes=440_000_000,
        local_offline_available=True,
        unsupported_operations=(),
        hardware_requirements=("cuda",),
        selection_evidence="small, open, compatible license",
    )


@pytest.fixture
def all_gates_available() -> tuple[ActivationGate, ...]:
    return tuple(
        ActivationGate(
            gate_id=gate.gate_id,
            depends_on_issue_id=gate.depends_on_issue_id,
            required_status=gate.required_status,
            available=True,
            evidence="mock evidence",
        )
        for gate in DEFAULT_ACTIVATION_GATES
    )


@pytest.fixture
def example_arms() -> tuple[PretrainedDenoiserArm, ...]:
    return DEFAULT_ARMS


def test_default_manifest_is_blocked() -> None:
    manifest = build_pretrained_denoiser_activation_manifest()
    assert manifest.activation_verdict == "activation_blocked"
    assert manifest.activation_status == "blocked"
    assert manifest.campaign_verdict == "unrun"
    assert manifest.schema_version == MANIFEST_SCHEMA
    assert manifest.hypothesis_id == HYPOTHESIS_ID


def test_ready_path(
    example_candidate: PretrainedDenoiserCandidate,
    all_gates_available: tuple[ActivationGate, ...],
    example_arms: tuple[PretrainedDenoiserArm, ...],
) -> None:
    manifest = build_pretrained_denoiser_activation_manifest(
        candidate=example_candidate,
        budget=BudgetCap(total_dollars=500.0),
        arms=example_arms,
        candidate_selection="candidate_selected",
        activation_gates=all_gates_available,
    )
    assert manifest.activation_verdict == "ready_to_spend"
    assert manifest.activation_status == "ready"
    assert validate_pretrained_denoiser_activation_manifest(manifest.to_dict()) == []


def test_no_eligible_candidate_path(
    example_candidate: PretrainedDenoiserCandidate,
    all_gates_available: tuple[ActivationGate, ...],
    example_arms: tuple[PretrainedDenoiserArm, ...],
) -> None:
    manifest = build_pretrained_denoiser_activation_manifest(
        candidate=example_candidate,
        budget=BudgetCap(total_dollars=500.0),
        arms=example_arms,
        candidate_selection="no_candidate_meets_constraints",
        activation_gates=all_gates_available,
    )
    assert manifest.activation_verdict == "no_eligible_candidate"
    assert manifest.activation_status == "closed"


def test_budget_blocked_path(
    example_candidate: PretrainedDenoiserCandidate,
    all_gates_available: tuple[ActivationGate, ...],
    example_arms: tuple[PretrainedDenoiserArm, ...],
) -> None:
    manifest = build_pretrained_denoiser_activation_manifest(
        candidate=example_candidate,
        budget=BudgetCap(total_dollars=0.0),
        arms=example_arms,
        candidate_selection="candidate_selected",
        activation_gates=all_gates_available,
    )
    assert manifest.activation_verdict == "budget_or_yield_blocked"
    assert manifest.activation_status == "blocked"


def test_license_incompatible_blocks_activation(
    example_candidate: PretrainedDenoiserCandidate,
    all_gates_available: tuple[ActivationGate, ...],
    example_arms: tuple[PretrainedDenoiserArm, ...],
) -> None:
    restricted = PretrainedDenoiserCandidate(
        **{**example_candidate.to_dict(), "license": LicenseTerms(
            spdx_id="Proprietary",
            commercial_use_allowed=False,
            redistribution_allowed=False,
            modification_allowed=False,
            attribution_required=False,
            notes="",
        )}
    )
    manifest = build_pretrained_denoiser_activation_manifest(
        candidate=restricted,
        budget=BudgetCap(total_dollars=500.0),
        arms=example_arms,
        candidate_selection="candidate_selected",
        activation_gates=all_gates_available,
    )
    assert manifest.activation_verdict == "license_incompatible"
    assert manifest.activation_status == "blocked"


def test_manifest_round_trip_dict(
    example_candidate: PretrainedDenoiserCandidate,
    all_gates_available: tuple[ActivationGate, ...],
    example_arms: tuple[PretrainedDenoiserArm, ...],
) -> None:
    manifest = build_pretrained_denoiser_activation_manifest(
        candidate=example_candidate,
        budget=BudgetCap(total_dollars=500.0),
        arms=example_arms,
        candidate_selection="candidate_selected",
        activation_gates=all_gates_available,
        primary_metric="component_recall",
        seeds=(7, 8),
        max_deployed_bytes=500_000_000,
        note="round trip",
    )
    data = manifest.to_dict()
    restored = PretrainedDenoiserActivationManifest.from_dict(data)
    assert restored.to_dict() == data
    assert validate_pretrained_denoiser_activation_manifest(data) == []


def test_default_arms_have_expected_eligibility() -> None:
    eligible = [a for a in DEFAULT_ARMS if a.eligible]
    omitted = [a for a in DEFAULT_ARMS if not a.eligible]
    assert len(DEFAULT_ARMS) == 7
    assert len(eligible) == 5
    assert len(omitted) == 2
    for arm in omitted:
        assert arm.omission_reason is not None and arm.omission_reason != ""


def test_validate_manifest_catches_errors() -> None:
    assert validate_pretrained_denoiser_activation_manifest({}) != []
    assert validate_pretrained_denoiser_activation_manifest(
        {"schema_version": "wrong"}
    ) != []

    manifest = build_pretrained_denoiser_activation_manifest()
    data = manifest.to_dict()
    data["activation_verdict"] = "nonsense"
    errors = validate_pretrained_denoiser_activation_manifest(data)
    assert any("activation_verdict" in e for e in errors)

    data = manifest.to_dict()
    data["arms"][0] = {
        "arm_id": "bad",
        "arm_kind": "not_a_kind",
        "eligible": True,
    }
    errors = validate_pretrained_denoiser_activation_manifest(data)
    assert any("arm_kind" in e for e in errors)

    data = manifest.to_dict()
    data["arms"][0] = {
        "arm_id": "bad",
        "arm_kind": "current_small_controller_baseline",
        "eligible": False,
    }
    errors = validate_pretrained_denoiser_activation_manifest(data)
    assert any("omission_reason" in e for e in errors)

    data = manifest.to_dict()
    data["candidate"]["model"] = ""
    errors = validate_pretrained_denoiser_activation_manifest(data)
    assert any("candidate.model" in e for e in errors)


def test_validate_budget_requires_at_least_one_cap() -> None:
    manifest = build_pretrained_denoiser_activation_manifest(
        budget=BudgetCap(),
    )
    errors = validate_pretrained_denoiser_activation_manifest(manifest.to_dict())
    assert any("budget must have at least one cap" in e for e in errors)


def test_verdict_sets_are_frozen() -> None:
    assert "ready_to_spend" in ACTIVATION_VERDICTS
    assert "unrun" in CAMPAIGN_VERDICTS


def test_cli_builds_manifest(tmp_path: Path) -> None:
    from scripts.build_pretrained_denoiser_activation_manifest import main

    out_json = tmp_path / "manifest.json"
    rc = main(["--out-json", str(out_json)])
    assert rc == 0
    assert out_json.is_file()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "version_stamp" in payload
    manifest = payload["manifest"]
    assert manifest["schema_version"] == MANIFEST_SCHEMA
    assert manifest["manifest_hash"] is not None
    assert len(manifest["activation_gates"]) == 6
    assert validate_pretrained_denoiser_activation_manifest(manifest) == []


def test_cli_builds_ready_manifest_with_markdown(tmp_path: Path) -> None:
    from scripts.build_pretrained_denoiser_activation_manifest import main

    out_json = tmp_path / "manifest.json"
    out_md = tmp_path / "manifest.md"
    gate_ids = [g.gate_id for g in DEFAULT_ACTIVATION_GATES]
    rc = main(
        [
            "--candidate-selection",
            "candidate_selected",
            "--provider",
            "huggingface",
            "--model",
            "bert-base-uncased",
            "--license-commercial-use",
            "--license-redistribution",
            "--license-modification",
            "--budget-total-dollars",
            "500",
            *[arg for gate_id in gate_ids for arg in ("--gate-available", gate_id)],
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )
    assert rc == 0
    manifest = json.loads(out_json.read_text(encoding="utf-8"))["manifest"]
    assert manifest["activation_verdict"] == "ready_to_spend"
    assert out_md.is_file()
    assert "# pretrained_denoiser_activation/v1" in out_md.read_text(encoding="utf-8")
