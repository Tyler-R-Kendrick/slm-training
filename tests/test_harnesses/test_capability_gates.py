from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from slm_training.harness_core.checkpoint_reference import CheckpointReferenceV1
from slm_training.harnesses.capability_gates import (
    CapabilityCertificateV1,
    CapabilityGateResultV1,
    CapabilityGateSpecV1,
    ConfidenceBoundV1,
    ConfidenceThresholdV1,
    GateRunStatus,
    PromotionAuthority,
    RetentionResultV1,
    issue_certificate,
    require_training_authorized,
)
from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.harnesses.staged import Capability

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _reference(*, claim_class: str = "frontier") -> CheckpointReferenceV1:
    return CheckpointReferenceV1(
        run_id="run",
        claim_class=claim_class,  # type: ignore[arg-type]
        checkpoint_role="best",
        checkpoint_filename="best.pt",
        size_bytes=1,
        sha256=SHA_A,
        remote_uri="hf://buckets/example/best.pt",
        bucket_id="example",
        training_source_commit=SHA_A,
        evaluation_source_commit=SHA_A,
        model_config_hash=SHA_A,
        tokenizer_hash=SHA_A,
        output_codec_hash=SHA_A,
        context_tower_id="scratch",
        corpus_manifest_hash=SHA_B,
        data_version="v1",
        verification_timestamp="2026-07-23T00:00:00Z",
        verifier_version="v1",
    )


def _evidence(
    capability: Capability,
    *,
    retention: tuple[str, ...] = (),
) -> tuple[CapabilityGateSpecV1, CapabilityGateResultV1, CheckpointReferenceV1]:
    reference = _reference()
    spec = CapabilityGateSpecV1(
        capability=capability,
        thresholds=(ConfidenceThresholdV1("meaningful", 0.8),),
        retention_suite_hashes=retention,
    )
    result = CapabilityGateResultV1(
        capability=capability,
        gate_spec_sha256=spec.sha,
        gate_implementation_sha256=SHA_C,
        checkpoint_reference_sha256=reference.sha,
        checkpoint_sha256=SHA_A,
        dataset_sha256=SHA_B,
        eval_suite_hashes=(SHA_A,),
        code_sha256=SHA_A,
        config_sha256=SHA_B,
        run_class="ship_eval",
        status=GateRunStatus.COMPLETED,
        confidence_bounds=(ConfidenceBoundV1("meaningful", 0.81),),
        retention_results=tuple(
            RetentionResultV1(suite_sha256=value, passed=True)
            for value in retention
        ),
    )
    return spec, result, reference


def _issue(
    capability: Capability,
    *,
    priors: tuple[CapabilityCertificateV1, ...] = (),
) -> CapabilityCertificateV1:
    retention = () if capability is Capability.CAP0_GRAMMAR else (SHA_A,)
    spec, result, reference = _evidence(capability, retention=retention)
    return issue_certificate(
        spec,
        result,
        reference,
        priors=priors,
        authority=PromotionAuthority.CI,
    )


def test_certificate_binds_all_gate_and_artifact_identities() -> None:
    certificate = _issue(Capability.CAP0_GRAMMAR)
    row = certificate.to_dict()
    for field in (
        "gate_spec_sha256",
        "gate_result_sha256",
        "gate_implementation_sha256",
        "checkpoint_reference_sha256",
        "checkpoint_sha256",
        "dataset_sha256",
        "eval_suite_hashes",
        "code_sha256",
        "config_sha256",
    ):
        assert row[field]
    assert CapabilityCertificateV1.from_dict(row) == certificate
    row["code_sha256"] = SHA_C
    with pytest.raises(ValueError, match="certificate_id"):
        CapabilityCertificateV1.from_dict(row)


def test_higher_certificate_requires_complete_ordered_chain() -> None:
    cap0 = _issue(Capability.CAP0_GRAMMAR)
    with pytest.raises(ValueError, match="exactly CAP0_GRAMMAR"):
        _issue(Capability.CAP1_SEMANTICS)
    cap1 = _issue(Capability.CAP1_SEMANTICS, priors=(cap0,))
    cap2 = _issue(Capability.CAP2_TRANSFORM, priors=(cap0, cap1))
    assert cap2.prior_certificate_ids == (cap0.certificate_id, cap1.certificate_id)


def test_retention_regression_prevents_certificate() -> None:
    spec, result, reference = _evidence(
        Capability.CAP1_SEMANTICS, retention=(SHA_A,)
    )
    result = replace(
        result,
        retention_results=(RetentionResultV1(SHA_A, passed=False),),
    )
    with pytest.raises(ValueError, match="retention regression"):
        issue_certificate(
            spec,
            result,
            reference,
            priors=(_issue(Capability.CAP0_GRAMMAR),),
            authority=PromotionAuthority.HUMAN,
        )


@pytest.mark.parametrize(
    "status",
    [
        GateRunStatus.DIAGNOSTIC,
        GateRunStatus.INTERRUPTED,
        GateRunStatus.INVALID,
        GateRunStatus.TIMEOUT,
    ],
)
def test_nonterminal_results_cannot_certify(status: GateRunStatus) -> None:
    spec, result, reference = _evidence(Capability.CAP0_GRAMMAR)
    with pytest.raises(ValueError, match="cannot certify"):
        issue_certificate(
            spec,
            replace(result, status=status),
            reference,
            authority=PromotionAuthority.HUMAN,
        )


def test_fixture_checkpoint_cannot_certify() -> None:
    spec, result, _ = _evidence(Capability.CAP0_GRAMMAR)
    reference = _reference(claim_class="fixture")
    result = replace(result, checkpoint_reference_sha256=reference.sha)
    with pytest.raises(ValueError, match="fixture or diagnostic"):
        issue_certificate(
            spec,
            result,
            reference,
            authority=PromotionAuthority.HUMAN,
        )


def test_training_preflight_rejects_uncertified_stage_and_forbidden_levers(
    tmp_path: Path,
) -> None:
    config = ModelBuildConfig(
        train_dir=tmp_path,
        requested_capability=Capability.CAP1_SEMANTICS.value,
    )
    with pytest.raises(ValueError, match="exactly CAP0_GRAMMAR"):
        require_training_authorized(config)

    cap0 = _issue(Capability.CAP0_GRAMMAR)
    path = tmp_path / "cap0.json"
    cap0.write(path)
    config.capability_certificates = (path,)
    config.action_shortlist_mode = "retrieve"
    with pytest.raises(ValueError, match="requires CAP2_TRANSFORM"):
        require_training_authorized(config)


def test_distillation_is_an_independent_permission(tmp_path: Path) -> None:
    cap0 = _issue(Capability.CAP0_GRAMMAR)
    path = tmp_path / "cap0.json"
    cap0.write(path)
    config = ModelBuildConfig(
        train_dir=tmp_path,
        requested_capability=Capability.CAP1_SEMANTICS.value,
        capability_certificates=(path,),
        capability_distillation=True,
    )
    with pytest.raises(ValueError, match="distillation requires"):
        require_training_authorized(config)


def test_dataset_capability_is_bound_to_explicit_plan(tmp_path: Path) -> None:
    from slm_training.harnesses.synthesis_plan import SynthesisPlanV1

    plan_path = Path(
        "src/slm_training/resources/synthesis_plans/dsh0_cap0_fixture.json"
    )
    plan = SynthesisPlanV1.load(plan_path)
    manifest = {
        "synthesis_plan": {"plan_id": plan.plan_id, "sha256": plan.sha}
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    config = ModelBuildConfig(
        train_dir=tmp_path,
        requested_capability=Capability.CAP0_GRAMMAR.value,
        capability_plan=plan_path,
    )
    require_training_authorized(config)

    manifest["synthesis_plan"]["sha256"] = SHA_C
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not bind"):
        require_training_authorized(config)


def test_train_fails_before_data_or_model_initialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from slm_training.harnesses.model_build import train_loop

    called = {"data": False, "model": False}
    monkeypatch.setattr(
        train_loop,
        "load_train_records",
        lambda _path: called.__setitem__("data", True),
    )
    monkeypatch.setattr(
        train_loop,
        "build_model",
        lambda *_args: called.__setitem__("model", True),
    )
    config = ModelBuildConfig(
        train_dir=tmp_path,
        requested_capability=Capability.CAP2_TRANSFORM.value,
    )
    with pytest.raises(ValueError, match="prior certificate capabilities"):
        train_loop.train(config)
    assert called == {"data": False, "model": False}


def test_dry_run_never_writes_and_human_promotion_is_explicit(
    tmp_path: Path,
) -> None:
    from scripts.manage_capability_certificate import main

    spec, result, reference = _evidence(Capability.CAP0_GRAMMAR)
    spec_path = tmp_path / "spec.json"
    result_path = tmp_path / "result.json"
    reference_path = tmp_path / "checkpoint.ref.json"
    output_path = tmp_path / "certificate.json"
    spec_path.write_text(json.dumps(spec.to_dict()), encoding="utf-8")
    result_path.write_text(json.dumps(result.to_dict()), encoding="utf-8")
    reference.write_json(reference_path)
    common = [
        "--spec",
        str(spec_path),
        "--result",
        str(result_path),
        "--checkpoint-reference",
        str(reference_path),
        "--output",
        str(output_path),
    ]

    assert main(["dry-run", *common]) == 0
    assert not output_path.exists()
    with pytest.raises(SystemExit):
        main(["promote", *common, "--authority", "human"])
    assert (
        main(
            [
                "promote",
                *common,
                "--authority",
                "human",
                "--confirm-human",
            ]
        )
        == 0
    )
    assert CapabilityCertificateV1.load(output_path).capability is Capability.CAP0_GRAMMAR
    ci_output = tmp_path / "ci-certificate.json"
    ci_common = [value if value != str(output_path) else str(ci_output) for value in common]
    with pytest.raises(SystemExit):
        main(["promote", *ci_common, "--authority", "ci"])
    assert (
        main(["promote", *ci_common, "--authority", "ci", "--ci-attested"]) == 0
    )
    assert CapabilityCertificateV1.load(ci_output).promotion_authority is PromotionAuthority.CI
