from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.evals.cap2_disposition import (
    Cap2Capability,
    Cap2CapabilityVerdict,
    CapabilityDispositionV1,
    build_cap2_disposition,
)

ROOT = Path(__file__).resolve().parents[2]


def _reports():
    cap2 = json.loads(
        (
            ROOT
            / "docs/design/dsh3-13-cap2-operator-eval-20260723/report.json"
        ).read_text()
    )
    token = json.loads(
        (
            ROOT
            / "docs/design/e803-reserved-operator-baseline-20260723/report.json"
        ).read_text()
    )
    return cap2, token


def test_disposition_rejects_certificate_and_covers_every_capability() -> None:
    cap2, token = _reports()
    result = build_cap2_disposition(
        cap2_report=cap2,
        token_report=token,
        version_stamp={"stamp_schema": "version_stamp/v1"},
    )
    assert result.cert_cap2_issued is False
    assert result.dsh4_action_distillation_open is False
    assert {item.capability for item in result.capabilities} == set(
        Cap2Capability
    )
    assert all(not item.implemented_benefit for item in result.capabilities)
    by_capability = {item.capability: item for item in result.capabilities}
    assert (
        by_capability[Cap2Capability.DISCRETE_TOKEN_ACTION].verdict
        is Cap2CapabilityVerdict.REJECTED
    )
    assert (
        by_capability[Cap2Capability.HIERARCHICAL_HEAD].verdict
        is Cap2CapabilityVerdict.UNRUN_CONDITIONAL
    )
    assert all(
        evidence.code_identity
        and evidence.data_identity
        and evidence.suite_identity
        and evidence.config_identity
        and evidence.hardware_identity
        and evidence.result_identity
        for evidence in result.evidence
    )


def test_supported_verdict_cannot_exist_without_implemented_evidence() -> None:
    with pytest.raises(ValueError, match="implemented evidence"):
        CapabilityDispositionV1(
            Cap2Capability.SYMBOLIC_TRANSFORM,
            Cap2CapabilityVerdict.SUPPORTED,
            "unsupported claim",
        )


def test_tampered_positive_token_result_fails_closed() -> None:
    cap2, token = _reports()
    token["result"]["verdict"] = "accept"
    token["result"]["accepted"] = True
    with pytest.raises(ValueError, match="does not support"):
        build_cap2_disposition(
            cap2_report=cap2,
            token_report=token,
            version_stamp={"stamp_schema": "version_stamp/v1"},
        )
