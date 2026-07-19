"""Regression tests for SDE3-03 proxy-metric calibration manifest (SLM-177)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.harnesses.experiments.proxy_metric_calibration import (
    ACTIVATION_VERDICTS,
    CAMPAIGN_VERDICTS,
    DEFAULT_ACTIVATION_GATES,
    DEFAULT_ARMS,
    HYPOTHESIS_ID,
    MANIFEST_SCHEMA,
    ActivationGate,
    BudgetCap,
    CalibrationArm,
    ProxyFeatureSet,
    ProxyMetricCalibrationManifest,
    build_proxy_metric_calibration_manifest,
    validate_proxy_metric_calibration_manifest,
)


@pytest.fixture
def example_feature_set() -> ProxyFeatureSet:
    return ProxyFeatureSet(
        feature_schema_version="proxy_features/v1",
        feature_names=(
            "parser_valid",
            "binding_aware_meaningful_rate",
            "component_count",
        ),
        target_primary="binding_aware_meaningful_program_rate",
        target_gate="full_gate_pass",
        allowed_sources=("parser", "binding_aware_metric"),
        forbidden_features=(
            "agentv_score",
            "external_judge_score",
            "full_gate_result",
            "gold_action_trace",
            "checkpoint_id",
        ),
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
def example_arms() -> tuple[CalibrationArm, ...]:
    return DEFAULT_ARMS


def test_default_manifest_is_blocked() -> None:
    manifest = build_proxy_metric_calibration_manifest(manifest_id="sde3-03-v1")
    assert manifest.activation_verdict == "activation_blocked"
    assert manifest.activation_status == "blocked"
    assert manifest.campaign_verdict == "unrun"
    assert manifest.schema_version == MANIFEST_SCHEMA
    assert manifest.hypothesis_id == HYPOTHESIS_ID
    assert manifest.proxy_eval_mode == "off"


def test_ready_path(
    example_feature_set: ProxyFeatureSet,
    all_gates_available: tuple[ActivationGate, ...],
    example_arms: tuple[CalibrationArm, ...],
) -> None:
    manifest = build_proxy_metric_calibration_manifest(
        manifest_id="sde3-03-v1",
        feature_set=example_feature_set,
        budget=BudgetCap(max_historical_rows=5000, total_dollars=100.0),
        arms=example_arms,
        activation_gates=all_gates_available,
        proxy_eval_mode="shadow",
    )
    assert manifest.activation_verdict == "ready_to_spend"
    assert manifest.activation_status == "ready"
    assert validate_proxy_metric_calibration_manifest(manifest.to_dict()) == []


def test_forbidden_feature_makes_contract_unsafe(
    all_gates_available: tuple[ActivationGate, ...],
    example_arms: tuple[CalibrationArm, ...],
) -> None:
    unsafe = ProxyFeatureSet(
        feature_schema_version="proxy_features/v1",
        feature_names=("parser_valid", "agentv_score"),
        target_primary="binding_aware_meaningful_program_rate",
        target_gate="full_gate_pass",
        allowed_sources=("parser",),
        forbidden_features=("agentv_score",),
    )
    manifest = build_proxy_metric_calibration_manifest(
        manifest_id="sde3-03-v1",
        feature_set=unsafe,
        budget=BudgetCap(max_historical_rows=5000),
        arms=example_arms,
        activation_gates=all_gates_available,
    )
    assert manifest.activation_verdict == "feature_contract_unsafe"
    assert manifest.activation_status == "blocked"


def test_budget_blocked_path(
    example_feature_set: ProxyFeatureSet,
    all_gates_available: tuple[ActivationGate, ...],
    example_arms: tuple[CalibrationArm, ...],
) -> None:
    manifest = build_proxy_metric_calibration_manifest(
        manifest_id="sde3-03-v1",
        feature_set=example_feature_set,
        budget=BudgetCap(),
        arms=example_arms,
        activation_gates=all_gates_available,
    )
    assert manifest.activation_verdict == "budget_or_yield_blocked"
    assert manifest.activation_status == "blocked"


def test_no_eligible_arm_blocks(
    example_feature_set: ProxyFeatureSet,
    all_gates_available: tuple[ActivationGate, ...],
) -> None:
    omitted = tuple(
        CalibrationArm(
            arm_id=arm.arm_id,
            arm_kind=arm.arm_kind,
            eligible=False,
            omission_reason="diagnostic only",
        )
        for arm in DEFAULT_ARMS
    )
    manifest = build_proxy_metric_calibration_manifest(
        manifest_id="sde3-03-v1",
        feature_set=example_feature_set,
        budget=BudgetCap(max_historical_rows=5000),
        arms=omitted,
        activation_gates=all_gates_available,
    )
    assert manifest.activation_verdict == "activation_blocked"


def test_manifest_round_trip_dict(
    example_feature_set: ProxyFeatureSet,
    all_gates_available: tuple[ActivationGate, ...],
    example_arms: tuple[CalibrationArm, ...],
) -> None:
    manifest = build_proxy_metric_calibration_manifest(
        manifest_id="sde3-03-v1",
        feature_set=example_feature_set,
        budget=BudgetCap(max_historical_rows=5000, total_dollars=100.0),
        arms=example_arms,
        activation_gates=all_gates_available,
        conservative_floor=0.75,
        risk_budget=0.02,
        proxy_eval_mode="triage",
        audit_rate=0.20,
        force_full_every_n=10,
        note="round trip",
    )
    data = manifest.to_dict()
    restored = ProxyMetricCalibrationManifest.from_dict(data)
    assert restored.to_dict() == data
    assert validate_proxy_metric_calibration_manifest(data) == []


def test_default_arms_have_expected_eligibility() -> None:
    eligible = [a for a in DEFAULT_ARMS if a.eligible]
    omitted = [a for a in DEFAULT_ARMS if not a.eligible]
    assert len(DEFAULT_ARMS) == 4
    assert len(eligible) == 3
    assert len(omitted) == 1
    for arm in omitted:
        assert arm.omission_reason is not None and arm.omission_reason != ""


def test_validate_manifest_catches_errors() -> None:
    assert validate_proxy_metric_calibration_manifest({}) != []
    assert validate_proxy_metric_calibration_manifest(
        {"schema_version": "wrong"}
    ) != []

    manifest = build_proxy_metric_calibration_manifest(manifest_id="sde3-03-v1")
    data = manifest.to_dict()
    data["activation_verdict"] = "nonsense"
    errors = validate_proxy_metric_calibration_manifest(data)
    assert any("activation_verdict" in e for e in errors)

    data = manifest.to_dict()
    data["arms"][0] = {
        "arm_id": "bad",
        "arm_kind": "not_a_kind",
        "eligible": True,
    }
    errors = validate_proxy_metric_calibration_manifest(data)
    assert any("arm_kind" in e for e in errors)

    data = manifest.to_dict()
    data["arms"][0] = {
        "arm_id": "bad",
        "arm_kind": "rule_baseline",
        "eligible": False,
    }
    errors = validate_proxy_metric_calibration_manifest(data)
    assert any("omission_reason" in e for e in errors)

    data = manifest.to_dict()
    data["feature_set"]["feature_names"].append("agentv_score")
    errors = validate_proxy_metric_calibration_manifest(data)
    assert any("forbidden" in e for e in errors)


def test_validate_budget_requires_at_least_one_cap() -> None:
    manifest = build_proxy_metric_calibration_manifest(
        manifest_id="sde3-03-v1",
        budget=BudgetCap(),
    )
    errors = validate_proxy_metric_calibration_manifest(manifest.to_dict())
    assert any("budget must have at least one cap" in e for e in errors)


def test_verdict_sets_are_frozen() -> None:
    assert "ready_to_spend" in ACTIVATION_VERDICTS
    assert "unrun" in CAMPAIGN_VERDICTS


def test_cli_builds_manifest(tmp_path: Path) -> None:
    from scripts.build_proxy_metric_calibration_manifest import main

    out_json = tmp_path / "manifest.json"
    rc = main(["--out-json", str(out_json)])
    assert rc == 0
    assert out_json.is_file()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "version_stamp" in payload
    manifest = payload["manifest"]
    assert manifest["schema_version"] == MANIFEST_SCHEMA
    assert manifest["manifest_hash"] is not None
    assert len(manifest["activation_gates"]) == 5
    assert validate_proxy_metric_calibration_manifest(manifest) == []


def test_cli_builds_ready_manifest_with_markdown(tmp_path: Path) -> None:
    from scripts.build_proxy_metric_calibration_manifest import main

    out_json = tmp_path / "manifest.json"
    out_md = tmp_path / "manifest.md"
    gate_ids = [g.gate_id for g in DEFAULT_ACTIVATION_GATES]
    rc = main(
        [
            "--manifest-id",
            "sde3-03-v1",
            "--max-historical-rows",
            "10000",
            "--total-dollars",
            "500",
            "--proxy-eval-mode",
            "shadow",
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
    assert "# SDE3-03 proxy-metric calibration" in out_md.read_text(encoding="utf-8")
