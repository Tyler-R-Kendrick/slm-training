"""Regression tests for the SDE3-04 constraint-backend benchmark manifest."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.harnesses.experiments.constraint_backend_benchmark import (
    HYPOTHESIS_ID,
    MANIFEST_SCHEMA,
    ActivationGate,
    BackendAdapter,
    BenchmarkArm,
    BudgetCap,
    ConstraintBackendBenchmarkManifest,
    build_constraint_backend_benchmark_manifest,
    validate_constraint_backend_benchmark_manifest,
)


def test_ready_path() -> None:
    manifest = build_constraint_backend_benchmark_manifest(
        manifest_id="test-ready",
        activation_gates=(
            ActivationGate(gate_id="eval_cache_or_cost_approved", available=True),
            ActivationGate(gate_id="budget_approved", available=True),
        ),
        budget=BudgetCap(
            microbenchmark_repetitions=10,
            end_to_end_repetitions=3,
            max_dollars=50.0,
            gpu_hours=2.0,
        ),
    )
    assert manifest.activation_verdict == "ready_to_spend"
    assert manifest.activation_status == manifest.activation_verdict
    assert manifest.campaign_verdict == "unrun"
    assert validate_constraint_backend_benchmark_manifest(manifest.to_dict()) == []


def test_blocked_gate_path() -> None:
    manifest = build_constraint_backend_benchmark_manifest(
        manifest_id="test-blocked",
        activation_gates=(
            ActivationGate(gate_id="eval_cache_or_cost_approved", available=False),
            ActivationGate(gate_id="budget_approved", available=True),
        ),
        budget=BudgetCap(
            microbenchmark_repetitions=10,
            max_dollars=50.0,
        ),
    )
    assert manifest.activation_verdict == "activation_blocked"
    assert validate_constraint_backend_benchmark_manifest(manifest.to_dict()) == []


def test_budget_blocked_path() -> None:
    manifest = build_constraint_backend_benchmark_manifest(
        manifest_id="test-budget-blocked",
        activation_gates=(
            ActivationGate(gate_id="eval_cache_or_cost_approved", available=True),
            ActivationGate(gate_id="budget_approved", available=True),
        ),
        budget=BudgetCap(
            microbenchmark_repetitions=0,
            end_to_end_repetitions=0,
            max_dollars=0.0,
            gpu_hours=0.0,
        ),
    )
    assert manifest.activation_verdict == "budget_or_yield_blocked"
    assert validate_constraint_backend_benchmark_manifest(manifest.to_dict()) == []


def test_default_manifest_is_blocked_and_valid() -> None:
    manifest = build_constraint_backend_benchmark_manifest(
        manifest_id="test-default",
    )
    assert manifest.activation_verdict == "activation_blocked"
    assert manifest.hypothesis_id == HYPOTHESIS_ID
    assert manifest.schema_version == MANIFEST_SCHEMA
    assert len(manifest.backends) == 5
    assert len(manifest.arms) == 15
    assert validate_constraint_backend_benchmark_manifest(manifest.to_dict()) == []


def test_manifest_round_trip_dict() -> None:
    manifest = build_constraint_backend_benchmark_manifest(
        manifest_id="test-round-trip",
        activation_gates=(
            ActivationGate(gate_id="eval_cache_or_cost_approved", available=True),
            ActivationGate(gate_id="budget_approved", available=True),
        ),
        budget=BudgetCap(
            microbenchmark_repetitions=5,
            end_to_end_repetitions=1,
            max_dollars=10.0,
            gpu_hours=0.5,
        ),
        primary_metric="component_recall",
        seeds=(7, 8),
        null_threshold_percent=3.0,
        note="round trip",
    )
    data = manifest.to_dict()
    rebuilt = ConstraintBackendBenchmarkManifest.from_dict(data)
    assert isinstance(rebuilt, ConstraintBackendBenchmarkManifest)
    assert rebuilt.manifest_id == manifest.manifest_id
    assert rebuilt.schema_version == MANIFEST_SCHEMA
    assert rebuilt.primary_metric == "component_recall"
    assert rebuilt.seeds == (7, 8)
    assert rebuilt.null_threshold_percent == 3.0
    assert rebuilt.activation_verdict == "ready_to_spend"
    assert rebuilt.manifest_hash == manifest.manifest_hash
    assert validate_constraint_backend_benchmark_manifest(data) == []


def test_validator_catches_invalid_backend_id() -> None:
    manifest = build_constraint_backend_benchmark_manifest(
        manifest_id="test-invalid-backend",
    )
    data = manifest.to_dict()
    data["backends"][0]["backend_id"] = "not_a_backend"
    errors = validate_constraint_backend_benchmark_manifest(data)
    assert any("backend_id" in e for e in errors)


def test_validator_catches_omitted_arm_missing_reason() -> None:
    manifest = build_constraint_backend_benchmark_manifest(
        manifest_id="test-omitted-arm",
        arms=(
            BenchmarkArm(
                arm_id="current_static_micro",
                backend_id="current",
                benchmark_layer="static_micro",
                eligible=False,
            ),
        ),
    )
    errors = validate_constraint_backend_benchmark_manifest(manifest.to_dict())
    assert any("omission_reason" in e for e in errors)


def test_validator_catches_unregistered_arm_backend() -> None:
    manifest = build_constraint_backend_benchmark_manifest(
        manifest_id="test-unregistered-arm-backend",
        backends=(
            BackendAdapter(
                backend_id="current",
                package_name="openui_current",
                package_version="repo",
            ),
        ),
        arms=(
            BenchmarkArm(
                arm_id="syncode_static_micro",
                backend_id="syncode",
                benchmark_layer="static_micro",
                eligible=True,
            ),
        ),
    )
    errors = validate_constraint_backend_benchmark_manifest(manifest.to_dict())
    assert any("not declared in backends" in e for e in errors)


def test_cli_builds_manifest_file(tmp_path: Path) -> None:
    from scripts.build_constraint_backend_benchmark_manifest import main

    out_json = tmp_path / "manifest.json"
    out_md = tmp_path / "manifest.md"
    rc = main(
        [
            "--manifest-id",
            "cli-test",
            "--microbenchmark-repetitions",
            "5",
            "--end-to-end-repetitions",
            "1",
            "--max-dollars",
            "10.0",
            "--gpu-hours",
            "0.5",
            "--gate-eval-cache",
            "--gate-budget",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
            "--note",
            "cli test",
        ]
    )
    assert rc == 0
    assert out_json.is_file()
    assert out_md.is_file()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["manifest"]["schema_version"] == MANIFEST_SCHEMA
    assert payload["manifest"]["manifest_id"] == "cli-test"
    assert payload["manifest"]["activation_verdict"] == "ready_to_spend"
    assert payload["version_stamp"]["stamp_schema"] == "version_stamp/v1"
    assert validate_constraint_backend_benchmark_manifest(payload["manifest"]) == []
