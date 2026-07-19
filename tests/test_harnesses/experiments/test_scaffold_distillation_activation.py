"""Regression tests for SDE4-01 scaffold-distillation activation manifest (SLM-179)."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.harnesses.experiments.scaffold_distillation_activation import (
    DEFAULT_ACTIVATION_GATES,
    DEFAULT_ARMS,
    HYPOTHESIS_ID,
    MANIFEST_SCHEMA,
    ActivationGate,
    BudgetCap,
    ScaffoldDistillationArm,
    ScaffoldDistillationActivationManifest,
    TeacherTraceContract,
    build_scaffold_distillation_activation_manifest,
    validate_scaffold_distillation_activation_manifest,
)


def _all_available_gates() -> tuple[ActivationGate, ...]:
    return tuple(
        ActivationGate(
            gate_id=g.gate_id,
            depends_on_issue_id=g.depends_on_issue_id,
            required_status=g.required_status,
            available=True,
            evidence=g.evidence,
        )
        for g in DEFAULT_ACTIVATION_GATES
    )


def _base_trace_contract() -> TeacherTraceContract:
    return TeacherTraceContract(
        teacher_checkpoint_id="teacher/checkpoint",
        teacher_run_id="teacher/run",
        trace_store_uri="memory://traces",
        trace_schema_version="v1",
        min_traces=0,
        max_traces=0,
        scaffold_config_hash="unknown",
    )


def _base_budget(total: float | None = 1000.0) -> BudgetCap:
    return BudgetCap(total_dollars=total)


def test_ready_path() -> None:
    manifest = build_scaffold_distillation_activation_manifest(
        manifest_id="m1",
        activation_gates=_all_available_gates(),
        teacher_trace_contract=_base_trace_contract(),
        budget=_base_budget(),
        arms=DEFAULT_ARMS,
        scaffold_decomposition="value_demonstrated",
    )
    assert manifest.activation_status == "ready"
    assert manifest.activation_verdict == "ready_to_spend"
    assert manifest.hypothesis_id == HYPOTHESIS_ID
    assert manifest.schema_version == MANIFEST_SCHEMA


def test_blocked_gate_path() -> None:
    manifest = build_scaffold_distillation_activation_manifest(
        manifest_id="m1",
        activation_gates=DEFAULT_ACTIVATION_GATES,
        teacher_trace_contract=_base_trace_contract(),
        budget=_base_budget(),
        arms=DEFAULT_ARMS,
        scaffold_decomposition="value_demonstrated",
    )
    assert manifest.activation_status == "blocked"
    assert manifest.activation_verdict == "activation_blocked"


def test_no_scaffold_value_path() -> None:
    manifest = build_scaffold_distillation_activation_manifest(
        manifest_id="m1",
        activation_gates=_all_available_gates(),
        teacher_trace_contract=_base_trace_contract(),
        budget=_base_budget(),
        arms=DEFAULT_ARMS,
        scaffold_decomposition="no_value",
    )
    assert manifest.activation_status == "closed"
    assert manifest.activation_verdict == "no_scaffold_value"


def test_budget_blocked_path() -> None:
    manifest = build_scaffold_distillation_activation_manifest(
        manifest_id="m1",
        activation_gates=_all_available_gates(),
        teacher_trace_contract=_base_trace_contract(),
        budget=_base_budget(total=0.0),
        arms=DEFAULT_ARMS,
        scaffold_decomposition="value_demonstrated",
    )
    assert manifest.activation_status == "blocked"
    assert manifest.activation_verdict == "budget_or_yield_blocked"


def test_inventory_information_blocked_path() -> None:
    gates = tuple(
        ActivationGate(
            gate_id=g.gate_id,
            depends_on_issue_id=g.depends_on_issue_id,
            required_status=g.required_status,
            available=(g.gate_id != "slm168_public_structured_contract_pointer"),
            evidence=g.evidence,
        )
        for g in _all_available_gates()
    )
    # Remove SLM-168 so the specific inventory-information block is reached
    # rather than the generic activation_blocked branch.
    gates_without_slm168 = tuple(
        g for g in gates if g.gate_id != "slm168_public_structured_contract_pointer"
    )
    manifest = build_scaffold_distillation_activation_manifest(
        manifest_id="m1",
        activation_gates=gates_without_slm168,
        teacher_trace_contract=_base_trace_contract(),
        budget=_base_budget(),
        arms=DEFAULT_ARMS,
        scaffold_decomposition="inventory_required",
    )
    assert manifest.activation_status == "blocked"
    assert manifest.activation_verdict == "inventory_information_blocked"


def test_manifest_round_trip_dict() -> None:
    manifest = build_scaffold_distillation_activation_manifest(
        manifest_id="m1",
        activation_gates=_all_available_gates(),
        teacher_trace_contract=_base_trace_contract(),
        budget=_base_budget(),
        arms=DEFAULT_ARMS,
        scaffold_decomposition="value_demonstrated",
        primary_metric="component_recall",
        seeds=(7,),
        max_attempts_for_teacher=3,
        note="round trip",
    )
    data = manifest.to_dict()
    restored = ScaffoldDistillationActivationManifest.from_dict(data)
    assert restored == manifest
    assert validate_scaffold_distillation_activation_manifest(data) == []


def test_validator_catches_invalid_arm_kind() -> None:
    data = build_scaffold_distillation_activation_manifest(
        manifest_id="m1",
        activation_gates=_all_available_gates(),
        teacher_trace_contract=_base_trace_contract(),
        budget=_base_budget(),
        arms=[
            ScaffoldDistillationArm(
                arm_id="bad_arm",
                arm_kind="not_a_real_kind",  # type: ignore[arg-type]
                eligible=True,
            )
        ],
        scaffold_decomposition="value_demonstrated",
    ).to_dict()
    errors = validate_scaffold_distillation_activation_manifest(data)
    assert any("arm_kind" in e for e in errors)


def test_validator_catches_omitted_arm_missing_reason() -> None:
    data = build_scaffold_distillation_activation_manifest(
        manifest_id="m1",
        activation_gates=_all_available_gates(),
        teacher_trace_contract=_base_trace_contract(),
        budget=_base_budget(),
        arms=[
            ScaffoldDistillationArm(
                arm_id="omitted",
                arm_kind="scaffolded_teacher_selected",
                eligible=False,
            )
        ],
        scaffold_decomposition="value_demonstrated",
    ).to_dict()
    errors = validate_scaffold_distillation_activation_manifest(data)
    assert any("omission_reason" in e for e in errors)


def test_validator_catches_empty_budget() -> None:
    data = build_scaffold_distillation_activation_manifest(
        manifest_id="m1",
        activation_gates=_all_available_gates(),
        teacher_trace_contract=_base_trace_contract(),
        budget=BudgetCap(),
        arms=DEFAULT_ARMS,
        scaffold_decomposition="value_demonstrated",
    ).to_dict()
    errors = validate_scaffold_distillation_activation_manifest(data)
    assert any("budget" in e.lower() for e in errors)


def test_cli_builds_manifest(tmp_path: Path) -> None:
    from scripts.build_scaffold_distillation_activation_manifest import main

    out_json = tmp_path / "manifest.json"
    out_md = tmp_path / "manifest.md"
    rc = main(
        [
            "--manifest-id",
            "cli-test",
            "--gate-slm-161",
            "--gate-slm-162",
            "--gate-slm-168",
            "--gate-scaffold-value",
            "--gate-latency",
            "--gate-budget",
            "--total-dollars",
            "500",
            "--scaffold-decomposition",
            "value_demonstrated",
            "--primary-metric",
            "component_recall",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ]
    )
    assert rc == 0
    assert out_json.is_file()
    assert out_md.is_file()
    manifest = json.loads(out_json.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == MANIFEST_SCHEMA
    assert manifest["hypothesis_id"] == HYPOTHESIS_ID
    assert manifest["manifest_id"] == "cli-test"
    assert manifest["primary_metric"] == "component_recall"
    assert manifest["activation_status"] == "ready"
    assert manifest["activation_verdict"] == "ready_to_spend"
    assert manifest["manifest_hash"] is not None
    assert len(manifest["arms"]) == 8
    assert "version_stamp" in manifest
    assert validate_scaffold_distillation_activation_manifest(manifest) == []
