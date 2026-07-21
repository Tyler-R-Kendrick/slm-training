"""Tests for the SLM-233 (DTL0-01) distillation trace-selection eval-holdout
leakage gate stress test."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm233_distill_select_eval_leakage_gate import (
    EXPERIMENT_ID,
    MATRIX_SET,
    DistillSelectEvalLeakageGateReport,
    LeakageArm,
    build_default_arms,
    render_markdown,
    run_eval_leakage_gate_stress_fixture,
)


def _result(report, name):
    return next(r for r in report.results if r.arm.name == name)


def test_default_arms_shape() -> None:
    arms = build_default_arms()
    names = {a.name for a in arms}
    assert names == {
        "train_prompts_control",
        "held_out_suite_traces",
        "adversarial_suite_traces",
        "ood_suite_traces",
        "smoke_suite_traces",
    }
    controls = {a.name for a in arms if a.is_negative_control}
    assert controls == {"train_prompts_control"}


def test_fixture_runs_all_arms() -> None:
    report = run_eval_leakage_gate_stress_fixture()
    assert report.matrix_set == MATRIX_SET
    assert report.experiment_id == EXPERIMENT_ID
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert len(report.results) == 5
    assert report.gate_hash


def test_control_arm_selected_fully() -> None:
    report = run_eval_leakage_gate_stress_fixture()
    control = _result(report, "train_prompts_control")
    assert control.n_traces == control.n_selected
    assert control.selection_rate == 1.0
    assert control.gameable is False


def test_eval_suite_arms_are_selected_at_control_rate() -> None:
    report = run_eval_leakage_gate_stress_fixture()
    for name in (
        "held_out_suite_traces",
        "adversarial_suite_traces",
        "ood_suite_traces",
        "smoke_suite_traces",
    ):
        result = _result(report, name)
        assert result.n_traces == result.n_selected, f"{name} traces should all be selected"
        assert result.selection_rate == report.control_selection_rate
        assert result.gameable is True
        assert not result.arm.is_negative_control


def test_static_audit_finds_no_source_suite_reference() -> None:
    report = run_eval_leakage_gate_stress_fixture()
    assert report.static_source_suite_audit
    assert all(v is False for v in report.static_source_suite_audit.values())


def test_disposition_confirms_the_gap() -> None:
    report = run_eval_leakage_gate_stress_fixture()
    assert report.disposition == "gap_confirmed"
    assert "gap" in report.disposition_rationale.lower() or "never train" in report.disposition_rationale.lower()


def test_report_roundtrips_through_dict() -> None:
    report = run_eval_leakage_gate_stress_fixture()
    payload = report.to_dict()
    restored = DistillSelectEvalLeakageGateReport.from_dict(payload)
    assert restored.to_dict() == payload


def test_gate_hash_is_deterministic() -> None:
    a = run_eval_leakage_gate_stress_fixture()
    b = run_eval_leakage_gate_stress_fixture()
    assert a.gate_hash == b.gate_hash
    assert a.disposition == b.disposition


def test_custom_arm_list_is_respected() -> None:
    arms = [
        LeakageArm(
            name="only_arm",
            description="d",
            records_source="held_out",
            source_suite="held_out",
        )
    ]
    report = run_eval_leakage_gate_stress_fixture(arms=arms)
    assert len(report.results) == 1
    assert report.results[0].arm.name == "only_arm"


def test_render_markdown_includes_disposition_and_table() -> None:
    report = run_eval_leakage_gate_stress_fixture()
    text = render_markdown(report)
    assert report.disposition in text
    assert "| arm | source_suite |" in text
    assert "No-go for promotion" in text
    assert "held_out_suite_traces" in text
