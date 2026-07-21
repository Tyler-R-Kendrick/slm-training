"""Tests for the SLM-232 (PEL0-01) preference build-pairs eval-holdout
leakage gate stress test."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm232_preference_eval_leakage_gate import (
    EXPERIMENT_ID,
    MATRIX_SET,
    LeakageArm,
    PreferenceEvalLeakageGateReport,
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
        "train_seeds_control",
        "held_out_as_train_records",
        "adversarial_as_train_records",
        "ood_as_train_records",
        "smoke_as_train_records",
    }
    controls = {a.name for a in arms if a.is_negative_control}
    assert controls == {"train_seeds_control"}


def test_fixture_runs_all_arms() -> None:
    report = run_eval_leakage_gate_stress_fixture()
    assert report.matrix_set == MATRIX_SET
    assert report.experiment_id == EXPERIMENT_ID
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert len(report.results) == 5
    assert report.gate_hash


def test_control_arm_builds_and_does_not_match_held_out() -> None:
    report = run_eval_leakage_gate_stress_fixture()
    control = _result(report, "train_seeds_control")
    assert control.build_pairs_succeeded is True
    assert control.pairs_written == control.n_records
    assert control.match_rate_against_source_suite_fingerprints < 0.5
    assert control.gameable is False


def test_eval_suite_arms_are_accepted_at_full_fingerprint_overlap() -> None:
    report = run_eval_leakage_gate_stress_fixture()
    for name in (
        "held_out_as_train_records",
        "adversarial_as_train_records",
        "ood_as_train_records",
        "smoke_as_train_records",
    ):
        result = _result(report, name)
        assert result.build_pairs_succeeded is True, f"{name} should build pairs successfully"
        assert result.build_pairs_error is None
        assert result.pairs_written == result.n_records
        assert result.match_rate_against_source_suite_fingerprints == 1.0
        assert result.gameable is True
        assert not result.arm.is_negative_control


def test_static_audit_finds_no_leakage_module_reference() -> None:
    report = run_eval_leakage_gate_stress_fixture()
    assert report.static_leakage_import_audit
    assert all(v is False for v in report.static_leakage_import_audit.values())


def test_disposition_confirms_the_gap() -> None:
    report = run_eval_leakage_gate_stress_fixture()
    assert report.disposition == "gap_confirmed"
    assert "gap" in report.disposition_rationale.lower() or "never train" in report.disposition_rationale.lower()


def test_report_roundtrips_through_dict() -> None:
    report = run_eval_leakage_gate_stress_fixture()
    payload = report.to_dict()
    restored = PreferenceEvalLeakageGateReport.from_dict(payload)
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
        )
    ]
    report = run_eval_leakage_gate_stress_fixture(arms=arms)
    assert len(report.results) == 1
    assert report.results[0].arm.name == "only_arm"


def test_render_markdown_includes_disposition_and_table() -> None:
    report = run_eval_leakage_gate_stress_fixture()
    text = render_markdown(report)
    assert report.disposition in text
    assert "| arm | n records |" in text
    assert "No-go for promotion" in text
    assert "held_out_as_train_records" in text
