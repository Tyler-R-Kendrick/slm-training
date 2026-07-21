"""Tests for the SLM-229 (RLRG0-02) RL-readiness declared-vs-actual suite-size
gate stress test."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm229_rl_readiness_declared_suite_size_gate import (
    EXPERIMENT_ID,
    MATRIX_SET,
    REQUIRED_RICO_HELD_N,
    RlReadinessSuiteSizeGateReport,
    SuiteSizeArm,
    build_default_arms,
    render_markdown,
    run_suite_size_gate_stress_fixture,
)


def _result(report, name):
    return next(r for r in report.results if r.arm.name == name)


def test_default_arms_shape() -> None:
    arms = build_default_arms()
    names = {a.name for a in arms}
    assert names == {
        "matched_actual_1500",
        "declared_only_ship_gate_floor_n20",
        "declared_only_smoke_scale_n25",
        "declared_far_exceeds_actual_n100",
        "no_declared_field_small_actual_control",
        "declared_below_floor_control",
    }
    controls = {a.name for a in arms if a.is_negative_control}
    assert controls == {
        "no_declared_field_small_actual_control",
        "declared_below_floor_control",
    }


def test_fixture_runs_all_arms() -> None:
    report = run_suite_size_gate_stress_fixture()
    assert report.matrix_set == MATRIX_SET
    assert report.experiment_id == EXPERIMENT_ID
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert len(report.results) == 6
    assert report.gate_hash


def test_negative_controls_are_rejected_by_the_real_gate() -> None:
    report = run_suite_size_gate_stress_fixture()
    no_declared = _result(report, "no_declared_field_small_actual_control")
    below_floor = _result(report, "declared_below_floor_control")
    assert no_declared.approved is False
    assert any("rico_held" in f for f in no_declared.failures)
    assert no_declared.assert_rl_ready_raised is True
    assert below_floor.approved is False
    assert any("rico_held" in f for f in below_floor.failures)
    assert below_floor.assert_rl_ready_raised is True


def test_healthy_arm_passes_both_real_gate_and_candidate_check() -> None:
    report = run_suite_size_gate_stress_fixture()
    healthy = _result(report, "matched_actual_1500")
    assert healthy.approved is True
    assert healthy.assert_rl_ready_raised is False
    assert healthy.candidate_would_pass is True
    assert healthy.gameable is False
    assert healthy.reported_rico_held_n == REQUIRED_RICO_HELD_N


def test_declared_only_arms_are_approved_by_real_gate_but_fail_candidate() -> None:
    report = run_suite_size_gate_stress_fixture()
    for name in (
        "declared_only_ship_gate_floor_n20",
        "declared_only_smoke_scale_n25",
        "declared_far_exceeds_actual_n100",
    ):
        result = _result(report, name)
        assert result.approved is True, f"{name} should pass the real (mechanical) gate"
        assert result.assert_rl_ready_raised is False, (
            f"{name} should also pass the downstream fail-closed assert_rl_ready"
        )
        assert result.reported_rico_held_n >= REQUIRED_RICO_HELD_N
        assert not result.arm.is_negative_control
        assert result.candidate_would_pass is False
        assert result.gameable is True

    floor_arm = _result(report, "declared_only_ship_gate_floor_n20")
    assert floor_arm.arm.actual_n == 20

    over_claim_arm = _result(report, "declared_far_exceeds_actual_n100")
    assert over_claim_arm.reported_rico_held_n == 5000


def test_disposition_confirms_the_gap() -> None:
    report = run_suite_size_gate_stress_fixture()
    assert report.disposition == "gap_confirmed"
    assert "gameable" in report.disposition_rationale.lower()


def test_report_roundtrips_through_dict() -> None:
    report = run_suite_size_gate_stress_fixture()
    payload = report.to_dict()
    restored = RlReadinessSuiteSizeGateReport.from_dict(payload)
    assert restored.to_dict() == payload


def test_gate_hash_is_deterministic() -> None:
    a = run_suite_size_gate_stress_fixture()
    b = run_suite_size_gate_stress_fixture()
    assert a.gate_hash == b.gate_hash
    assert a.disposition == b.disposition


def test_custom_arm_list_is_respected() -> None:
    arms = [
        SuiteSizeArm(
            name="only_arm",
            description="d",
            actual_n=1500,
            declared_n=1500,
        )
    ]
    report = run_suite_size_gate_stress_fixture(arms=arms)
    assert len(report.results) == 1
    assert report.results[0].arm.name == "only_arm"


def test_render_markdown_includes_disposition_and_table() -> None:
    report = run_suite_size_gate_stress_fixture()
    text = render_markdown(report)
    assert report.disposition in text
    assert "| arm | actual n | declared n |" in text
    assert "No-go for promotion" in text
    assert "matched_actual_1500" in text
