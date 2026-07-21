"""Tests for the SLM-228 (RLRG0-01) RL-readiness reward-variance gate stress test."""

from __future__ import annotations

from slm_training.harnesses.experiments.slm228_rl_readiness_variance_gate import (
    CANDIDATE_MIN_SAMPLES,
    CANDIDATE_MIN_SPREAD,
    EXPERIMENT_ID,
    MATRIX_SET,
    RewardVarianceArm,
    RlReadinessVarianceGateReport,
    build_default_arms,
    render_markdown,
    run_variance_gate_stress_fixture,
)


def _result(report, name):
    return next(r for r in report.results if r.arm.name == name)


def test_default_arms_shape() -> None:
    arms = build_default_arms()
    names = {a.name for a in arms}
    assert names == {
        "healthy_diverse_n8",
        "two_sample_wide",
        "two_sample_epsilon",
        "large_n_epsilon_outlier",
        "all_identical_control",
        "single_sample_control",
    }
    controls = {a.name for a in arms if a.is_negative_control}
    assert controls == {"all_identical_control", "single_sample_control"}


def test_fixture_runs_all_arms() -> None:
    report = run_variance_gate_stress_fixture()
    assert report.matrix_set == MATRIX_SET
    assert report.experiment_id == EXPERIMENT_ID
    assert report.status == "fixture"
    assert report.claim_class == "wiring"
    assert len(report.results) == 6
    assert report.gate_hash


def test_negative_controls_are_rejected_by_the_real_gate() -> None:
    report = run_variance_gate_stress_fixture()
    zero_var = _result(report, "all_identical_control")
    single = _result(report, "single_sample_control")
    assert zero_var.approved is False
    assert zero_var.reward_variance == 0.0
    assert any("variance" in f for f in zero_var.failures)
    assert single.approved is False
    assert any("variance" in f for f in single.failures)


def test_healthy_arm_passes_both_real_gate_and_candidate_check() -> None:
    report = run_variance_gate_stress_fixture()
    healthy = _result(report, "healthy_diverse_n8")
    assert healthy.approved is True
    assert healthy.candidate_would_pass is True
    assert healthy.gameable is False
    assert healthy.n_samples == 8
    assert healthy.spread is not None and healthy.spread > CANDIDATE_MIN_SPREAD


def test_degenerate_arms_are_approved_by_real_gate_but_fail_candidate() -> None:
    report = run_variance_gate_stress_fixture()
    for name in (
        "two_sample_wide",
        "two_sample_epsilon",
        "large_n_epsilon_outlier",
    ):
        result = _result(report, name)
        assert result.approved is True, f"{name} should pass the real (mechanical) gate"
        assert result.reward_variance > 0.0
        assert not result.arm.is_negative_control

    epsilon_arm = _result(report, "two_sample_epsilon")
    assert epsilon_arm.candidate_would_pass is False
    assert epsilon_arm.gameable is True

    outlier_arm = _result(report, "large_n_epsilon_outlier")
    assert outlier_arm.n_samples > CANDIDATE_MIN_SAMPLES
    assert outlier_arm.candidate_would_pass is False
    assert outlier_arm.gameable is True

    # Wide two-sample arm fails the candidate purely on sample count, even
    # though its spread is genuinely large.
    wide_arm = _result(report, "two_sample_wide")
    assert wide_arm.spread is not None and wide_arm.spread >= CANDIDATE_MIN_SPREAD
    assert wide_arm.n_samples < CANDIDATE_MIN_SAMPLES
    assert wide_arm.candidate_would_pass is False
    assert wide_arm.gameable is True


def test_disposition_confirms_the_gap() -> None:
    report = run_variance_gate_stress_fixture()
    assert report.disposition == "gap_confirmed"
    assert "gameable" in report.disposition_rationale.lower()


def test_report_roundtrips_through_dict() -> None:
    report = run_variance_gate_stress_fixture()
    payload = report.to_dict()
    restored = RlReadinessVarianceGateReport.from_dict(payload)
    assert restored.to_dict() == payload


def test_gate_hash_is_deterministic() -> None:
    a = run_variance_gate_stress_fixture()
    b = run_variance_gate_stress_fixture()
    assert a.gate_hash == b.gate_hash
    assert a.disposition == b.disposition


def test_custom_arm_list_is_respected() -> None:
    arms = [RewardVarianceArm(name="only_arm", description="d", reward_samples=(0.1, 0.2))]
    report = run_variance_gate_stress_fixture(arms=arms)
    assert len(report.results) == 1
    assert report.results[0].arm.name == "only_arm"


def test_render_markdown_includes_disposition_and_table() -> None:
    report = run_variance_gate_stress_fixture()
    text = render_markdown(report)
    assert report.disposition in text
    assert "| arm | n | spread | reward_variance |" in text
    assert "No-go for promotion" in text
    assert "healthy_diverse_n8" in text
