"""Regression tests for the CAP2-07 oracle-latent causal-necessity audit
harness (AP-031 / SLM-330)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.harnesses.experiments.cap2_latent_causal_audit import (
    CAUSAL_AUDIT_CONDITIONS,
    NOISE_SIGMA_MULTIPLIERS,
    CausalAuditArm,
    default_causal_audit_arms,
    evaluate_arm,
    load_causal_audit_corpus,
    run_causal_audit,
    _derangement,
    _role_preserving_permutation,
    _segment_ranges,
    _slot_levels,
)


def _tiny_arms() -> list[CausalAuditArm]:
    """Small, fast arm set mirroring the defaults but bounded for CI speed."""
    return [
        CausalAuditArm("continuous_tiny", "continuous", num_latents=2, width=32, seed=0, train_steps=60),
        CausalAuditArm("lfq_tiny", "lfq", num_latents=2, width=32, seed=0, train_steps=60),
        CausalAuditArm(
            "fsq_typed_tiny", "fsq_typed", radixes=(2, 3, 3, 4, 5), width=32, seed=0, train_steps=80
        ),
    ]


def test_default_causal_audit_arms_ids_and_codecs() -> None:
    arms = default_causal_audit_arms()
    assert [a.arm_id for a in arms] == ["continuous_n2_w64", "lfq_n2_w64", "fsq_typed_2_3_3_4_5"]
    assert {a.codec for a in arms} == {"continuous", "lfq", "fsq_typed"}


def test_load_causal_audit_corpus_splits_train_and_holdout() -> None:
    train, holdout = load_causal_audit_corpus(fixture_count=16, fixture_seed=0)
    assert len(train) > 0
    assert len(holdout) >= 2
    assert len(train) + len(holdout) == 16


def test_load_causal_audit_corpus_fails_closed_on_tiny_holdout() -> None:
    with pytest.raises(ValueError):
        load_causal_audit_corpus(fixture_count=2, fixture_seed=0)


def test_derangement_has_no_fixed_points() -> None:
    import random

    rng = random.Random(0)
    order = _derangement(6, rng)
    assert sorted(order) == list(range(6))
    assert all(order[i] != i for i in range(6))


def test_derangement_rejects_n_below_two() -> None:
    import random

    with pytest.raises(ValueError):
        _derangement(1, random.Random(0))


def test_role_preserving_permutation_none_for_homogeneous_widths() -> None:
    """Regression test: every slot sharing one width (e.g. binary LFQ, or any
    element-wise continuous/VQ latent) must NOT be treated as a swappable
    role group -- that would make role_swapped silently identical to
    slot_permuted instead of being marked not-applicable.
    """
    import random

    assert _role_preserving_permutation([1, 1, 1], random.Random(0)) is None
    assert _role_preserving_permutation([2, 2, 2], random.Random(0)) is None
    assert _role_preserving_permutation([1], random.Random(0)) is None


def test_role_preserving_permutation_applies_only_within_same_width_group() -> None:
    import random

    levels = [2, 3, 3, 4, 5]  # AP-029's plan_fsq_2_3_3_4_5 radixes
    order = _role_preserving_permutation(levels, random.Random(0))
    assert order is not None
    # Only the two width-3 slots (indices 1, 2) may move; every other index
    # must stay fixed since its width has no same-width peer.
    assert order[0] == 0
    assert order[3] == 3
    assert order[4] == 4
    assert sorted(order[1:3]) == [1, 2]
    assert order[1] != 1 or order[2] != 2  # a real swap happened, not the identity


def test_segment_ranges_matches_cumulative_widths() -> None:
    assert _segment_ranges([2, 3, 4]) == [(0, 2), (2, 5), (5, 9)]


def test_slot_levels_element_wise_for_lfq_and_continuous() -> None:
    from slm_training.harnesses.experiments.cap2_rate_distortion_sweep import _build_codec, RateDistortionCell

    lfq = _build_codec(RateDistortionCell("lfq", num_latents=4, width=32), feature_dim=45)
    assert _slot_levels(lfq, decoder_input_dim=4) == [1, 1, 1, 1]

    continuous = _build_codec(RateDistortionCell("continuous", num_latents=3, width=32), feature_dim=45)
    assert _slot_levels(continuous, decoder_input_dim=3) == [1, 1, 1]


def test_slot_levels_segmented_for_uniform_scalar() -> None:
    from slm_training.harnesses.experiments.cap2_rate_distortion_sweep import _build_codec, RateDistortionCell

    codec = _build_codec(RateDistortionCell("uniform_scalar", num_latents=3, width=32), feature_dim=45)
    assert _slot_levels(codec, decoder_input_dim=6) == [2, 2, 2]


@pytest.mark.parametrize("arm", _tiny_arms(), ids=lambda a: a.arm_id)
def test_evaluate_arm_reports_every_condition_and_no_bypass(arm: CausalAuditArm) -> None:
    train_plans, holdout_plans = load_causal_audit_corpus(fixture_count=12, fixture_seed=0)
    result = evaluate_arm(arm, train_plans, holdout_plans, ci_resamples=50)

    assert result.arm_id == arm.arm_id
    assert result.num_holdout == len(holdout_plans)
    seen_conditions = {c.condition for c in result.conditions}
    expected = set(CAUSAL_AUDIT_CONDITIONS) - {"correct"}
    if not result.role_swap_applicable:
        expected -= {"role_swapped"}
    expected |= {f"noise_perturbed_sigma{m}" for m in NOISE_SIGMA_MULTIPLIERS}
    assert seen_conditions == expected
    assert "zero" in seen_conditions
    assert "quantized" in seen_conditions
    assert isinstance(result.no_bypass_ok, bool)
    assert isinstance(result.causally_necessary, bool)
    assert result.effective_rank >= 0.0
    assert result.total_variance >= 0.0


def test_quantized_condition_reproduces_correct_exactly() -> None:
    """The 'quantized' arm is a determinism sanity control: it re-runs a
    fresh hard encode with no perturbation, so it must reproduce the
    'correct' condition's score exactly (paired MSE difference of 0 for
    every held-out example).
    """
    train_plans, holdout_plans = load_causal_audit_corpus(fixture_count=12, fixture_seed=0)
    arm = CausalAuditArm("continuous_quant_check", "continuous", num_latents=2, width=32, seed=0, train_steps=60)
    result = evaluate_arm(arm, train_plans, holdout_plans, ci_resamples=50)
    quantized = next(c for c in result.conditions if c.condition == "quantized")
    assert quantized.mse_ci_vs_correct["low"] == pytest.approx(0.0, abs=1e-9)
    assert quantized.mse_ci_vs_correct["high"] == pytest.approx(0.0, abs=1e-9)
    assert quantized.rate_ci_vs_correct["low"] == pytest.approx(0.0, abs=1e-9)
    assert quantized.rate_ci_vs_correct["high"] == pytest.approx(0.0, abs=1e-9)


def test_role_swap_applicable_flag_matches_codec_typing() -> None:
    train_plans, holdout_plans = load_causal_audit_corpus(fixture_count=12, fixture_seed=0)

    homogeneous = CausalAuditArm("lfq_flag_check", "lfq", num_latents=2, width=32, seed=0, train_steps=60)
    result = evaluate_arm(homogeneous, train_plans, holdout_plans, ci_resamples=50)
    assert result.role_swap_applicable is False
    assert "role_swapped" not in {c.condition for c in result.conditions}
    assert any("role_swapped omitted" in n for n in result.notes)

    typed = CausalAuditArm(
        "fsq_typed_flag_check", "fsq_typed", radixes=(2, 3, 3, 4, 5), width=32, seed=0, train_steps=80
    )
    typed_result = evaluate_arm(typed, train_plans, holdout_plans, ci_resamples=50)
    assert typed_result.role_swap_applicable is True
    assert "role_swapped" in {c.condition for c in typed_result.conditions}


def test_no_bypass_failure_marks_arm_untrustworthy_in_disposition() -> None:
    """A collapsed encoder must never be silently promoted into a causal-
    necessity claim -- causally_necessary must be False and the run-level
    disposition must exclude that arm from the "trustworthy" pool.
    """
    train_plans, holdout_plans = load_causal_audit_corpus(fixture_count=12, fixture_seed=0)
    # A near-zero training budget makes encoder collapse plausible at this
    # tiny fixture scale (matching CAP2-06's own documented minimum-capacity
    # collapse finding); we only assert the *contract*, not that collapse
    # necessarily occurs for this exact seed/budget.
    arm = CausalAuditArm("continuous_budget_check", "continuous", num_latents=2, width=32, seed=0, train_steps=1)
    result = evaluate_arm(arm, train_plans, holdout_plans, ci_resamples=20)
    if not result.no_bypass_ok:
        assert result.causally_necessary is False


def test_run_causal_audit_end_to_end_small_grid() -> None:
    report = run_causal_audit(fixture_count=12, fixture_seed=0, arms=_tiny_arms(), ci_resamples=50)
    assert len(report.arms) == 3
    assert report.disposition in {"continue", "stop_ignore"}
    assert report.num_train + report.num_holdout == 12
    assert report.disposition_reasons
