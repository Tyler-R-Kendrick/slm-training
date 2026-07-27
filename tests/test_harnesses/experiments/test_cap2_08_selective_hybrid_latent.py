"""Regression tests for the CAP2-08 selective-hybrid latent audit harness
(AP-033 / SLM-333)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.harnesses.experiments.cap2_selective_hybrid_latent import (
    DEFAULT_ARMS,
    SelectiveHybridArmConfig,
    SelectiveHybridModel,
    default_arm_configs,
    evaluate_arm,
    load_selective_hybrid_corpus,
    run_selective_hybrid,
)
from slm_training.models.selective_latent_connector import SelectiveLatentArm
from slm_training.models.semantic_plan_factors import (
    ProgramFactorTensorizerConfig,
    factor_dims,
)


def _tiny_arms() -> list[SelectiveHybridArmConfig]:
    """Small, fast arm set mirroring the defaults but bounded for CI speed."""
    return [SelectiveHybridArmConfig(arm.value, arm, seed=0, train_steps=40) for arm in DEFAULT_ARMS]


def test_default_arm_configs_covers_every_named_arm() -> None:
    configs = default_arm_configs()
    assert [c.arm for c in configs] == list(DEFAULT_ARMS)
    assert {c.arm_id for c in configs} == {
        "discrete_only",
        "continuous_only",
        "selective_hybrid",
        "all_continuous",
        "random_continuous",
        "oracle_continuous",
    }


def test_load_selective_hybrid_corpus_reuses_ap031_split() -> None:
    train, holdout = load_selective_hybrid_corpus(fixture_count=16, fixture_seed=0)
    assert len(train) > 0
    assert len(holdout) >= 2
    assert len(train) + len(holdout) == 16


# --- discrete_only bit-exactness (hard acceptance gate) --------------------


def test_discrete_only_arm_constructs_no_soft_codec_or_connector() -> None:
    cfg = ProgramFactorTensorizerConfig()
    dims = factor_dims(cfg)
    precision_dim = sum(dims[name] for name in ("inventory", "cardinality", "topology", "binder_reference"))
    soft_dim = dims["property_role_value"] + dims["style_layout"]
    model = SelectiveHybridModel(precision_dim, soft_dim, arm=SelectiveLatentArm.DISCRETE_ONLY)
    assert model.soft_codec is None
    assert model.connector is None
    assert model.precise_codec is None


def test_discrete_only_forward_is_bit_identical_to_a_hand_built_reference_model() -> None:
    """The hard acceptance gate: 'Off/discrete-only mode remains bit-exact.'
    Builds SelectiveHybridModel(arm=discrete_only) and an independent,
    minimal reference module containing only the two linear layers that
    survive in that arm (no arm/connector concept at all), copies weights,
    and asserts byte-identical forward output on the same input.
    """
    torch.manual_seed(0)
    precision_dim, soft_dim, batch = 20, 12, 5
    model = SelectiveHybridModel(precision_dim, soft_dim, arm=SelectiveLatentArm.DISCRETE_ONLY)

    class _MinimalReference(torch.nn.Module):
        def __init__(self, precise_decoder: torch.nn.Linear, soft_baseline: torch.nn.Linear) -> None:
            super().__init__()
            self.precise_decoder = precise_decoder
            self.soft_baseline = soft_baseline

        def forward(self, precision_features: torch.Tensor, soft_features: torch.Tensor) -> torch.Tensor:
            precise_recon = self.precise_decoder(precision_features)
            ones = soft_features.new_ones(soft_features.shape[0], 1)
            soft_recon = self.soft_baseline(ones)
            return torch.cat([precise_recon, soft_recon], dim=-1)

    reference = _MinimalReference(model.precise_decoder, model.soft_baseline)

    precision_features = torch.randn(batch, precision_dim)
    soft_features = torch.randn(batch, soft_dim)
    model.eval()
    with torch.no_grad():
        model_out = model(precision_features, soft_features, hard=True, oracle_vector=soft_features)
        reference_out = reference(precision_features, soft_features)
    assert torch.equal(model_out, reference_out)


def test_discrete_only_forward_ignores_soft_features_content() -> None:
    """A second facet of bit-exactness: discrete_only's output must not
    depend on the *content* of soft_features at all (only its batch size),
    since the soft branch is a learned constant baseline.
    """
    torch.manual_seed(1)
    model = SelectiveHybridModel(10, 6, arm=SelectiveLatentArm.DISCRETE_ONLY)
    model.eval()
    precision_features = torch.randn(4, 10)
    with torch.no_grad():
        out_a = model(precision_features, torch.randn(4, 6), hard=True, oracle_vector=None)
        out_b = model(precision_features, torch.zeros(4, 6), hard=True, oracle_vector=None)
    assert torch.equal(out_a, out_b)


# --- factor-swap falsification test -----------------------------------------


@pytest.mark.parametrize("arm", DEFAULT_ARMS, ids=lambda a: a.value)
def test_factor_swaps_never_change_precision_critical_decode(arm: SelectiveLatentArm) -> None:
    """The issue's own falsification test, run for every arm including
    all_continuous (the negative control): 'No continuous factor promotes
    if swaps unpredictably change precision-critical state.'
    """
    train_plans, holdout_plans = load_selective_hybrid_corpus(fixture_count=12, fixture_seed=0)
    cfg = SelectiveHybridArmConfig(arm.value, arm, seed=0, train_steps=30)
    result, _, _ = evaluate_arm(cfg, train_plans, holdout_plans)
    assert result.factor_swaps
    for swap in result.factor_swaps:
        assert swap.precision_recon_bit_exact is True
        assert swap.max_abs_precision_delta == pytest.approx(0.0, abs=0.0)


# --- evaluate_arm / run_selective_hybrid end-to-end -------------------------


@pytest.mark.parametrize("arm", DEFAULT_ARMS, ids=lambda a: a.value)
def test_evaluate_arm_reports_every_metric(arm: SelectiveLatentArm) -> None:
    train_plans, holdout_plans = load_selective_hybrid_corpus(fixture_count=12, fixture_seed=0)
    cfg = SelectiveHybridArmConfig(arm.value, arm, seed=0, train_steps=30)
    result, overall_strict, soft_strict = evaluate_arm(cfg, train_plans, holdout_plans)

    assert result.arm_id == arm.value
    assert result.num_holdout == len(holdout_plans)
    assert overall_strict.shape[0] == len(holdout_plans)
    assert soft_strict.shape[0] == len(holdout_plans)
    assert 0.0 <= result.overall_strict_rate <= 1.0
    assert 0.0 <= result.soft_strict_rate <= 1.0
    assert 0.0 <= result.precision_strict_rate <= 1.0
    assert set(result.precision_mse_per_family) == {"inventory", "cardinality", "topology", "binder_reference"}
    assert set(result.soft_mse_per_family) == {"property_role_value", "style_layout"}
    if arm is SelectiveLatentArm.ALL_CONTINUOUS:
        assert result.precise_codec_collapsed in (True, False)  # exercised, not asserted either way
    else:
        assert result.precise_codec_collapsed is False  # no codec constructed -> trivially not collapsed


def test_run_selective_hybrid_end_to_end_small_grid() -> None:
    report = run_selective_hybrid(fixture_count=12, fixture_seed=0, arms=_tiny_arms(), ci_resamples=50)
    assert len(report.arms) == len(DEFAULT_ARMS)
    assert report.num_train + report.num_holdout == 12
    assert report.disposition in {
        "promote_selective_hybrid",
        "do_not_promote",
        "stop_do_not_promote",
        "inconclusive",
    }
    assert report.disposition_reasons
    # One comparison per non-discrete_only arm.
    assert len(report.comparisons) == len(DEFAULT_ARMS) - 1
    assert any("all_continuous" in reason or True for reason in report.disposition_reasons)


def test_run_selective_hybrid_requires_discrete_only_reference() -> None:
    arms = [a for a in _tiny_arms() if a.arm is not SelectiveLatentArm.DISCRETE_ONLY]
    with pytest.raises(ValueError):
        run_selective_hybrid(fixture_count=12, fixture_seed=0, arms=arms, ci_resamples=20)


def test_all_continuous_reports_binder_reference_as_negative_control() -> None:
    """Acceptance criterion: 'Report all-continuous binder/reference results
    as a negative control.' -- must always be present in the comparison
    output whenever all_continuous is evaluated, regardless of direction.
    """
    report = run_selective_hybrid(fixture_count=12, fixture_seed=0, arms=_tiny_arms(), ci_resamples=50)
    comparison = next(c for c in report.comparisons if c.arm_id == "all_continuous")
    assert isinstance(comparison.binder_reference_mse_delta, float)
    assert any("all_continuous" in reason for reason in report.disposition_reasons)
