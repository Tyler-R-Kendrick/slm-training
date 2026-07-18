"""Regression tests for the SDE0-01 decode-scaffolding ablation harness."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.harnesses.eval.ablate_decode_scaffolding import (
    ScaffoldFactors,
    build_stage_a_arms,
    resolve_arm_config,
    run_arm,
    stage_a_needs_stage_b,
)
from slm_training.harnesses.model_build.config import ModelBuildConfig


@pytest.fixture
def base_config() -> ModelBuildConfig:
    return ModelBuildConfig(
        train_dir=Path("outputs/data/train/v1"),
        run_root=Path("outputs/runs"),
        run_id="sde0-01-test",
        device="cpu",
    )


def test_stage_a_arm_count() -> None:
    """Stage A contains baseline, four one-factor-off arms, and all-off."""
    arms = build_stage_a_arms()
    assert len(arms) == 6
    assert arms[0].arm_id == "baseline"
    assert arms[-1].arm_id == "all_off"
    one_off_ids = {a.arm_id for a in arms[1:-1]}
    assert one_off_ids == {
        "one_off_content_floor",
        "one_off_prompt_inventory",
        "one_off_semantic_constraints",
        "one_off_attempts",
    }


def test_baseline_has_all_factors_enabled() -> None:
    arms = build_stage_a_arms()
    baseline = arms[0]
    assert baseline.factors == ScaffoldFactors()
    assert baseline.best_of_n == 4
    assert baseline.decode_path_id == "current_exact_or_compiler"


def test_one_off_arms_disable_exactly_one_factor() -> None:
    arms = build_stage_a_arms()
    baseline = arms[0].factors
    for arm in arms[1:-1]:
        diff = {
            k: (getattr(baseline, k), getattr(arm.factors, k))
            for k in baseline.to_dict()
            if getattr(baseline, k) != getattr(arm.factors, k)
        }
        assert len(diff) == 1, f"{arm.arm_id} should differ in exactly one factor"


def test_all_off_arm_has_all_factors_disabled() -> None:
    arms = build_stage_a_arms()
    all_off = arms[-1]
    assert not any(all_off.factors.to_dict().values())
    assert all_off.best_of_n == 1
    assert all_off.decode_path_id == "current_native"


def test_resolve_baseline_config_for_choice_codec(base_config: ModelBuildConfig) -> None:
    arms = build_stage_a_arms()
    baseline = arms[0]
    config, path, ok, reason = resolve_arm_config(
        base_config, baseline, output_codec="choice"
    )
    assert ok
    assert reason is None
    assert path.path_id == "current_exact_or_compiler"
    assert config.decode_min_content == -1
    assert config.honest_slot_contract is True
    assert config.slot_contract_in_context is True
    assert config.slot_contract_constrained_decode is True
    assert config.best_of_n == 4
    assert config.generate_max_attempts == 3
    assert config.allow_unconstrained_fallback is False


def test_resolve_all_off_config_for_choice_codec(base_config: ModelBuildConfig) -> None:
    arms = build_stage_a_arms()
    all_off = arms[-1]
    config, path, ok, reason = resolve_arm_config(
        base_config, all_off, output_codec="choice"
    )
    assert ok
    assert path.path_id == "current_native"
    assert config.decode_min_content == 0
    assert config.honest_slot_contract is False
    assert config.slot_contract_in_context is False
    assert config.slot_contract_constrained_decode is False
    assert config.best_of_n == 1
    assert config.generate_max_attempts == 1
    assert config.allow_unconstrained_fallback is True


def test_run_arm_fixture_mode_returns_compatible(base_config: ModelBuildConfig) -> None:
    arms = build_stage_a_arms()
    for arm in arms:
        result = run_arm(arm, base_config=base_config, output_codec="choice")
        assert result.compatible
        assert result.arm_id == arm.arm_id
        assert result.decode_path_id == arm.decode_path_id
        assert result.best_of_n == arm.best_of_n


def test_stage_b_triggered_by_nonlinear_residual() -> None:
    """If all-off deviates from additive prediction by >0.05, request Stage B."""
    from slm_training.harnesses.eval.ablate_decode_scaffolding import ArmResult

    baseline = ArmResult(
        arm_id="baseline",
        factors=ScaffoldFactors(),
        decode_path_id="current_exact_or_compiler",
        best_of_n=4,
        compatible=True,
        incompatible_reason=None,
        metrics={"meaningful_program_rate": 0.80},
    )
    # Each factor off costs 0.05 additive.
    one_offs = [
        ArmResult(
            arm_id=f"one_off_{name}",
            factors=ScaffoldFactors(**{name: False}),
            decode_path_id="current_exact_or_compiler",
            best_of_n=4,
            compatible=True,
            incompatible_reason=None,
            metrics={"meaningful_program_rate": 0.75},
        )
        for name in ScaffoldFactors().to_dict()
    ]
    # Additive prediction for all-off: 0.80 - 4*0.05 = 0.60.
    # Observed all-off: 0.50 (interaction: worse than additive).
    all_off = ArmResult(
        arm_id="all_off",
        factors=ScaffoldFactors(
            content_floor=False,
            prompt_inventory=False,
            semantic_constraints=False,
            attempts=False,
        ),
        decode_path_id="current_native",
        best_of_n=1,
        compatible=True,
        incompatible_reason=None,
        metrics={"meaningful_program_rate": 0.50},
    )
    assert stage_a_needs_stage_b((baseline,) + tuple(one_offs) + (all_off,))


def test_stage_b_not_triggered_when_additive_holds() -> None:
    from slm_training.harnesses.eval.ablate_decode_scaffolding import ArmResult

    baseline = ArmResult(
        arm_id="baseline",
        factors=ScaffoldFactors(),
        decode_path_id="current_exact_or_compiler",
        best_of_n=4,
        compatible=True,
        incompatible_reason=None,
        metrics={"meaningful_program_rate": 0.80},
    )
    one_offs = [
        ArmResult(
            arm_id=f"one_off_{name}",
            factors=ScaffoldFactors(**{name: False}),
            decode_path_id="current_exact_or_compiler",
            best_of_n=4,
            compatible=True,
            incompatible_reason=None,
            metrics={"meaningful_program_rate": 0.75},
        )
        for name in ScaffoldFactors().to_dict()
    ]
    all_off = ArmResult(
        arm_id="all_off",
        factors=ScaffoldFactors(
            content_floor=False,
            prompt_inventory=False,
            semantic_constraints=False,
            attempts=False,
        ),
        decode_path_id="current_native",
        best_of_n=1,
        compatible=True,
        incompatible_reason=None,
        metrics={"meaningful_program_rate": 0.60},
    )
    assert not stage_a_needs_stage_b((baseline,) + tuple(one_offs) + (all_off,))
