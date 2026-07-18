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


def test_checkpoint_sha256_mismatch_marks_all_arms_incompatible(
    base_config: ModelBuildConfig, tmp_path: Path
) -> None:
    from slm_training.harnesses.eval.ablate_decode_scaffolding import run_stage_a

    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"not the real checkpoint")
    report = run_stage_a(
        base_config,
        checkpoint_id="fake",
        checkpoint_sha256="0" * 64,
        checkpoint_path=checkpoint,
        output_codec="choice",
    )
    assert all(not a.compatible for a in report.arms)
    assert any("sha256 mismatch" in (a.incompatible_reason or "") for a in report.arms)


def test_missing_checkpoint_marks_all_arms_incompatible(
    base_config: ModelBuildConfig, tmp_path: Path
) -> None:
    from slm_training.harnesses.eval.ablate_decode_scaffolding import run_stage_a

    missing = tmp_path / "missing.pt"
    report = run_stage_a(
        base_config,
        checkpoint_id="missing",
        checkpoint_sha256="0" * 64,
        checkpoint_path=missing,
        output_codec="choice",
    )
    assert all(not a.compatible for a in report.arms)
    assert any("not found" in (a.incompatible_reason or "") for a in report.arms)


def test_verify_checkpoint_matching_sha256(tmp_path: Path) -> None:
    import hashlib

    from slm_training.harnesses.eval.ablate_decode_scaffolding import _verify_checkpoint

    checkpoint = tmp_path / "fake.pt"
    checkpoint.write_bytes(b"good checkpoint bytes")
    sha = hashlib.sha256(b"good checkpoint bytes").hexdigest()
    ok, reason = _verify_checkpoint(checkpoint, sha)
    assert ok
    assert reason is None


def test_verify_checkpoint_missing_file(tmp_path: Path) -> None:
    from slm_training.harnesses.eval.ablate_decode_scaffolding import _verify_checkpoint

    missing = tmp_path / "missing.pt"
    ok, reason = _verify_checkpoint(missing, "0" * 64)
    assert not ok
    assert "not found" in (reason or "")


def test_no_inventory_arm_does_not_surface_contract(base_config: ModelBuildConfig) -> None:
    """A prompt-inventory-disabled arm must turn off all slot-contract surfacing."""
    arms = build_stage_a_arms()
    no_inventory = next(a for a in arms if a.arm_id == "one_off_prompt_inventory")
    config, _path, ok, _reason = resolve_arm_config(
        base_config, no_inventory, output_codec="choice"
    )
    assert ok
    assert config.honest_slot_contract is False
    assert config.slot_contract_in_context is False
    assert config.slot_contract_constrained_decode is False


def test_one_attempt_arm_has_single_attempt(base_config: ModelBuildConfig) -> None:
    """Disabling attempts must set best_of_n=1 and generate_max_attempts=1."""
    arms = build_stage_a_arms()
    one_attempt = next(a for a in arms if a.arm_id == "one_off_attempts")
    config, _path, ok, _reason = resolve_arm_config(
        base_config, one_attempt, output_codec="choice"
    )
    assert ok
    assert config.best_of_n == 1
    assert config.generate_max_attempts == 1


def test_grammar_only_arm_keeps_grammar_constrained(base_config: ModelBuildConfig) -> None:
    """With semantic constraints disabled, grammar/syntax enforcement remains on."""
    arms = build_stage_a_arms()
    grammar_only = next(a for a in arms if a.arm_id == "all_off")
    config, _path, ok, _reason = resolve_arm_config(
        base_config, grammar_only, output_codec="choice"
    )
    assert ok
    assert config.grammar_constrained is True


def test_replay_same_arm_produces_identical_config(base_config: ModelBuildConfig) -> None:
    """Config resolution is deterministic for the same arm and base config."""
    from dataclasses import asdict

    arms = build_stage_a_arms()
    arm = arms[0]
    c1, _p1, ok1, _ = resolve_arm_config(base_config, arm, output_codec="choice")
    c2, _p2, ok2, _ = resolve_arm_config(base_config, arm, output_codec="choice")
    assert ok1 and ok2
    assert asdict(c1) == asdict(c2)


def test_stage_b_excludes_stage_a_cells_by_default() -> None:
    from slm_training.harnesses.eval.ablate_decode_scaffolding import build_stage_b_arms

    stage_a = build_stage_a_arms()
    stage_b = build_stage_b_arms(exclude_stage_a=True)
    stage_a_factor_tuples = {tuple(a.factors.to_dict().values()) for a in stage_a}
    stage_b_factor_tuples = {tuple(a.factors.to_dict().values()) for a in stage_b}
    assert len(stage_b) == 10
    assert not stage_a_factor_tuples & stage_b_factor_tuples


def test_stage_b_can_include_all_16_cells() -> None:
    from slm_training.harnesses.eval.ablate_decode_scaffolding import build_stage_b_arms

    stage_b = build_stage_b_arms(exclude_stage_a=False)
    assert len(stage_b) == 16


def test_compute_paired_deltas() -> None:
    from slm_training.harnesses.eval.ablate_decode_scaffolding import (
        ArmResult,
        ScaffoldFactors,
        compute_paired_deltas,
    )

    baseline = ArmResult(
        arm_id="baseline",
        factors=ScaffoldFactors(),
        decode_path_id="current_exact_or_compiler",
        best_of_n=4,
        compatible=True,
        incompatible_reason=None,
        metrics={"meaningful_program_rate": 0.80, "placeholder_fidelity": 0.90},
    )
    arm = ArmResult(
        arm_id="one_off_content_floor",
        factors=ScaffoldFactors(content_floor=False),
        decode_path_id="current_exact_or_compiler",
        best_of_n=4,
        compatible=True,
        incompatible_reason=None,
        metrics={"meaningful_program_rate": 0.75, "placeholder_fidelity": 0.88},
    )
    deltas = compute_paired_deltas(baseline, (arm,))
    by_metric = {d.metric: d for d in deltas}
    assert "meaningful_program_rate" in by_metric
    assert by_metric["meaningful_program_rate"].absolute_delta == pytest.approx(-0.05)
    assert by_metric["meaningful_program_rate"].relative_delta == pytest.approx(-0.05 / 0.80)


def test_estimate_additive_interaction_detects_nonadditivity() -> None:
    from slm_training.harnesses.eval.ablate_decode_scaffolding import (
        ArmResult,
        ScaffoldFactors,
        estimate_additive_interaction,
    )

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
        metrics={"meaningful_program_rate": 0.50},
    )
    estimate = estimate_additive_interaction((baseline,) + tuple(one_offs) + (all_off,))
    assert estimate["needs_stage_b"] is True
    assert estimate["residual"] == pytest.approx(-0.10)


def test_estimate_additive_interaction_ignores_incompatible_arms() -> None:
    from slm_training.harnesses.eval.ablate_decode_scaffolding import (
        ArmResult,
        ScaffoldFactors,
        estimate_additive_interaction,
    )

    baseline = ArmResult(
        arm_id="baseline",
        factors=ScaffoldFactors(),
        decode_path_id="current_exact_or_compiler",
        best_of_n=4,
        compatible=True,
        incompatible_reason=None,
        metrics={"meaningful_program_rate": 0.80},
    )
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
        metrics={"meaningful_program_rate": 0.80},
    )
    estimate = estimate_additive_interaction((baseline, all_off))
    assert estimate["needs_stage_b"] is False
    assert estimate["residual"] == pytest.approx(0.0)
