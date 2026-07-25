"""Tests for SLM-219 trap metrics and warning evaluation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from slm_training.harnesses.experiments.slm219_correlation_traps import (
    CollapseRuleV1,
    TrajectoryPointV1,
    WarningRuleV1,
    build_trajectory_inventory,
    collapse_onset,
    evaluate_warning_rule,
    native_trap_metrics,
    warning_onset,
)


def _point(
    step: int,
    *,
    structure: float,
    repetition: float,
    trap_z: float,
) -> TrajectoryPointV1:
    metrics = native_trap_metrics(torch.eye(4), null_draws=4, seed=step)
    metrics = replace(metrics, trap_z=trap_z)
    return TrajectoryPointV1(
        trajectory_id="fixture",
        family="fixture",
        seed=0,
        step=step,
        tokens=step * 10,
        role="mlp_out",
        trap=metrics,
        heldout_nll=1.0,
        gradient_norm=1.0,
        rms_drift=0.1,
        update_norm=0.1,
        structural_similarity=structure,
        repetition_rate=repetition,
        recall=0.5,
        fidelity=0.5,
        debt_rate=0.1,
        elapsed_seconds=0.0,
    )


def test_trap_metrics_are_scale_invariant_and_spike_sensitive() -> None:
    generator = torch.Generator().manual_seed(7)
    bulk = torch.randn((16, 16), generator=generator)
    spike = bulk + 20 * torch.eye(16)[:, :1] @ torch.eye(16)[:1, :]
    base = native_trap_metrics(bulk, null_draws=16, seed=11)
    scaled = native_trap_metrics(bulk * 9, null_draws=16, seed=11)
    trapped = native_trap_metrics(spike, null_draws=16, seed=11)
    assert scaled.top_gap_ratio == pytest.approx(base.top_gap_ratio)
    assert scaled.outlier_energy_fraction == pytest.approx(base.outlier_energy_fraction)
    assert trapped.outlier_energy_fraction > base.outlier_energy_fraction
    assert trapped.trap_z > base.trap_z


def test_collapse_onset_ignores_spectral_fields() -> None:
    points = [
        _point(0, structure=0.6, repetition=0.1, trap_z=100),
        _point(10, structure=0.1, repetition=0.8, trap_z=-10),
        _point(20, structure=0.1, repetition=0.8, trap_z=-10),
    ]
    assert collapse_onset(points, CollapseRuleV1()) == 10
    changed = [replace(point, trap=replace(point.trap, trap_z=999)) for point in points]
    assert collapse_onset(changed, CollapseRuleV1()) == 10


def test_warning_precedes_known_synthetic_onset_and_shuffle_is_controlled() -> None:
    points = [
        _point(0, structure=0.6, repetition=0.1, trap_z=0),
        _point(10, structure=0.6, repetition=0.1, trap_z=2.5),
        _point(20, structure=0.6, repetition=0.1, trap_z=3.0),
        _point(30, structure=0.1, repetition=0.8, trap_z=3.5),
        _point(40, structure=0.1, repetition=0.8, trap_z=4.0),
    ]
    result = evaluate_warning_rule([points], warning_rule=WarningRuleV1())
    assert result["true_positive"] == 1
    assert result["rows"][0]["collapse_onset_step"] == 30
    assert result["rows"][0]["warning_onset_step"] == 20
    shuffled = evaluate_warning_rule([points], time_shuffle=True)
    assert shuffled["time_shuffled"] is True
    assert shuffled["rows"][0]["warning_onset_step"] != 20


def test_warning_onset_is_confirmation_step_not_backdated() -> None:
    points = [
        _point(0, structure=0.6, repetition=0.1, trap_z=0),
        _point(40, structure=0.6, repetition=0.1, trap_z=2.5),
        _point(50, structure=0.1, repetition=0.8, trap_z=3.0),
    ]
    assert warning_onset(points) == 50
    result = evaluate_warning_rule([points])
    assert result["true_positive"] == 0
    assert result["false_negative"] == 1
    assert result["rows"][0]["valid_pre_collapse_warning"] is False


def test_empty_trajectory_is_excluded_not_counted_true_negative() -> None:
    result = evaluate_warning_rule([[]])
    assert result["excluded_empty_trajectories"] == 1
    assert result["true_negative"] == 0
    assert result["rows"][0]["exclusion_reason"] == "empty trajectory"


def test_input_order_does_not_change_warning_or_collapse() -> None:
    ordered = [
        _point(0, structure=0.6, repetition=0.1, trap_z=0),
        _point(10, structure=0.6, repetition=0.1, trap_z=3),
        _point(20, structure=0.6, repetition=0.1, trap_z=3),
        _point(30, structure=0.1, repetition=0.8, trap_z=3),
    ]
    assert evaluate_warning_rule([ordered]) == evaluate_warning_rule(
        [[ordered[2], ordered[0], ordered[3], ordered[1]]]
    )


def test_missing_intervals_are_explicit_and_inventory_is_deterministic() -> None:
    root = Path(__file__).resolve().parents[3]
    first = build_trajectory_inventory(root)
    second = build_trajectory_inventory(root)
    assert first == second
    assert first["eligible_historical_trajectories"] == 0
    assert first["actual_reproduction"]["checkpoint_count"] == 6
    assert first["actual_reproduction"]["eligible_pre_collapse_trajectory"] is True
    assert all(row["missing_checkpoint_intervals"] for row in first["sources"])


def test_invalid_matrix_and_null_budget_fail_closed() -> None:
    with pytest.raises(ValueError, match="two-dimensional"):
        native_trap_metrics(torch.ones(4), null_draws=4)
    with pytest.raises(ValueError, match="at least 3"):
        native_trap_metrics(torch.eye(4), null_draws=2)
