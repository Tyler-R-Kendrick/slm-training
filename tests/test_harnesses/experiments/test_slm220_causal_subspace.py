"""Tests for activation-side causal restriction diagnostics."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from slm_training.harnesses.experiments.slm218_cross_attention_retention import (
    restriction_energy,
)
from slm_training.harnesses.experiments.slm220_causal_subspace import (
    exact_jacobian,
    functional_activation_subspaces,
    hutchinson_jacobian_frobenius,
    jvp_restriction_energy,
    make_linear_input_target,
    run_fixture_retrospective,
    validate_state_manifest,
)


def test_exact_and_jvp_restriction_agree_on_linear_system() -> None:
    matrix = torch.tensor([[2.0, 0.0, 1.0], [0.0, 3.0, 0.0]], dtype=torch.float64)
    activation = torch.tensor([0.2, -0.4, 0.7], dtype=torch.float64)
    basis = torch.eye(3, dtype=torch.float64)[:, :2]

    def target(value: torch.Tensor) -> torch.Tensor:
        return matrix @ value

    jacobian = exact_jacobian(target, activation)
    exact = restriction_energy(jacobian, basis)
    via_jvp = jvp_restriction_energy(
        target,
        activation,
        basis,
        denominator=float(jacobian.square().sum()),
    )
    assert jacobian == pytest.approx(matrix)
    assert via_jvp == pytest.approx(exact)


def test_hutchinson_is_exact_for_single_output_and_deterministic() -> None:
    matrix = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
    activation = torch.zeros(3, dtype=torch.float64)
    kwargs = {"probes": 16, "seed": 220}
    first = hutchinson_jacobian_frobenius(
        lambda value: matrix @ value, activation, **kwargs
    )
    second = hutchinson_jacobian_frobenius(
        lambda value: matrix @ value, activation, **kwargs
    )
    assert first == second
    assert first["frobenius_squared"] == pytest.approx(14.0)
    assert first["standard_error"] == pytest.approx(0.0)


def test_hutchinson_converges_for_multioutput_fixture() -> None:
    matrix = torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]], dtype=torch.float64)
    result = hutchinson_jacobian_frobenius(
        lambda value: matrix @ value,
        torch.zeros(3, dtype=torch.float64),
        probes=4096,
        seed=221,
    )
    assert result["frobenius_squared"] == pytest.approx(
        float(matrix.square().sum()), rel=0.03
    )
    assert result["ci95_low"] <= float(matrix.square().sum())
    assert result["ci95_high"] >= float(matrix.square().sum())


def test_functional_subspaces_have_valid_activation_orientation_and_controls() -> None:
    weight = torch.diag(torch.tensor([4.0, 2.0, 1.0]))
    covariance = torch.diag(torch.tensor([9.0, 1.0, 0.25]))
    bases = functional_activation_subspaces(weight, covariance, k=1)
    assert set(bases) == {
        "raw_weight_top",
        "functional_top",
        "functional_middle",
        "functional_bottom",
        "covariance_top",
    }
    assert all(value.shape == (3, 1) for value in bases.values())
    assert abs(float(bases["functional_top"][0, 0])) == pytest.approx(1.0)


def test_dimension_and_orientation_errors_fail_loudly() -> None:
    with pytest.raises(ValueError, match="in_features"):
        functional_activation_subspaces(torch.ones(2, 3), torch.eye(2), k=1)
    with pytest.raises(ValueError, match="one-dimensional"):
        exact_jacobian(lambda value: value, torch.ones(1, 2))
    with pytest.raises(ValueError, match="input dimension"):
        jvp_restriction_energy(
            lambda value: value,
            torch.ones(3),
            torch.ones(2, 1),
            denominator=1.0,
        )


def test_state_group_and_split_identity_cannot_mix() -> None:
    base = {
        "state_id": "state-a",
        "group_id": "group-a",
        "module_path": "probe",
        "decision_stratum": "component",
        "split": "held_out",
    }
    assert validate_state_manifest((base,)) == validate_state_manifest((base,))
    with pytest.raises(ValueError, match="cannot mix"):
        validate_state_manifest(
            (base, {**base, "state_id": "state-b", "split": "train"})
        )
    with pytest.raises(ValueError, match="cannot repeat"):
        validate_state_manifest((base, dict(base)))


def test_random_subspace_null_matches_isotropic_k_over_dimension() -> None:
    root = Path(__file__).resolve().parents[3]
    report = run_fixture_retrospective(root)
    for row in report.snapshots:
        assert row.random_subspace_null_mean == pytest.approx(0.25, abs=0.09)
        assert (
            row.random_subspace_null_interval[0]
            <= row.random_subspace_null_mean
            <= row.random_subspace_null_interval[1]
        )


def test_isospectral_rotation_changes_restriction_energy() -> None:
    weight = torch.diag(torch.tensor([4.0, 2.0]))
    covariance = torch.eye(2)
    rotation = torch.tensor([[0.0, -1.0], [1.0, 0.0]])
    rotated = weight @ rotation
    basis = functional_activation_subspaces(weight, covariance, k=1)["functional_top"]
    rotated_basis = functional_activation_subspaces(rotated, covariance, k=1)[
        "functional_top"
    ]
    jacobian = torch.tensor([[1.0, 0.0]])
    assert torch.linalg.svdvals(weight) == pytest.approx(torch.linalg.svdvals(rotated))
    assert restriction_energy(jacobian, basis) == pytest.approx(1.0)
    assert restriction_energy(jacobian, rotated_basis) == pytest.approx(0.0)


class _Probe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.probe = nn.Linear(3, 2, bias=False, dtype=torch.float64)
        self.probe.weight.data.copy_(torch.tensor([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0]]))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.probe(value)


def test_intervention_target_reproduces_logits_and_freezes_membership() -> None:
    model = _Probe()
    value = torch.tensor([0.2, 0.3, -0.1], dtype=torch.float64)
    membership = {"hash": "legal-v1"}
    baseline, target, frozen_hash = make_linear_input_target(
        model,
        module_path="probe",
        run_model=lambda: model(value),
        project_output=lambda output: output,
        legal_membership_hash=lambda: membership["hash"],
    )
    assert frozen_hash == "legal-v1"
    assert torch.allclose(target(baseline), model(value))
    membership["hash"] = "changed"
    with pytest.raises(RuntimeError, match="membership changed"):
        target(baseline)
