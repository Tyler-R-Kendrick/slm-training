"""SLM-231 recurrence-dynamics estimator and transition-parity tests."""

from __future__ import annotations

import pytest
import torch

from slm_training.harnesses.experiments.slm231_recurrence_dynamics import (
    DynamicsVerdict,
    block_singular_estimate,
    classify_dynamics,
    exact_product,
    finite_time_lyapunov,
    flatten_projected_state,
    linear_cka,
    principal_angles,
    replace_projected_state,
    residual_jacobians,
    state_projection,
    trajectory_product_estimate,
)
from slm_training.models.recursive_denoiser import SharedRecursiveDenoiserTower


def test_residual_increment_is_separate_from_identity() -> None:
    increment = torch.diag(torch.tensor([0.01, -0.02], dtype=torch.float64))
    transition = torch.eye(2, dtype=torch.float64) + increment
    state = torch.tensor([0.2, -0.3], dtype=torch.float64)
    observed_increment, observed_transition = residual_jacobians(
        lambda value: transition @ value, state
    )
    torch.testing.assert_close(observed_increment, increment)
    torch.testing.assert_close(observed_transition, transition)
    assert (
        classify_dynamics(
            increment_spectral_norm=0.02,
            product_spectral_norm=1.01,
            maximum_ftle=0.005,
            update_alignment_cosine=1.0,
            outcome_verdict="stagnant",
        )
        is DynamicsVerdict.IDENTITY_NEAR_ONE_ONLY
    )


def test_block_jvp_vjp_agrees_with_exact_singular_values() -> None:
    matrix = torch.diag(torch.tensor([3.0, 2.0, 0.5], dtype=torch.float64))
    state = torch.zeros(3, dtype=torch.float64)
    estimate = block_singular_estimate(
        lambda value: matrix @ value,
        state,
        k=2,
        iterations=16,
        seed=231,
    )
    assert estimate.singular_values == pytest.approx((3.0, 2.0), rel=1e-6)
    assert estimate.residual < 1e-5


def test_product_order_and_ftle_for_noncommuting_system() -> None:
    first = torch.tensor([[1.0, 1.0], [0.0, 1.0]], dtype=torch.float64)
    second = torch.tensor([[2.0, 0.0], [0.0, 0.5]], dtype=torch.float64)
    product = exact_product((first, second))
    torch.testing.assert_close(product, second @ first)
    assert not torch.allclose(product, first @ second)
    singular = tuple(float(value) for value in torch.linalg.svdvals(product))
    exponents = finite_time_lyapunov(singular, depth=2)
    assert exponents[0] == pytest.approx(
        0.5 * torch.log(torch.tensor(singular[0])).item()
    )


def test_trajectory_jvp_vjp_product_matches_exact() -> None:
    matrices = (
        torch.diag(torch.tensor([1.2, 0.8], dtype=torch.float64)),
        torch.tensor([[1.0, 0.2], [0.0, 0.9]], dtype=torch.float64),
    )
    states = (
        torch.tensor([0.1, 0.2], dtype=torch.float64),
        torch.tensor([0.12, 0.16], dtype=torch.float64),
    )
    estimate = trajectory_product_estimate(
        tuple(lambda value, matrix=matrix: matrix @ value for matrix in matrices),
        states,
        k=2,
        iterations=12,
        seed=232,
    )
    exact = torch.linalg.svdvals(exact_product(matrices))
    assert estimate.singular_values == pytest.approx(
        tuple(float(value) for value in exact), rel=1e-6
    )
    assert estimate.residual < 1e-5


def test_identity_transition_has_zero_increment_and_zero_ftle() -> None:
    state = torch.ones(3, dtype=torch.float64)
    increment, composite = residual_jacobians(lambda value: value, state)
    assert torch.count_nonzero(increment) == 0
    assert finite_time_lyapunov(
        tuple(float(value) for value in torch.linalg.svdvals(composite)),
        depth=4,
    ) == pytest.approx((0.0, 0.0, 0.0), abs=1e-12)


def test_projection_roundtrip_preserves_inactive_positions() -> None:
    y = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    z = -y
    projection = state_projection(y, z, (0, 2))
    flat = flatten_projected_state(y, z, projection)
    changed = flat + 1
    next_y, next_z = replace_projected_state(changed, y, z, projection)
    assert next_z is not None
    torch.testing.assert_close(next_y[:, 1], y[:, 1])
    torch.testing.assert_close(next_z[:, 1], z[:, 1])
    torch.testing.assert_close(
        flatten_projected_state(next_y, next_z, projection), changed
    )


def test_transition_step_exactly_matches_normal_recurrence_without_mutation() -> None:
    torch.manual_seed(231)
    tower = SharedRecursiveDenoiserTower(
        vocab_size=17,
        d_model=8,
        n_layers=2,
        n_heads=2,
        max_len=8,
        recursive_steps=3,
        recursive_transition_layers=2,
    ).eval()
    noisy = torch.tensor([[1, 2, 0]])
    context = torch.randn(1, 2, 8)
    ctx_pad = torch.tensor([[False, False]])
    before = {key: value.detach().clone() for key, value in tower.state_dict().items()}

    normal = tower.recursive_outputs(noisy, context, 0, ctx_pad)
    initial = tower.initial_transition_state(noisy, context, 0, ctx_pad)
    y, z = initial["y"], initial["z"]
    assert isinstance(y, torch.Tensor)
    assert z is None or isinstance(z, torch.Tensor)
    manual_logits = []
    for _ in range(3):
        step = tower.transition_step(
            y,
            z,
            context,
            initial["self_pad_mask"],
            ctx_pad,
            runtime_symbol_features=initial["runtime_symbol_features"],
        )
        y, z = step["y"], step["z"]
        assert isinstance(y, torch.Tensor)
        assert z is None or isinstance(z, torch.Tensor)
        logits = step["logits"]
        assert isinstance(logits, torch.Tensor)
        manual_logits.append(logits)

    for observed, expected in zip(normal["depth_logits"], manual_logits):
        torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    for key, value in tower.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_alignment_metrics_distinguish_same_and_orthogonal_subspaces() -> None:
    left = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0]])
    same = left.clone()
    orthogonal = torch.tensor([[0.0, 1.0], [0.0, -1.0], [0.0, 2.0]])
    assert principal_angles(left, same, rank=1) == pytest.approx((0.0,))
    assert principal_angles(left, orthogonal, rank=1) == pytest.approx(
        (torch.pi / 2,), abs=1e-7
    )
    assert linear_cka(left, same) == pytest.approx(1.0)
    assert linear_cka(left, orthogonal) == pytest.approx(1.0)
