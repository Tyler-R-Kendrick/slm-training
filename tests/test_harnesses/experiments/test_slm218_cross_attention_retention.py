"""Tests for SLM-218 subspace and context-interface geometry."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from slm_training.harnesses.experiments.slm218_cross_attention_retention import (
    CheckpointCompatibilityV1,
    activation_subspace_alignment,
    analyze_retention,
    assert_checkpoint_compatible,
    build_family_coverage,
    principal_angles,
    qk_bilinear_summary,
    restriction_energy,
)


def test_identical_scaled_and_orthogonal_subspaces() -> None:
    parent = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    identical = analyze_retention(parent, parent, k=2)
    scaled = analyze_retention(parent, parent * 3, k=2)
    reversed_order = analyze_retention(
        parent, torch.diag(torch.tensor([1.0, 2.0, 3.0, 4.0])), k=2
    )
    assert identical.projection_overlap == pytest.approx(1.0)
    assert identical.rms_drift == 0.0
    assert scaled.projection_overlap == pytest.approx(1.0)
    assert scaled.rms_drift > 0
    assert reversed_order.projection_overlap == pytest.approx(0.0)
    assert all(angle == pytest.approx(math.pi / 2) for angle in reversed_order.principal_angles_radians)


def test_isospectral_rotation_changes_overlap_not_singular_values() -> None:
    parent = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    rotation = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, 0.0],
        ]
    )
    result = analyze_retention(parent, parent @ rotation, k=1)
    assert result.parent_singular_values == pytest.approx(result.child_singular_values)
    assert result.projection_overlap == pytest.approx(0.0)


def test_principal_angle_dimension_guard() -> None:
    with pytest.raises(ValueError, match="equal dimensions"):
        principal_angles(torch.eye(2), torch.eye(3))


def test_qk_orientation_and_dimensions() -> None:
    query = torch.randn(3, 4)
    key = torch.randn(3, 5)
    summary = qk_bilinear_summary(query, key)
    assert summary["shape"] == [4, 5]
    assert summary["orientation"] == "input_query_by_input_context = Wq.T @ Wk"
    with pytest.raises(ValueError, match="output dimensions"):
        qk_bilinear_summary(query, torch.randn(2, 5))


def test_activation_alignment_and_restriction_energy() -> None:
    covariance = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    weight = torch.diag(torch.tensor([4.0, 3.0, 2.0, 1.0]))
    alignment = activation_subspace_alignment(covariance, weight, k=2)
    assert alignment["projection_overlap"] == pytest.approx(1.0)
    jacobian = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    basis = torch.eye(4)[:, :2]
    assert restriction_energy(jacobian, basis) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="input dimension"):
        restriction_energy(torch.eye(3), basis)


def test_checkpoint_compatibility_fails_closed() -> None:
    parent = CheckpointCompatibilityV1("twotower", "tok", "cfg", "v1")
    assert_checkpoint_compatible(parent, parent)
    with pytest.raises(ValueError, match="tokenizer_sha"):
        assert_checkpoint_compatible(
            parent, CheckpointCompatibilityV1("twotower", "other", "cfg", "v1")
        )


def test_family_coverage_is_immutable_and_incomplete() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    first = build_family_coverage(repo_root)
    second = build_family_coverage(repo_root)
    assert first == second
    assert first["manifest_hash"]
    assert first["complete_context_families"] == 0
    assert first["complete_retention_families"] == 0
    assert all(row["exclusion_reason"] for row in first["sources"])
