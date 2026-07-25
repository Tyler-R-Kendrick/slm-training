"""Tests for SLM-214 (NCS0-01) SpectralSnapshotV1 fixture harness."""

from __future__ import annotations

import math

import pytest
import torch

from slm_training.harnesses.experiments.slm214_spectral_snapshot import (
    MATRIX_SET,
    MATRIX_VERSION,
    SpectralSnapshotReport,
    build_toy_model,
    make_low_rank_matrix,
    make_pareto_tail_matrix,
    make_spiked_matrix,
    run_spectral_snapshot_fixture,
    sample_null_summary,
    spectral_trap_statistics,
)


@pytest.mark.parametrize("initializer", ["xavier_uniform", "kaiming_uniform"])
def test_uniform_null_initializers_are_deterministic(initializer: str) -> None:
    from slm_training.harnesses.experiments.slm214_spectral_snapshot import (
        sample_null_summary,
    )

    left = sample_null_summary(16, 16, torch.float32, initializer, draws=3)
    right = sample_null_summary(16, 16, torch.float32, initializer, draws=3)
    assert left == right

pytest.importorskip("torch")


def test_toy_model_has_tied_embedding_head() -> None:
    model = build_toy_model()
    names = {n for n, _ in model.named_parameters()}
    assert "token_embed.weight" in names
    # PyTorch lists a tied parameter under one name; storage-pointer grouping still
    # deduplicates it at the snapshot level.
    assert model.lm_head.weight is model.token_embed.weight


def test_role_classification_on_toy_model() -> None:
    report = run_spectral_snapshot_fixture(build_toy_model(), null_draws=5)
    roles = {s.semantic_role for s in report.snapshots}
    assert "token_embedding" in roles
    assert "mlp_in" in roles or "mlp_out" in roles or "mlp" in roles
    assert "action_head" in roles


def _model_with_explicit_alias() -> torch.nn.Module:
    """Module whose two parameters share the same underlying storage."""

    class AliasModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            base = torch.randn(16, 16)
            self.shared_a = torch.nn.Parameter(base)
            self.shared_b = torch.nn.Parameter(base)
            self.other = torch.nn.Parameter(torch.randn(8, 8))

    return AliasModel()


def test_tied_storage_emitted_once_with_aliases() -> None:
    report = run_spectral_snapshot_fixture(_model_with_explicit_alias(), null_draws=5)
    snap = next((s for s in report.snapshots if s.matrix_id == "shared_a"), None)
    assert snap is not None
    assert "shared_b" in snap.tied_aliases
    # Only one snapshot should exist for the tied pair.
    assert sum(1 for s in report.snapshots if "shared" in s.matrix_id) == 1


def test_null_calibration_shape_dependence() -> None:
    """Reproduce the documented shape dependence of raw Hill alpha on null matrices."""
    s128 = sample_null_summary(128, 128, torch.float32, "gaussian", draws=50)
    s512_128 = sample_null_summary(512, 128, torch.float32, "gaussian", draws=50)
    assert s128["mean_alpha"] is not None
    assert s512_128["mean_alpha"] is not None
    # Documented ranges are approximate and estimator-specific. The harness uses
    # native PyTorch SVD + a tail Hill estimator; we check the qualitative shape
    # dependence (larger aspect ratio -> larger raw alpha) within declared bounds.
    assert 1.5 <= s128["mean_alpha"] <= 3.0
    assert 3.0 <= s512_128["mean_alpha"] <= 6.0
    assert s512_128["mean_alpha"] > s128["mean_alpha"]


def test_pareto_tail_alpha_recovery() -> None:
    w = make_pareto_tail_matrix(128, 128, alpha_true=2.5, rank=64)
    s = torch.linalg.svdvals(w)
    # Hill alpha over the tail should be in the neighborhood of the true alpha.
    from slm_training.harnesses.experiments.slm214_spectral_snapshot import _hill_alpha

    alpha, xmin, k = _hill_alpha(s)
    assert 1.8 <= alpha <= 3.5
    assert xmin > 0
    assert k >= 4


def test_low_rank_matrix_has_zero_tail_singular_values() -> None:
    w = make_low_rank_matrix(32, 32, rank=3)
    s = torch.linalg.svdvals(w)
    assert (s[3:] < 1e-4).all()


def test_spiked_matrix_has_large_top_singular_values() -> None:
    w = make_spiked_matrix(64, 64, spike_count=4)
    s = torch.linalg.svdvals(w)
    # The fourth singular value should be much larger than the fifth.
    assert float(s[3]) > 2.0 * float(s[4])


def test_isospectral_rotation_invariants() -> None:
    w = make_pareto_tail_matrix(32, 32, alpha_true=2.5, rank=16)
    q, _ = torch.linalg.qr(torch.randn(32, 32))
    w_rot = q @ w
    s_orig = torch.linalg.svdvals(w)
    s_rot = torch.linalg.svdvals(w_rot)
    assert torch.allclose(s_orig, s_rot, atol=1e-4)


def test_scalar_rescaling_invariants() -> None:
    w = make_pareto_tail_matrix(32, 32, alpha_true=2.5, rank=16)
    s = torch.linalg.svdvals(w)
    s_scaled = torch.linalg.svdvals(w * 7.0)
    assert torch.allclose(s * 7.0, s_scaled, atol=1e-4)


def test_ineligible_small_matrix_is_rejected() -> None:
    class Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.small = torch.nn.Linear(4, 4, bias=False)

    report = run_spectral_snapshot_fixture(Tiny(), null_draws=5)
    snap = report.snapshots[0]
    assert snap.eligibility == "ineligible"
    assert "<8 singular values" in snap.ineligibility_reason
    assert snap.hill_alpha is None


def test_stable_and_effective_rank_on_known_matrix() -> None:
    # Exact rank-5 matrix with five equal singular values -> stable rank = 5.
    u = torch.linalg.qr(torch.randn(16, 5))[0]
    v = torch.linalg.qr(torch.randn(16, 5))[0]
    w = u @ v.T
    report = run_spectral_snapshot_fixture(_Wrapper(w), null_draws=5)
    snap = report.snapshots[0]
    assert math.isclose(snap.stable_rank, 5.0, rel_tol=0.15)
    assert 4.0 <= snap.effective_rank <= 6.0


class _Wrapper(torch.nn.Module):
    def __init__(self, w: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(w)


def test_report_round_trip() -> None:
    report = run_spectral_snapshot_fixture(build_toy_model(), null_draws=5)
    recovered = SpectralSnapshotReport.from_dict(report.to_dict())
    assert recovered.matrix_set == MATRIX_SET
    assert recovered.matrix_version == MATRIX_VERSION
    assert recovered.n_matrices == report.n_matrices


def test_null_cache_is_deterministic() -> None:
    a = sample_null_summary(16, 16, torch.float32, "gaussian", draws=10)
    b = sample_null_summary(16, 16, torch.float32, "gaussian", draws=10)
    assert a["null_key"] == b["null_key"]
    assert a["mean_alpha"] is not None
    assert b["mean_alpha"] is not None


def test_canonical_trap_projection_is_scale_invariant() -> None:
    matrix = torch.randn(16, 16, generator=torch.Generator().manual_seed(214))
    first = spectral_trap_statistics(matrix, null_draws=8, seed=9)
    scaled = spectral_trap_statistics(matrix * 11, null_draws=8, seed=9)
    assert scaled["top_gap_ratio"] == pytest.approx(first["top_gap_ratio"])
    assert scaled["outlier_energy_fraction"] == pytest.approx(
        first["outlier_energy_fraction"]
    )
    assert scaled["trap_z"] == pytest.approx(first["trap_z"])
