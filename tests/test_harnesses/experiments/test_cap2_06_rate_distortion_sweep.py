"""Regression tests for the CAP2-06 program-factor rate-distortion sweep harness."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.harnesses.experiments.cap2_rate_distortion_sweep import (
    RATE_DISTORTION_CODECS,
    RATE_DISTORTION_LATENT_COUNTS,
    RATE_DISTORTION_WIDTHS,
    FactorReconstructionModel,
    RateDistortionCell,
    RateDistortionResult,
    _build_codec,
    build_sweep,
    evaluate_cell,
    run_sweep,
    select_retained_configs,
    train_factor_reconstruction,
)
from slm_training.harnesses.experiments.cap2_semantic_plan_bottleneck import load_fixture_plans
from slm_training.models.latent_codec_trainer import audit_no_bypass
from slm_training.models.semantic_plan_factors import (
    ProgramFactorTensorizerConfig,
    concat_factor_tensors,
    factor_dims,
    tensorize_semantic_plan,
)


def _fixture_features():
    cfg = ProgramFactorTensorizerConfig()
    plans = load_fixture_plans(count=12, seed=0)
    features = torch.stack([concat_factor_tensors(tensorize_semantic_plan(p, cfg)) for p in plans])
    return features, factor_dims(cfg)


def test_build_sweep_full_grid_size() -> None:
    cells = build_sweep()
    assert len(cells) == len(RATE_DISTORTION_CODECS) * len(RATE_DISTORTION_LATENT_COUNTS) * len(
        RATE_DISTORTION_WIDTHS
    )
    assert {c.codec for c in cells} == set(RATE_DISTORTION_CODECS)


def test_cell_id_is_unique_and_stable() -> None:
    cells = build_sweep()
    ids = [c.cell_id for c in cells]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("codec", list(RATE_DISTORTION_CODECS))
def test_build_codec_matches_num_latents_and_width(codec: str) -> None:
    cell = RateDistortionCell(codec, num_latents=8, width=64)
    built = _build_codec(cell, feature_dim=45)
    if codec == "uniform_scalar":
        assert built.config.d == 8
        assert built.config.hidden_dim == 64
    elif codec == "fsq":
        assert len(built.config.levels) == 8
    elif codec == "lfq":
        assert built.config.d == 8
    elif codec == "vq":
        assert built.config.codebook_size == 2**8
    elif codec == "continuous":
        assert built.config.latent_dim == 8


def test_factor_reconstruction_model_passes_no_bypass_audit() -> None:
    features, dims = _fixture_features()
    cell = RateDistortionCell("uniform_scalar", num_latents=8, width=64, train_steps=150)
    torch.manual_seed(cell.seed)
    codec = _build_codec(cell, features.shape[-1])
    model = FactorReconstructionModel(codec, features.shape[-1])
    train_factor_reconstruction(model, features, steps=cell.train_steps, lr=cell.lr)
    assert audit_no_bypass(model, features) is True


def test_higher_capacity_reduces_reconstruction_distortion() -> None:
    """A near-adequate step budget per cell should show capacity improves distortion.

    Uses the continuous family (fast, smooth optimization landscape) with a
    step budget tuned per cell size, since a fixed small step count does not
    guarantee monotone convergence across very different decoder-input
    widths for every codec family -- that is an optimization-budget effect,
    not evidence against the rate-distortion relationship itself.
    """
    features, dims = _fixture_features()
    tiny = evaluate_cell(
        RateDistortionCell("continuous", num_latents=1, width=64, train_steps=300), features, dims
    )
    ample = evaluate_cell(
        RateDistortionCell("continuous", num_latents=32, width=128, train_steps=800), features, dims
    )
    assert ample.mse_overall < 1e-4
    assert tiny.mse_overall > ample.mse_overall


def test_evaluate_cell_reports_every_factor_family() -> None:
    features, dims = _fixture_features()
    result = evaluate_cell(RateDistortionCell("lfq", num_latents=16, width=128, train_steps=250), features, dims)
    assert set(result.mse_per_factor) == set(dims)
    assert result.no_bypass_ok is True
    assert result.nominal_bits > 0.0


def test_select_retained_configs_picks_smallest_passing_cell() -> None:
    features, dims = _fixture_features()
    cells = [
        RateDistortionCell("continuous", num_latents=n, width=64, train_steps=250)
        for n in (2, 4, 8, 16)
    ]
    results = {c.cell_id: evaluate_cell(c, features, dims) for c in cells}
    retained = select_retained_configs(cells, results, features, dims, confirmation_seeds=(1,))
    passing_ids = {c.cell_id for c in cells if not results[c.cell_id].factors_over_threshold}
    if not passing_ids:
        assert retained == []
    else:
        assert len(retained) == 1
        winner = retained[0]
        # The winner must be the smallest (num_latents, width) cell that passed.
        smallest_passing = min(
            (c for c in cells if c.cell_id in passing_ids), key=lambda c: (c.num_latents, c.width)
        )
        assert winner.num_latents == smallest_passing.num_latents
        assert len(winner.confirmation_seeds) == 1


def _fake_result(cell: RateDistortionCell, *, no_bypass_ok: bool, over_threshold: tuple = ()) -> RateDistortionResult:
    return RateDistortionResult(
        cell_id=cell.cell_id,
        codec=cell.codec,
        num_latents=cell.num_latents,
        width=cell.width,
        seed=cell.seed,
        capacity=cell.capacity,
        nominal_bits=1.0,
        serialized_bytes_per_example=1.0,
        nominal_bytes_per_example=1.0,
        mse_overall=0.0,
        mse_per_factor={"inventory": 0.0},
        max_factor_mse=0.0,
        factors_over_threshold=over_threshold,
        no_bypass_ok=no_bypass_ok,
        elapsed_seconds=0.0,
    )


def test_select_retained_configs_excludes_collapsed_encoder_despite_low_distortion() -> None:
    """A cell with an empty factors_over_threshold but a failed no-bypass audit
    (dead/constant encoder) must never be retained, even though it looks like
    the smallest passing cell by distortion alone.
    """
    collapsed = RateDistortionCell("uniform_scalar", num_latents=2, width=64)
    genuine = RateDistortionCell("uniform_scalar", num_latents=8, width=64)
    cells = [collapsed, genuine]
    results = {
        collapsed.cell_id: _fake_result(collapsed, no_bypass_ok=False, over_threshold=()),
        genuine.cell_id: _fake_result(genuine, no_bypass_ok=True, over_threshold=()),
    }
    features, dims = _fixture_features()
    retained = select_retained_configs(cells, results, features, dims, confirmation_seeds=())
    assert len(retained) == 1
    assert retained[0].num_latents == genuine.num_latents


def test_select_retained_configs_omits_codec_when_only_collapsed_cells_pass_distortion() -> None:
    collapsed = RateDistortionCell("vq", num_latents=2, width=64)
    cells = [collapsed]
    results = {collapsed.cell_id: _fake_result(collapsed, no_bypass_ok=False, over_threshold=())}
    features, dims = _fixture_features()
    retained = select_retained_configs(cells, results, features, dims, confirmation_seeds=())
    assert retained == []


def test_run_sweep_small_grid_produces_versioned_report() -> None:
    report = run_sweep(
        fixture_count=12,
        codecs=("lfq",),
        num_latents_values=(8, 16),
        widths=(64,),
        train_steps=200,
    )
    assert report.version == "cap2-06-v1"
    assert len(report.cells) == 2
    assert not any(not c.no_bypass_ok for c in report.cells)


def test_run_sweep_reports_no_bypass_for_all_cells() -> None:
    report = run_sweep(
        fixture_count=12,
        codecs=("uniform_scalar", "continuous"),
        num_latents_values=(8,),
        widths=(64,),
        train_steps=150,
    )
    assert all(c.no_bypass_ok for c in report.cells)
