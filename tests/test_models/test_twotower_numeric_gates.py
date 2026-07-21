"""SLM-242 (RSC-A06): fail-closed numeric/schedule validation gates.

These tests prove that ``ModelBuildConfig`` and ``TwoTowerConfig`` reject
invalid weights, masks, schedules, and recursive-depth contracts at
construction time rather than silently truncating or zeroing them.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from slm_training.harnesses.model_build.config import ModelBuildConfig
from slm_training.models.twotower_numeric_gates import (
    NumericValidationError,
    finite_scalar,
    finite_non_negative_vector,
    interval_scalar,
    non_negative_scalar,
    non_empty_vector,
    normalized_probability_vector,
    positive_scalar,
    positive_sum_vector,
    strictly_increasing_sequence,
    unique_enum_sequence,
)


def test_finite_scalar_passes_for_finite_real() -> None:
    assert finite_scalar("x", 1.5) == 1.5
    assert finite_scalar("x", 0) == 0.0
    assert finite_scalar("x", -3) == -3.0
    assert finite_scalar("x", None) is None


def test_finite_scalar_rejects_non_real() -> None:
    with pytest.raises(NumericValidationError, match="real number"):
        finite_scalar("x", "1.0")
    with pytest.raises(NumericValidationError, match="real number"):
        finite_scalar("x", object())


def test_finite_scalar_rejects_nan_and_inf() -> None:
    with pytest.raises(NumericValidationError, match="finite"):
        finite_scalar("x", float("nan"))
    with pytest.raises(NumericValidationError, match="finite"):
        finite_scalar("x", float("inf"))
    with pytest.raises(NumericValidationError, match="finite"):
        finite_scalar("x", float("-inf"))


def test_non_negative_scalar_rejects_negative() -> None:
    with pytest.raises(NumericValidationError, match="non-negative"):
        non_negative_scalar("x", -1e-9)


def test_positive_scalar_rejects_non_positive() -> None:
    with pytest.raises(NumericValidationError, match="positive"):
        positive_scalar("x", 0)
    with pytest.raises(NumericValidationError, match="positive"):
        positive_scalar("x", -1)
    with pytest.raises(NumericValidationError, match="real number"):
        positive_scalar("x", True)


def test_interval_scalar_enforces_closed_bounds() -> None:
    assert interval_scalar("x", 0.5, 0.0, 1.0) == 0.5
    assert interval_scalar("x", 0.0, 0.0, 1.0) == 0.0
    assert interval_scalar("x", 1.0, 0.0, 1.0) == 1.0
    with pytest.raises(NumericValidationError, match="must be in"):
        interval_scalar("x", 1.1, 0.0, 1.0)
    with pytest.raises(NumericValidationError, match="must be in"):
        interval_scalar("x", -0.01, 0.0, 1.0)


def test_non_empty_vector_rejects_empty() -> None:
    with pytest.raises(NumericValidationError, match="non-empty"):
        non_empty_vector("v", ())


def test_finite_non_negative_vector_rejects_bad_elements() -> None:
    with pytest.raises(NumericValidationError, match="finite"):
        finite_non_negative_vector("v", (1.0, float("nan")))
    with pytest.raises(NumericValidationError, match="non-negative"):
        finite_non_negative_vector("v", (1.0, -0.5))


def test_positive_sum_vector_rejects_all_zero() -> None:
    with pytest.raises(NumericValidationError, match="positive sum"):
        positive_sum_vector("v", (0.0, 0.0))


def test_normalized_probability_vector_enforces_sum_one() -> None:
    assert normalized_probability_vector("v", (0.5, 0.5)) == (0.5, 0.5)
    with pytest.raises(NumericValidationError, match="sum to 1"):
        normalized_probability_vector("v", (0.5, 0.4))


def test_strictly_increasing_sequence_requires_positive_sorted_integers() -> None:
    assert strictly_increasing_sequence("s", (1, 2, 5)) == (1, 2, 5)
    with pytest.raises(NumericValidationError, match="strictly increasing"):
        strictly_increasing_sequence("s", (1, 1, 2))
    with pytest.raises(NumericValidationError, match="positive"):
        strictly_increasing_sequence("s", (0, 1, 2))
    with pytest.raises(NumericValidationError, match="integer"):
        strictly_increasing_sequence("s", (1.5, 2.5))


def test_unique_enum_sequence_rejects_unknown_and_duplicates() -> None:
    assert unique_enum_sequence("s", ("a", "b"), {"a", "b", "c"}) == ("a", "b")
    with pytest.raises(NumericValidationError, match="one of"):
        unique_enum_sequence("s", ("a", "x"), {"a", "b"})
    with pytest.raises(NumericValidationError, match="duplicate"):
        unique_enum_sequence("s", ("a", "a"), {"a", "b"})


# --------------------------------------------------------------------------- #
# ModelBuildConfig numeric gates
# --------------------------------------------------------------------------- #


def _valid_build_config(**overrides: object) -> ModelBuildConfig:
    defaults: dict[str, object] = {
        "train_dir": Path("outputs/data/train"),
        "run_id": "slm242-test",
    }
    defaults.update(overrides)
    return ModelBuildConfig(**defaults)  # type: ignore[arg-type]


def test_model_build_config_rejects_negative_weight() -> None:
    with pytest.raises(ValueError, match="weight"):
        _valid_build_config(ltr_loss_weight=-0.1)


def test_model_build_config_rejects_nan_weight() -> None:
    with pytest.raises(ValueError, match="finite"):
        _valid_build_config(fidelity_loss_weight=float("nan"))


def test_model_build_config_rejects_inf_weight() -> None:
    with pytest.raises(ValueError, match="finite"):
        _valid_build_config(structural_bias=float("inf"))


def test_model_build_config_rejects_mask_min_greater_than_mask_max() -> None:
    with pytest.raises(ValueError, match="mask_min"):
        _valid_build_config(mask_min=0.9, mask_max=0.1)


def test_model_build_config_rejects_mask_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="must be in"):
        _valid_build_config(mask_min=-0.1)
    with pytest.raises(ValueError, match="must be in"):
        _valid_build_config(mask_max=1.1)


def test_model_build_config_rejects_unsorted_ltr_stages() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        _valid_build_config(grammar_ltr_stages=(16, 8, 32))


def test_model_build_config_rejects_non_positive_ltr_stage() -> None:
    with pytest.raises(ValueError, match="positive"):
        _valid_build_config(grammar_ltr_stages=(0, 8))


def test_model_build_config_rejects_bad_diffusion_length_buckets() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        _valid_build_config(diffusion_length_buckets=(64, 32, 128))


def test_model_build_config_accepts_valid_schedules_and_weights() -> None:
    cfg = _valid_build_config(
        recursive_depth_supervision_weights=(0.5, 1.0, 0.5),
        grammar_ltr_stages=(16, 32, 64),
        diffusion_length_buckets=(32, 64, 128),
    )
    assert cfg.recursive_depth_supervision_weights == (0.5, 1.0, 0.5)


# --------------------------------------------------------------------------- #
# TwoTowerConfig numeric gates (torch-backed module)
# --------------------------------------------------------------------------- #

torch = pytest.importorskip("torch")

from slm_training.models.twotower import TwoTowerConfig  # noqa: E402


def _valid_twotower_config(**overrides: object) -> TwoTowerConfig:
    defaults: dict[str, object] = {
        "d_model": 32,
        "n_heads": 2,
        "context_layers": 1,
        "denoiser_layers": 2,
    }
    defaults.update(overrides)
    return TwoTowerConfig(**defaults)  # type: ignore[arg-type]


# The six SLM-237 recursive-depth defects must now raise at config construction
# time rather than silently corrupting the loss.


def test_twotower_config_rejects_non_recursive_arch_with_depth_weights() -> None:
    with pytest.raises(ValueError, match="recursive"):
        _valid_twotower_config(
            denoiser_arch="stacked",
            recursive_steps=3,
            recursive_depth_supervision_weights=(1.0, 1.0, 1.0),
        )


def test_twotower_config_rejects_wrong_weight_length() -> None:
    with pytest.raises(ValueError, match="length"):
        _valid_twotower_config(
            denoiser_arch="shared_recursive",
            recursive_steps=3,
            recursive_depth_supervision_weights=(1.0, 1.0),
        )


def test_twotower_config_rejects_negative_depth_weight() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        _valid_twotower_config(
            denoiser_arch="shared_recursive",
            recursive_steps=2,
            recursive_depth_supervision_weights=(1.0, -0.5),
        )


def test_twotower_config_rejects_nan_depth_weight() -> None:
    with pytest.raises(ValueError, match="finite"):
        _valid_twotower_config(
            denoiser_arch="shared_recursive",
            recursive_steps=2,
            recursive_depth_supervision_weights=(1.0, float("nan")),
        )


def test_twotower_config_rejects_intermediate_only_with_r1() -> None:
    # R=1 has 0 eligible intermediate depths, so any non-empty weight tuple
    # violates the intermediate_only contract.
    with pytest.raises(ValueError, match="intermediate_only|length"):
        _valid_twotower_config(
            denoiser_arch="shared_recursive",
            recursive_steps=1,
            recursive_depth_supervision_weights=(1.0,),
            recursive_depth_aux_mode="intermediate_only",
        )


def test_twotower_config_rejects_invalid_aux_weight() -> None:
    with pytest.raises(ValueError, match="finite|negative"):
        _valid_twotower_config(
            denoiser_arch="shared_recursive",
            recursive_steps=2,
            recursive_depth_supervision_weights=(1.0, 1.0),
            recursive_depth_aux_weight=float("nan"),
        )
    with pytest.raises(ValueError, match="negative"):
        _valid_twotower_config(
            denoiser_arch="shared_recursive",
            recursive_steps=2,
            recursive_depth_supervision_weights=(1.0, 1.0),
            recursive_depth_aux_weight=-1.0,
        )


def test_twotower_config_accepts_valid_recursive_depth_setup() -> None:
    cfg = _valid_twotower_config(
        denoiser_arch="shared_recursive",
        recursive_steps=3,
        recursive_depth_supervision_weights=(0.5, 1.0, 0.5),
        recursive_depth_aux_mode="all_depths",
        recursive_depth_aux_weight=0.75,
    )
    assert cfg.recursive_depth_aux_mode == "all_depths"
    assert math.isclose(cfg.recursive_depth_aux_weight, 0.75)
