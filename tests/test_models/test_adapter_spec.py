"""Torch-free tests for the TwoTower adapter spec (LDI2-01 / SLM-123)."""

from __future__ import annotations

import pytest

from slm_training.models.adapters import TwoTowerAdapterSpec


def _spec(**overrides) -> TwoTowerAdapterSpec:
    base = dict(
        method="low_rank",
        rank=4,
        alpha=8.0,
        dropout=0.0,
        target_modules=("attn_q", "attn_v"),
        base_compatibility_fingerprint="base-fp",
        base_checkpoint_sha="ckpt-sha",
        tokenizer_sha="tok-sha",
    )
    base.update(overrides)
    return TwoTowerAdapterSpec(**base)


def test_spec_round_trips_through_dict() -> None:
    spec = _spec(target_layer_indices=(0, 2), include_output_head=True)
    restored = TwoTowerAdapterSpec.from_dict(spec.to_dict())
    assert restored == spec
    assert restored.scaling == 8.0 / 4


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"rank": 0}, "rank must be positive"),
        ({"alpha": 0.0}, "alpha must be positive"),
        ({"dropout": 1.0}, r"dropout must be in \[0, 1\)"),
        ({"method": "dora"}, "only method='low_rank'"),
        ({"train_bias": "all"}, "only train_bias='none'"),
        ({"target_modules": ()}, "must not be empty"),
        ({"target_modules": ("attn_q", "attn_q")}, "must be unique"),
        ({"base_checkpoint_sha": ""}, "must be non-empty"),
    ],
)
def test_spec_validation_rejects_bad_configs(overrides, match) -> None:
    with pytest.raises(ValueError, match=match):
        _spec(**overrides)


def test_from_dict_rejects_unknown_fields() -> None:
    data = _spec().to_dict()
    data["mystery"] = 1
    with pytest.raises(ValueError, match="unknown adapter spec fields"):
        TwoTowerAdapterSpec.from_dict(data)
