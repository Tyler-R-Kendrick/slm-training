"""Curation-profile resolution tests (strict-by-default, permissive escape)."""

from __future__ import annotations

import pytest

from slm_training.harnesses.train_data import (
    PROFILES,
    TrainDataConfig,
    resolve_profile,
)


def test_strict_profile_is_default_and_fills_curation_knobs() -> None:
    config = resolve_profile(TrainDataConfig())
    assert config.profile == "strict"
    assert config.fuzzy_dedup is True
    assert config.semantic_cluster_cap == 8
    assert config.min_verification_tier == "Bronze"
    assert config.max_records_per_parent == 6
    assert config.sanitize_mode == "enforce"


def test_explicit_values_survive_profile_resolution() -> None:
    config = resolve_profile(
        TrainDataConfig(
            semantic_cluster_cap=3,
            min_verification_tier="Silver",
            max_records_per_parent=2,
        )
    )
    # Untouched knob inherits from the profile...
    assert config.fuzzy_dedup is True
    # ...while explicit choices always win.
    assert config.semantic_cluster_cap == 3
    assert config.min_verification_tier == "Silver"
    assert config.max_records_per_parent == 2


def test_permissive_profile_keeps_legacy_defaults() -> None:
    config = resolve_profile(TrainDataConfig(profile="permissive"))
    assert config.fuzzy_dedup is False
    assert config.semantic_cluster_cap is None
    assert config.min_verification_tier is None
    assert config.max_records_per_parent is None
    assert config.sanitize_mode is None  # resolves to "off" at build time


def test_unknown_profile_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown train-data profile"):
        resolve_profile(TrainDataConfig(profile="yolo"))


def test_every_profile_knob_is_a_config_field() -> None:
    field_names = set(TrainDataConfig.__dataclass_fields__)
    for name, overrides in PROFILES.items():
        unknown = set(overrides) - field_names
        assert not unknown, f"profile {name} references unknown fields: {unknown}"
