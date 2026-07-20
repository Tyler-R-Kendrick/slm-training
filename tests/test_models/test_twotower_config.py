"""Tests for TwoTowerConfig defaults."""

from __future__ import annotations

import pytest

from slm_training.models.twotower import TwoTowerConfig


def test_action_embedding_init_defaults_to_none() -> None:
    cfg = TwoTowerConfig()
    assert cfg.action_embedding_init == "none"
    assert cfg.action_embedding_train == "frozen"


def test_legal_margin_mode_defaults_to_none() -> None:
    cfg = TwoTowerConfig()
    assert cfg.legal_margin_mode == "none"
    assert cfg.targeted_margin_manifest is None
    assert cfg.targeted_margin_value == pytest.approx(1.0)
    assert cfg.targeted_margin_family_weights == ()


def test_pointer_mode_defaults_to_legacy_tokens() -> None:
    cfg = TwoTowerConfig()
    assert cfg.pointer_mode == "legacy_tokens"
    assert cfg.pointer_candidate_source == "structured_contract"
    assert cfg.pointer_hidden_dim == 256
    assert cfg.pointer_heads == 4
    assert cfg.pointer_temperature == pytest.approx(1.0)
    assert cfg.pointer_dropout == pytest.approx(0.0)
