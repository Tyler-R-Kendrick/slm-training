"""Tests for TwoTowerConfig defaults."""

from __future__ import annotations

from slm_training.models.twotower import TwoTowerConfig


def test_action_embedding_init_defaults_to_none() -> None:
    cfg = TwoTowerConfig()
    assert cfg.action_embedding_init == "none"
    assert cfg.action_embedding_train == "frozen"
