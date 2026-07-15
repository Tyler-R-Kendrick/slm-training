"""Regression: train source=all must not leak into disjoint test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.dsl import bridge_available, validate
from slm_training.harnesses.test_data import TestDataConfig, build_test_data
from slm_training.harnesses.train_data import TrainDataConfig, build_train_data

pytestmark = pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing",
)


def test_switchitem_null_description_allowed() -> None:
    src = 'root = Stack([s], "column", "m")\ns = SwitchItem(":s.label", null, "s")'
    program = validate(src)
    assert ":s.label" in program.placeholders


def test_train_all_and_test_fixtures_are_disjoint(tmp_path: Path) -> None:
    train = build_train_data(
        TrainDataConfig(
            seed_path=Path("src/slm_training/resources/train_seeds.jsonl"),
            rico_path=Path("src/slm_training/resources/rico/semantic_train.jsonl"),
            source="all",
            output_root=tmp_path / "train",
            version="v0",
            rico_limit=20,
            synthesizer="none",
        )
    )
    assert train["stats"]["error_count"] == 0 or train["stats"]["record_count"] > 0
    # Rebuild should succeed without fixture openui collisions.
    test = build_test_data(
        TestDataConfig(
            seed_path=Path("src/slm_training/resources/test_seeds.jsonl"),
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "test",
            version="v0",
            suites=("smoke", "held_out", "adversarial", "ood"),
            train_manifest=Path(train["output_dir"]) / "manifest.json",
            require_train_manifest=True,
        )
    )
    assert sum(test["stats"]["suite_counts"].values()) >= 4
