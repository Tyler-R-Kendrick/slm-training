"""Tests for train/test structural disjointness."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.dsl import bridge_available
from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.test_data import TestDataConfig, build_test_data
from slm_training.harnesses.train_data import TrainDataConfig, build_train_data

pytestmark = pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)


def test_train_excludes_test_fixture_structures(tmp_path: Path) -> None:
    """Synthesized train layouts must not collide with test fixture structures."""
    train_seeds = tmp_path / "train_seeds.jsonl"
    test_seeds = tmp_path / "test_seeds.jsonl"
    tabs_train = (
        'root = Stack([tabs], "column")\n'
        'body1 = TextContent(":tab.one.body")\n'
        'body2 = TextContent(":tab.two.body")\n'
        'i1 = TabItem("$0", ":tab.one.trigger", [body1])\n'
        'i2 = TabItem("$1", ":tab.two.trigger", [body2])\n'
        "tabs = Tabs([i1, i2])"
    )
    tabs_test = (
        'root = Stack([panel], "column")\n'
        'overview = TextContent(":held.tabs.overview")\n'
        'details = TextContent(":held.tabs.details")\n'
        'tab1 = TabItem("$0", ":held.tabs.tab1", [overview])\n'
        'tab2 = TabItem("$1", ":held.tabs.tab2", [details])\n'
        "panel = Tabs([tab1, tab2])"
    )
    write_jsonl(
        train_seeds,
        [
            ExampleRecord(
                id="train_tabs_01",
                prompt="Two-tab layout",
                openui=tabs_train,
                placeholders=[
                    ":tab.one.body",
                    ":tab.two.body",
                    ":tab.one.trigger",
                    ":tab.two.trigger",
                ],
                split="train",
            )
        ],
    )
    write_jsonl(
        test_seeds,
        [
            ExampleRecord(
                id="held_out_tabs_01",
                prompt="Two-tab panel",
                openui=tabs_test,
                placeholders=[
                    ":held.tabs.overview",
                    ":held.tabs.details",
                    ":held.tabs.tab1",
                    ":held.tabs.tab2",
                ],
                split="held_out",
                meta={"suite": "held_out"},
            )
        ],
    )

    train_result = build_train_data(
        TrainDataConfig(
            seed_path=train_seeds,
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "train_data",
            version="v0",
            synthesizer="template",
            test_seed_path=test_seeds,
        )
    )
    train_ids = set(train_result["manifest"]["ids"])
    assert "train_tabs_01" not in train_ids
    assert train_result["stats"]["structure_reserved_rejected"] >= 1

    test_result = build_test_data(
        TestDataConfig(
            seed_path=test_seeds,
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "test_data",
            version="v0",
            suites=("held_out",),
            train_manifest=Path(train_result["output_dir"]) / "manifest.json",
        )
    )
    assert test_result["stats"]["suite_counts"]["held_out"] == 1
