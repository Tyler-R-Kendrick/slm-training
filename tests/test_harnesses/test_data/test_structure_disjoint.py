"""Tests for train/test structural disjointness."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.data.leakage import (
    fingerprint_openui_structure,
    load_reserved_test_structure_fingerprints,
)
from slm_training.dsl import bridge_available
from slm_training.dsl.schema import ExampleRecord, load_jsonl, write_jsonl
from slm_training.harnesses.test_data import TestDataConfig, build_test_data
from slm_training.harnesses.train_data import TrainDataConfig, build_train_data
from slm_training.harnesses.train_data.sanitize import (
    SanitizeOptions,
    sanitize_openui,
    sanitized_reserved_structures,
)

pytestmark = pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)


def test_committed_train_fixtures_are_disjoint_after_strict_sanitize() -> None:
    """Source fixtures must not rely on the downstream firewall for isolation."""
    train_path = Path("src/slm_training/resources/train_seeds.jsonl")
    test_path = Path("src/slm_training/resources/test_seeds.jsonl")
    options = SanitizeOptions(mode="enforce")
    reserved = load_reserved_test_structure_fingerprints(test_path)
    reserved |= sanitized_reserved_structures(test_path, options)

    collisions = []
    for record in load_jsonl(train_path):
        outcome = sanitize_openui(
            record.openui,
            prompt=record.prompt,
            options=options,
        )
        if fingerprint_openui_structure(outcome.openui) in reserved:
            collisions.append(record.id)

    assert collisions == []


def test_train_excludes_test_fixture_structures(tmp_path: Path) -> None:
    """Synthesized train layouts must not collide with test fixture structures."""
    train_seeds = tmp_path / "train_seeds.jsonl"
    test_seeds = tmp_path / "test_seeds.jsonl"
    tabs_train = (
        'root = Stack([tabs], "column")\n'
        'body1 = TextContent(":tab.one.body")\n'
        'body2 = TextContent(":tab.two.body")\n'
        'i1 = TabItem("one", ":tab.one.trigger", [body1])\n'
        'i2 = TabItem("two", ":tab.two.trigger", [body2])\n'
        "tabs = Tabs([i1, i2])"
    )
    tabs_test = (
        'root = Stack([panel], "column")\n'
        'overview = TextContent(":held.tabs.overview")\n'
        'details = TextContent(":held.tabs.details")\n'
        'tab1 = TabItem("one", ":held.tabs.tab1", [overview])\n'
        'tab2 = TabItem("two", ":held.tabs.tab2", [details])\n'
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
