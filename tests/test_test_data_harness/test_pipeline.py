"""Testing-data harness tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.dsl import bridge_available
from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.test_data import TestDataConfig, build_test_data
from slm_training.harnesses.train_data import TrainDataConfig, build_train_data

pytestmark = pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd tools/openui_bridge && npm ci",
)


def test_build_test_data_suites(tmp_path: Path) -> None:
    seeds = tmp_path / "test_seeds.jsonl"
    write_jsonl(
        seeds,
        [
            ExampleRecord(
                id="smoke_1",
                prompt="Smoke",
                openui='root = Stack([cta])\ncta = Button(":cta.label")',
                placeholders=[":cta.label"],
                split="smoke",
                meta={"suite": "smoke"},
            ),
            ExampleRecord(
                id="held_1",
                prompt="Held",
                openui='root = Stack([blurb])\nblurb = Text(":page.blurb")',
                placeholders=[":page.blurb"],
                split="held_out",
                meta={"suite": "held_out"},
            ),
            ExampleRecord(
                id="adv_1",
                prompt="x",
                openui='root = Stack([fallback])\nfallback = Text(":fallback.text")',
                placeholders=[":fallback.text"],
                split="adversarial",
                meta={"suite": "adversarial"},
            ),
        ],
    )
    result = build_test_data(
        TestDataConfig(
            seed_path=seeds,
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "test_data",
            version="vtest",
            suites=("smoke", "held_out", "adversarial"),
            train_manifest=None,
            require_train_manifest=False,
        )
    )
    out_dir = Path(result["output_dir"])
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["suite_counts"]["smoke"] == 1
    assert (out_dir / "suites" / "smoke" / "records.jsonl").exists()


def test_leakage_detection_by_id(tmp_path: Path) -> None:
    train_seeds = tmp_path / "train.jsonl"
    write_jsonl(
        train_seeds,
        [
            ExampleRecord(
                id="shared_id",
                prompt="Train",
                openui='root = Stack([cta])\ncta = Button(":cta.label")',
                split="train",
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
            synthesizer="none",
        )
    )
    train_manifest = Path(train_result["output_dir"]) / "manifest.json"

    test_seeds = tmp_path / "test.jsonl"
    write_jsonl(
        test_seeds,
        [
            ExampleRecord(
                id="shared_id",
                prompt="Test different prompt",
                openui='root = Stack([blurb])\nblurb = Text(":page.blurb")',
                split="smoke",
                meta={"suite": "smoke"},
            )
        ],
    )
    with pytest.raises(ValueError, match="overlap"):
        build_test_data(
            TestDataConfig(
                seed_path=test_seeds,
                rico_path=None,
                source="fixture",
                output_root=tmp_path / "test_data",
                version="v0",
                suites=("smoke",),
                train_manifest=train_manifest,
            )
        )


def test_leakage_detection_by_openui(tmp_path: Path) -> None:
    openui = 'root = Stack([cta])\ncta = Button(":cta.label")'
    train_seeds = tmp_path / "train.jsonl"
    write_jsonl(
        train_seeds,
        [
            ExampleRecord(
                id="train_1",
                prompt="Train prompt",
                openui=openui,
                split="train",
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
            synthesizer="none",
        )
    )
    train_manifest = Path(train_result["output_dir"]) / "manifest.json"

    test_seeds = tmp_path / "test.jsonl"
    write_jsonl(
        test_seeds,
        [
            ExampleRecord(
                id="test_1",
                prompt="Completely different prompt text",
                openui=openui,
                split="smoke",
                meta={"suite": "smoke"},
            )
        ],
    )
    with pytest.raises(ValueError, match="openui"):
        build_test_data(
            TestDataConfig(
                seed_path=test_seeds,
                rico_path=None,
                source="fixture",
                output_root=tmp_path / "test_data",
                version="v0",
                suites=("smoke",),
                train_manifest=train_manifest,
            )
        )


def test_rico_train_and_test_are_disjoint(tmp_path: Path) -> None:
    train_rico = Path("fixtures/rico/semantic_train.jsonl")
    test_rico = Path("fixtures/rico/semantic_test.jsonl")
    if not train_rico.exists() or not test_rico.exists():
        pytest.skip("RICO fixtures missing")

    train_result = build_train_data(
        TrainDataConfig(
            seed_path=None,
            rico_path=train_rico,
            source="rico",
            output_root=tmp_path / "train_data",
            version="v0",
            synthesizer="none",
            rico_limit=20,
        )
    )
    test_result = build_test_data(
        TestDataConfig(
            seed_path=None,
            rico_path=test_rico,
            source="rico",
            output_root=tmp_path / "test_data",
            version="v0",
            suites=("smoke", "held_out"),
            train_manifest=Path(train_result["output_dir"]) / "manifest.json",
            rico_limit=15,
        )
    )
    assert test_result["stats"]["total_records"] >= 5
    train_ids = set(train_result["manifest"]["ids"])
    test_ids = set(test_result["manifest"]["ids"])
    assert train_ids.isdisjoint(test_ids)
