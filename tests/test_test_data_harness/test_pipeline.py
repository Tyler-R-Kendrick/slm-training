"""Testing-data harness tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.test_data import TestDataConfig, build_test_data
from slm_training.harnesses.train_data import TrainDataConfig, build_train_data


def test_build_test_data_suites(tmp_path: Path) -> None:
    seeds = tmp_path / "test_seeds.jsonl"
    write_jsonl(
        seeds,
        [
            ExampleRecord(
                id="smoke_1",
                prompt="Smoke",
                openui="root = Button(label=:cta.label)",
                placeholders=[":cta.label"],
                split="smoke",
                meta={"suite": "smoke"},
            ),
            ExampleRecord(
                id="held_1",
                prompt="Held",
                openui="root = Text(text=:page.blurb)",
                placeholders=[":page.blurb"],
                split="held_out",
                meta={"suite": "held_out"},
            ),
            ExampleRecord(
                id="adv_1",
                prompt="x",
                openui="root = Text(text=:fallback.text)",
                placeholders=[":fallback.text"],
                split="adversarial",
                meta={"suite": "adversarial"},
            ),
        ],
    )
    result = build_test_data(
        TestDataConfig(
            seed_path=seeds,
            output_root=tmp_path / "test_data",
            version="vtest",
            suites=("smoke", "held_out", "adversarial"),
        )
    )
    out_dir = Path(result["output_dir"])
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["suite_counts"]["smoke"] == 1
    assert (out_dir / "suites" / "smoke" / "records.jsonl").exists()


def test_leakage_detection(tmp_path: Path) -> None:
    train_seeds = tmp_path / "train.jsonl"
    write_jsonl(
        train_seeds,
        [
            ExampleRecord(
                id="shared_id",
                prompt="Train",
                openui="root = Button(label=:cta.label)",
                split="train",
            )
        ],
    )
    train_result = build_train_data(
        TrainDataConfig(
            seed_path=train_seeds,
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
                prompt="Test",
                openui="root = Button(label=:cta.label)",
                split="smoke",
                meta={"suite": "smoke"},
            )
        ],
    )
    with pytest.raises(ValueError, match="overlap"):
        build_test_data(
            TestDataConfig(
                seed_path=test_seeds,
                output_root=tmp_path / "test_data",
                version="v0",
                suites=("smoke",),
                train_manifest=train_manifest,
            )
        )
