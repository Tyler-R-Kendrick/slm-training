"""Model-building harness tests (stub only)."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.model_build import (
    ModelBuildConfig,
    StubModel,
    evaluate,
    train,
)
from slm_training.harnesses.test_data import TestDataConfig, build_test_data
from slm_training.harnesses.train_data import TrainDataConfig, build_train_data


def _prepare_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    train_seeds = tmp_path / "train.jsonl"
    write_jsonl(
        train_seeds,
        [
            ExampleRecord(
                id="tr1",
                prompt="Hero",
                openui=(
                    'root = Stack(direction="vertical", children=hero)\n'
                    "hero = Card(title=:hero.title, body=:hero.body)"
                ),
                split="train",
            ),
            ExampleRecord(
                id="tr2",
                prompt="CTA",
                openui="root = Button(label=:cta.label)",
                split="train",
            ),
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
    train_dir = Path(train_result["output_dir"])

    test_seeds = tmp_path / "test.jsonl"
    write_jsonl(
        test_seeds,
        [
            ExampleRecord(
                id="sm1",
                prompt="Hero",
                openui=(
                    'root = Stack(direction="vertical", children=hero)\n'
                    "hero = Card(title=:hero.title, body=:hero.body)"
                ),
                split="smoke",
                meta={"suite": "smoke"},
            ),
            ExampleRecord(
                id="sm2",
                prompt="CTA",
                openui="root = Button(label=:cta.label)",
                split="smoke",
                meta={"suite": "smoke"},
            ),
        ],
    )
    test_result = build_test_data(
        TestDataConfig(
            seed_path=test_seeds,
            output_root=tmp_path / "test_data",
            version="v0",
            suites=("smoke",),
            train_manifest=train_dir / "manifest.json",
        )
    )
    return train_dir, Path(test_result["output_dir"])


def test_train_and_eval_stub(tmp_path: Path) -> None:
    train_dir, test_dir = _prepare_artifacts(tmp_path)
    config = ModelBuildConfig(
        train_dir=train_dir,
        test_dir=test_dir,
        suite="smoke",
        run_root=tmp_path / "runs",
        run_id="test_run",
        steps=2,
        batch_size=2,
    )
    summary = train(config)
    assert summary["steps"] == 2
    ckpt = Path(summary["checkpoint"])
    assert ckpt.exists()

    metrics = evaluate(config, checkpoint=ckpt)
    assert metrics["n"] == 2
    assert "parse_rate" in metrics
    assert metrics["parse_rate"] == 1.0
    assert (config.run_dir / "eval.json").exists()
    assert (config.run_dir / "metrics.jsonl").exists()


def test_stub_save_load(tmp_path: Path) -> None:
    model = StubModel(seed=1)
    model.memory["p"] = "root = Button(label=:cta.label)"
    path = tmp_path / "stub.json"
    model.save(path)
    loaded = StubModel()
    loaded.load(path)
    assert loaded.generate("p") == "root = Button(label=:cta.label)"
