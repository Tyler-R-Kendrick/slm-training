"""Model-building harness tests (stub only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.dsl import bridge_available
from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.model_build import (
    ModelBuildConfig,
    StubModel,
    evaluate,
    train,
)
from slm_training.harnesses.test_data import TestDataConfig, build_test_data
from slm_training.harnesses.train_data import TrainDataConfig, build_train_data

pytestmark = pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)

HERO = 'root = Stack([hero], "column")\nhero_title = TextContent(":hero.title")\nhero_body = TextContent(":hero.body")\nhero = Card([hero_title, hero_body])'
CTA = 'root = Stack([cta])\ncta = Button(":cta.label")'


def _prepare_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    train_seeds = tmp_path / "train.jsonl"
    write_jsonl(
        train_seeds,
        [
            ExampleRecord(id="tr1", prompt="Hero", openui=HERO, split="train"),
            ExampleRecord(id="tr2", prompt="CTA", openui=CTA, split="train"),
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
    train_dir = Path(train_result["output_dir"])

    test_seeds = tmp_path / "test.jsonl"
    write_jsonl(
        test_seeds,
        [
            ExampleRecord(
                id="sm1",
                prompt="Hero",
                openui=HERO,
                split="smoke",
                meta={"suite": "smoke"},
            ),
            ExampleRecord(
                id="sm2",
                prompt="CTA",
                openui=CTA,
                split="smoke",
                meta={"suite": "smoke"},
            ),
        ],
    )
    test_result = build_test_data(
        TestDataConfig(
            seed_path=test_seeds,
            rico_path=None,
            source="fixture",
            output_root=tmp_path / "test_data",
            version="v0",
            suites=("smoke",),
            train_manifest=None,
            require_train_manifest=False,
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
        model_name="stub",
        eval_every=1,
    )
    summary = train(config)
    assert summary["steps"] == 2
    assert summary["eval_history"]
    assert summary["best_ship_score"] is not None
    assert (config.run_dir / "eval_history.jsonl").exists()
    assert (config.checkpoint_dir / "best_ship_score.pt").exists()
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
    model.memory["p"] = CTA
    path = tmp_path / "stub.json"
    model.save(path)
    loaded = StubModel()
    loaded.load(path)
    assert loaded.generate("p") == CTA
