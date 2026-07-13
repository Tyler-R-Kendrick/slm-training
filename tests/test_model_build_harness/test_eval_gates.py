"""Eval-driven gates and scoreboard helpers."""

from __future__ import annotations

from pathlib import Path

from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.model_build import ModelBuildConfig
from slm_training.harnesses.model_build.eval_runner import (
    evaluate,
    evaluate_suites,
    structural_similarity,
)
from slm_training.harnesses.model_build.plugin import StubModel
from slm_training.preference import composite_reward


def test_structural_similarity_identical() -> None:
    src = 'root = Stack([a])\na = Button(":x")'
    assert structural_similarity(src, src) == 1.0


def test_structural_similarity_partial() -> None:
    gold = 'root = Stack([a, b])\na = Button(":a")\nb = Button(":b")'
    pred = 'root = Stack([a])\na = Button(":a")'
    score = structural_similarity(pred, gold)
    assert 0.0 < score < 1.0


def test_reward_score_on_valid_pred() -> None:
    gold = ExampleRecord(
        id="t",
        prompt="button",
        openui='root = Stack([cta])\ncta = Button(":cta")',
        placeholders=[":cta"],
    )
    score = composite_reward(gold.openui, gold=gold)
    assert score > 0.5


def test_evaluate_suites_scoreboard(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "test"
    train_dir.mkdir()
    (test_dir / "suites").mkdir(parents=True)
    hero = (
        'root = Stack([hero], "column")\n'
        'hero_title = TextContent(":hero.title")\n'
        'hero_body = TextContent(":hero.body")\n'
        "hero = Card([hero_title, hero_body])"
    )
    records = [
        ExampleRecord(id="a", prompt="Hero", openui=hero, split="train"),
    ]
    write_jsonl(train_dir / "records.jsonl", records)
    write_jsonl(
        test_dir / "suites" / "smoke.jsonl",
        [
            ExampleRecord(
                id="s1",
                prompt="Hero",
                openui=hero,
                split="smoke",
                meta={"suite": "smoke"},
            )
        ],
    )

    config = ModelBuildConfig(
        train_dir=train_dir,
        test_dir=test_dir,
        suite="smoke",
        run_root=tmp_path / "runs",
        run_id="gates",
        model_name="stub",
        noise_rate=0.0,
    )
    model = StubModel(noise_rate=0.0, seed=0)
    model.forward(records)
    metrics = evaluate(config, model=model)
    assert "reward_score" in metrics
    assert metrics["n"] == 1
    assert metrics["parse_rate"] == 1.0

    board = evaluate_suites(config, ["smoke"])
    assert "suites" in board
    assert (tmp_path / "runs" / "gates" / "scoreboard.json").exists()
