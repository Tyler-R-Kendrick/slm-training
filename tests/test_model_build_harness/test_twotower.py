"""Tokenizer + TwoTower model tests."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl import bridge_available
from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.model_build import ModelBuildConfig, evaluate, train
from slm_training.harnesses.test_data import TestDataConfig, build_test_data
from slm_training.harnesses.train_data import TrainDataConfig, build_train_data
from slm_training.models.tokenizer import OpenUITokenizer, tokenize_text
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

pytestmark_bridge = pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd tools/openui_bridge && npm ci",
)

HERO = 'root = Stack([hero], "column")\nhero_title = TextContent(":hero.title")\nhero_body = TextContent(":hero.body")\nhero = Card([hero_title, hero_body])'
CTA = 'root = Stack([cta])\ncta = Button(":cta.label")'


def test_tokenize_preserves_placeholders_and_whitespace() -> None:
    text = 'hero = Card(":hero.title", ":hero.body")\n'
    tokens = tokenize_text(text)
    assert ":" in tokens
    assert "hero" in tokens
    assert "title" in tokens
    assert "body" in tokens
    assert "\n" in tokens
    assert "Card" in tokens


def test_tokenizer_roundtrip() -> None:
    tok = OpenUITokenizer.build([HERO, CTA, "Hero card layout"])
    encoded = tok.encode(HERO)
    assert encoded[0] == tok.bos_id
    assert encoded[-1] == tok.eos_id
    decoded = tok.decode(encoded)
    assert decoded == HERO


def test_tokenizer_save_load(tmp_path: Path) -> None:
    tok = OpenUITokenizer.build([HERO])
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = OpenUITokenizer.load(path)
    assert loaded.encode(HERO) == tok.encode(HERO)


def test_twotower_training_loss_decreases() -> None:
    records = [
        ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train"),
        ExampleRecord(id="b", prompt="CTA", openui=CTA, split="train"),
    ]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=64,
            n_heads=4,
            context_layers=1,
            denoiser_layers=2,
            gen_steps=4,
            seed=0,
        ),
        device="cpu",
    )
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=3e-3)
    losses: list[float] = []
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        loss = model.training_loss(records)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    assert losses[-1] < 2.0


def test_twotower_save_load_generate(tmp_path: Path) -> None:
    records = [
        ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train"),
    ]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(d_model=64, n_heads=4, context_layers=1, denoiser_layers=2),
        device="cpu",
    )
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=3e-3)
    for _ in range(60):
        opt.zero_grad(set_to_none=True)
        model.training_loss(records).backward()
        opt.step()

    ckpt = tmp_path / "model.pt"
    model.save(ckpt)
    assert ckpt.with_suffix(".tokenizer.json").exists()
    loaded = TwoTowerModel.from_checkpoint(ckpt, device="cpu")
    pred = loaded.generate("Hero")
    assert "Stack" in pred or "Card" in pred or "root" in pred


@pytestmark_bridge
def test_twotower_train_eval_overfit(tmp_path: Path) -> None:
    train_seeds = tmp_path / "train.jsonl"
    write_jsonl(
        train_seeds,
        [
            ExampleRecord(
                id="tr1",
                prompt="Hero",
                openui=HERO,
                split="train",
                placeholders=[":hero.title", ":hero.body"],
            ),
            ExampleRecord(
                id="tr2",
                prompt="CTA",
                openui=CTA,
                split="train",
                placeholders=[":cta.label"],
            ),
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
                placeholders=[":hero.title", ":hero.body"],
            ),
            ExampleRecord(
                id="sm2",
                prompt="CTA",
                openui=CTA,
                split="smoke",
                meta={"suite": "smoke"},
                placeholders=[":cta.label"],
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
    test_dir = Path(test_result["output_dir"])

    config = ModelBuildConfig(
        train_dir=train_dir,
        test_dir=test_dir,
        suite="smoke",
        run_root=tmp_path / "runs",
        run_id="twotower_overfit",
        steps=120,
        batch_size=2,
        lr=3e-3,
        seed=0,
        model_name="twotower",
        d_model=64,
        n_heads=4,
        context_layers=1,
        denoiser_layers=2,
        gen_steps=6,
        context_backend="scratch",
        freeze_context=False,
        slot_contract_in_context=True,
        slot_contract_constrained_decode=True,
    )
    summary = train(config)
    assert summary["steps"] == 120
    assert summary["last_loss"] < 3.0
    ckpt = Path(summary["checkpoint"])
    assert ckpt.exists()
    assert ckpt.with_suffix(".tokenizer.json").exists()

    metrics = evaluate(config, checkpoint=ckpt)
    assert metrics["n"] == 2
    # Overfit smoke: should parse at least one; ideally both
    assert metrics["parse_rate"] >= 0.5
    assert metrics["placeholder_fidelity"] >= 0.5
