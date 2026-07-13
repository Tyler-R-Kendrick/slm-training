"""Tests for curriculum mix, GRPO-lite, preference, and telemetry."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.preference import (
    PreferencePair,
    build_pairs_from_candidates,
)
from slm_training.preference.train import dpo_loss
from slm_training.quality import (
    apply_curriculum_tags,
    curriculum_mix_weights,
    sample_curriculum_batch,
    soft_corrupt_openui,
    strip_adv_placeholders,
)
from slm_training.rl import grpo_loss_for_group, structure_reward, train_grpo
from slm_training.rl import GRPOConfig
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel
from slm_training.telemetry import CycleTelemetry, bind_telemetry, timed


HERO = 'root = Stack([hero], "column")\nhero_title = TextContent(":hero.title")\nhero = Card([hero_title])\n'


def test_strip_adv_placeholders() -> None:
    assert '":item.' in strip_adv_placeholders('x = TextContent(":adv.title")')


def test_curriculum_mix_keeps_b_late() -> None:
    w = curriculum_mix_weights(90, 100)
    assert w["B"] >= 0.30
    assert abs(sum(w.values()) - 1.0) < 1e-6


def test_sample_curriculum_batch_mix() -> None:
    import random

    records = apply_curriculum_tags(
        [
            ExampleRecord(id="rico_1", prompt="a", openui=HERO, source="rico"),
            ExampleRecord(id="fx_1", prompt="b", openui=HERO, source="fixture"),
            ExampleRecord(
                id="adversarial_1",
                prompt="c",
                openui='root = TextContent(":adv.x")\n',
                source="fixture",
                meta={"suite": "adversarial"},
            ),
        ]
    )
    # Non-C records should not keep :adv.
    assert all(
        ":adv." not in r.openui
        for r in records
        if (r.meta or {}).get("curriculum") != "C"
    )
    rng = random.Random(0)
    batch = sample_curriculum_batch(
        records, batch_size=6, step=80, total_steps=100, rng=rng, mix=True
    )
    assert len(batch) == 6


def test_dpo_loss_runs() -> None:
    records = [
        ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train"),
    ]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=64, n_heads=2, context_layers=1, denoiser_layers=2, seed=0
        ),
        device="cpu",
    )
    pair = PreferencePair(
        prompt="Hero",
        chosen=HERO,
        rejected='root = TextContent(":wrong.x")\n',
    )
    loss = dpo_loss(model, pair, beta=0.1)
    assert float(loss.detach()) == float(loss.detach())


def test_build_pairs_prefer_valid() -> None:
    gold = ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")
    pair = build_pairs_from_candidates(
        "Hero",
        [HERO, soft_corrupt_openui(HERO), "root = Broken()"],
        gold=gold,
        design_md=None,
        prefer_valid_rejects=True,
    )
    assert pair is not None
    assert "Broken" not in pair.rejected


def test_grpo_smoke(tmp_path: Path) -> None:
    records = [
        ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train"),
        ExampleRecord(
            id="b",
            prompt="CTA",
            openui='root = Button(":cta.label")\n',
            split="train",
        ),
    ]
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            d_model=64,
            n_heads=2,
            context_layers=1,
            denoiser_layers=2,
            grammar_ltr_primary=True,
            seed=0,
        ),
        device="cpu",
    )
    summary = train_grpo(
        model,
        records,
        config=GRPOConfig(steps=2, group_size=2, batch_prompts=1, lr=1e-3),
        out_dir=tmp_path / "rl",
    )
    assert summary["steps"] == 2
    assert Path(summary["checkpoint"]).exists()
    assert "bottlenecks" in summary["telemetry"]


def test_structure_reward_ignores_design_md_style() -> None:
    r = structure_reward(HERO, gold=None)
    assert 0.0 <= r <= 1.0


def test_telemetry_spans(tmp_path: Path) -> None:
    tel = CycleTelemetry(enabled=True, meta={"t": 1})
    with bind_telemetry(tel):
        with timed("a"):
            pass
        with timed("b"):
            pass
        with timed("a"):
            pass
    summary = tel.summary()
    assert summary["spans"]["a"]["count"] == 2
    assert summary["bottlenecks"]
    path = tel.write(tmp_path / "tel.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "total_ms" in data
