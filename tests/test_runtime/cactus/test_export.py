"""Cactus bundle export smoke test."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slm_training.runtime.cactus import export_checkpoint_bundle
from slm_training.dsl.schema import ExampleRecord
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    "hero = Card([hero_title, hero_body])"
)


def test_export_cactus_bundle(tmp_path: Path) -> None:
    model = TwoTowerModel.from_records(
        [ExampleRecord(id="a", prompt="Hero", openui=HERO, split="train")],
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            context_backend="scratch",
        ),
        device="cpu",
    )
    ckpt = tmp_path / "model.pt"
    model.save(ckpt)
    out = export_checkpoint_bundle(ckpt, tmp_path / "bundle")
    assert (out / "model.pt").exists()
    assert (out / "manifest.json").exists()
