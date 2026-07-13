"""E54 / E35 honesty: grammar_diffusion must not read gold.placeholders."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.model_build.plugin import GenerationRequest
from slm_training.models.grammar_diffusion import (
    GrammarDiffusionConfig,
    GrammarDiffusionModel,
)

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    "hero = Card([hero_title, hero_body])"
)


def test_generate_ignores_gold_placeholders() -> None:
    records = [
        ExampleRecord(
            id="a",
            prompt="Build a hero.\nPlaceholders: :hero.title, :hero.body",
            openui=HERO,
            split="train",
            placeholders=[":hero.title", ":hero.body"],
        )
    ]
    model = GrammarDiffusionModel.from_records(
        records,
        config=GrammarDiffusionConfig(d_model=64, n_heads=4, context_layers=1, denoiser_layers=1),
        device="cpu",
    )
    # Poisoned gold inventory must not leak into decode when honest.
    poisoned = ExampleRecord(
        id="evil",
        prompt="Build a hero.\nPlaceholders: :hero.title, :hero.body",
        openui=HERO,
        split="smoke",
        placeholders=[":evil.secret", ":evil.leak"],
    )
    assert model.config.honest_slot_contract is True
    out = model.generate(poisoned.prompt, gold=poisoned)
    assert ":evil.secret" not in out
    assert ":evil.leak" not in out


def test_generate_batch_requests_fills_inventory_from_prompt() -> None:
    records = [
        ExampleRecord(
            id="a",
            prompt="CTA button",
            openui='root = Stack([cta])\ncta = Button(":cta.label")',
            split="train",
            placeholders=[":cta.label"],
        )
    ]
    model = GrammarDiffusionModel.from_records(
        records,
        config=GrammarDiffusionConfig(d_model=64, n_heads=4, context_layers=1, denoiser_layers=1),
        device="cpu",
    )
    texts = model.generate_batch_requests(
        [
            GenerationRequest(
                prompt="Make a CTA.\nPlaceholders: :cta.label",
                slot_contract=(),
            )
        ]
    )
    assert len(texts) == 1
    assert isinstance(texts[0], str)
