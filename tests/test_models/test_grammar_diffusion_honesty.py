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
    'hero_title = TextContent(":slot_0")\n'
    'hero_body = TextContent(":slot_1")\n'
    "hero = Card([hero_title, hero_body])"
)


def test_generate_ignores_gold_placeholders() -> None:
    records = [
        ExampleRecord(
            id="a",
            prompt="Build a hero.\nPlaceholders: :slot_0, :slot_1",
            openui=HERO,
            split="train",
            placeholders=[":slot_0", ":slot_1"],
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
        prompt="Build a hero.\nPlaceholders: :slot_0, :slot_1",
        openui=HERO,
        split="smoke",
        placeholders=[":slot_2", ":slot_3"],
    )
    assert model.config.honest_slot_contract is True
    out = model.generate(poisoned.prompt, gold=poisoned)
    assert ":slot_2" not in out
    assert ":slot_3" not in out


def test_generate_batch_requests_fills_inventory_from_prompt() -> None:
    records = [
        ExampleRecord(
            id="a",
            prompt="CTA button",
            openui='root = Stack([cta])\ncta = Button(":slot_0")',
            split="train",
            placeholders=[":slot_0"],
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
                prompt="Make a CTA.",
                slot_contract=(":slot_0",),
            )
        ]
    )
    assert len(texts) == 1
    assert isinstance(texts[0], str)
