"""Deployment inference smoke tests."""

from __future__ import annotations

import pytest

pytest.importorskip("onnxruntime")

from slm_training.dsl.parser import validate
from slm_training.models.onnx_inference import OnnxTwoTowerModel
from slm_training.models.paths import PLAYGROUND_DEMO_CHECKPOINT


def test_onnx_playground_checkpoint_returns_real_decode_without_canned_fallback() -> None:
    model = OnnxTwoTowerModel.from_checkpoint(PLAYGROUND_DEMO_CHECKPOINT)
    generated = model.generate(
        "Hero card with a title and body", grammar_constrained=True
    )

    assert generated.strip()
    assert "root" in generated
    canned = {
        'root = Button(":cta.label")',
        'root = TextContent(":hero.title")',
        'root = Card([title])\ntitle = TextContent(":hero.title")',
    }
    assert generated.strip() not in canned
    try:
        program = validate(generated)
    except Exception:
        program = None
    if program is not None:
        assert program.root is not None
