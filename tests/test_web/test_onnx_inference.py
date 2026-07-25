"""Deployment inference smoke tests."""

from __future__ import annotations

import pytest

pytest.importorskip("onnxruntime")

from slm_training.dsl.parser import validate
from slm_training.models.onnx_inference import (
    GrammarCertificationError,
    OnnxTwoTowerModel,
)
from slm_training.models.paths import PLAYGROUND_DEMO_CHECKPOINT

CANNED = {
    'root = Button(":cta.label")',
    'root = TextContent(":hero.title")',
    'root = Card([title])\ntitle = TextContent(":hero.title")',
}


def test_onnx_constrained_decode_is_certified_or_fails_closed() -> None:
    """I6: the serving backend never hands back uncertified OpenUI."""
    model = OnnxTwoTowerModel.from_checkpoint(PLAYGROUND_DEMO_CHECKPOINT)
    try:
        generated = model.generate(
            "Hero card with a title and body", grammar_constrained=True
        )
    except GrammarCertificationError:
        # Fail-closed is the contract: the web harness owns retry and the
        # browser fallback, and a failed decode must be visible as a failure.
        return

    assert generated.strip()
    assert "root" in generated
    # A real decode, not a canned template dressed up as a model attempt.
    assert generated.strip() not in CANNED
    program = validate(generated)
    assert program.root is not None


def test_onnx_forced_singletons_commit_without_a_forward() -> None:
    """I2: scope-proven singletons are committed with no denoiser run."""
    model = OnnxTwoTowerModel.from_checkpoint(PLAYGROUND_DEMO_CHECKPOINT)
    try:
        model.generate("Hero card with a title and body", grammar_constrained=True)
    except GrammarCertificationError:
        pass

    # The OpenUI grammar always forces at least one deterministic lexeme, and
    # each one is emitted outside the forward budget.
    assert model.last_forced_tokens_without_forward > 0


def test_onnx_forced_tokens_cost_zero_denoiser_runs() -> None:
    """Forced tokens are outside the forward budget, not merely cheaper."""
    model = OnnxTwoTowerModel.from_checkpoint(PLAYGROUND_DEMO_CHECKPOINT)
    session = model.denoiser_session
    runs = 0

    def _counting_run(*args: object, **kwargs: object) -> object:
        nonlocal runs
        runs += 1
        return session.run(*args, **kwargs)

    model.denoiser_session = type("Counting", (), {"run": staticmethod(_counting_run)})()
    try:
        model.generate("Hero card with a title and body", grammar_constrained=True)
    except GrammarCertificationError:
        pass

    assert model.last_forced_tokens_without_forward > 0
    assert runs == model.last_forwards_count
