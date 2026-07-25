"""Deployment inference smoke tests."""

from __future__ import annotations

import pytest

pytest.importorskip("onnxruntime")

from slm_training.dsl.parser import validate
from slm_training.models.onnx_inference import OnnxTwoTowerModel
from slm_training.models.paths import PLAYGROUND_DEMO_CHECKPOINT

CANNED = {
    'root = Button(":cta.label")',
    'root = TextContent(":hero.title")',
    'root = Card([title])\ntitle = TextContent(":hero.title")',
}


def test_onnx_constrained_decode_returns_certified_openui() -> None:
    """I6: the serving backend never hands back uncertified OpenUI."""
    model = OnnxTwoTowerModel.from_checkpoint(PLAYGROUND_DEMO_CHECKPOINT)
    generated = model.generate(
        "Hero card with a title and body", grammar_constrained=True
    )

    assert generated.strip()
    program = validate(generated)
    assert program.root is not None

    evidence = model.consume_generation_evidence()
    assert evidence and evidence[0]["grammar_constrained"] is True
    if not evidence[0]["fallback_used"]:
        # A real decode, not a canned template dressed up as a model attempt.
        assert generated.strip() not in CANNED


def test_onnx_refuses_an_unconstrained_request() -> None:
    """I6: unconstrained generation is not a mode this backend offers."""
    model = OnnxTwoTowerModel.from_checkpoint(PLAYGROUND_DEMO_CHECKPOINT)
    with pytest.raises(ValueError, match="unsafe"):
        model.generate("Hero card", grammar_constrained=False)


def test_onnx_forced_singletons_commit_without_a_forward() -> None:
    """I2: scope-proven singletons are committed with no denoiser run."""
    model = OnnxTwoTowerModel.from_checkpoint(PLAYGROUND_DEMO_CHECKPOINT)
    model.generate("Hero card with a title and body", grammar_constrained=True)

    evidence = model.consume_generation_evidence()[0]
    # The OpenUI grammar always forces at least one deterministic lexeme, and
    # each one is emitted outside the forward budget.
    assert int(evidence["singleton_bypasses"]) > 0


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
    model.generate("Hero card with a title and body", grammar_constrained=True)

    evidence = model.consume_generation_evidence()[0]
    assert int(evidence["singleton_bypasses"]) > 0
    assert runs == int(evidence["model_forwards"])
