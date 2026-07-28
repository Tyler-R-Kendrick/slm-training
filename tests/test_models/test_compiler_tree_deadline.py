"""Compiler-tree decode must observe a cooperative decode deadline.

Regression coverage for
docs/design/decode-compiler-tree-deadline-swallow-fix.md: a deadline
TimeoutError must propagate out of build_completion_forest instead of being
swallowed as a fake grammar dead-end, and the compiler-tree per-position loop
must call check_decode_deadline() each step (matching the LTR-repair/MaskGIT
loops' existing coverage).
"""

from __future__ import annotations

import time

import pytest

from slm_training.data.contract import canonicalize_example_template_markers
from slm_training.dsl.grammar.fastpath.compiler_draft import build_completion_forest
from slm_training.dsl.grammar_capabilities import GrammarCapabilityAdapterV1
from slm_training.dsl.schema import ExampleRecord
from slm_training.models.decode_stats import clear_decode_deadline, set_decode_deadline
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel


def _model() -> TwoTowerModel:
    record = ExampleRecord(
        id="compiler",
        prompt="card",
        openui='root = Card([title])\ntitle = TextContent(":hero.title")\n',
        placeholders=[":hero.title"],
        split="train",
        source="fixture",
    )
    config = TwoTowerConfig(
        context_backend="scratch",
        output_tokenizer="lexer",
        compiler_decode_mode="tree",
        d_model=32,
        n_heads=2,
        context_layers=1,
        denoiser_layers=1,
        max_prompt_len=32,
        max_target_len=32,
        grammar_ltr_max_tokens=32,
        gen_steps=1,
        seed=0,
    )
    model = TwoTowerModel.from_records(
        [canonicalize_example_template_markers(record)], config=config, device="cpu"
    )
    model.eval()
    return model


def test_build_completion_forest_propagates_deadline_timeout(monkeypatch) -> None:
    tokenizer = DSLNativeTokenizer.build()
    prefix = [tokenizer.bos_id]

    def raise_timeout(self, request):
        raise TimeoutError("decode deadline exceeded")

    monkeypatch.setattr(GrammarCapabilityAdapterV1, "completion_domain", raise_timeout)
    with pytest.raises(TimeoutError, match="decode deadline exceeded"):
        build_completion_forest(tokenizer, prefix, remaining_tokens=8)


def test_build_completion_forest_still_fails_closed_on_other_errors(
    monkeypatch,
) -> None:
    tokenizer = DSLNativeTokenizer.build()
    prefix = [tokenizer.bos_id]

    def raise_value_error(self, request):
        raise ValueError("unrelated adapter failure")

    monkeypatch.setattr(
        GrammarCapabilityAdapterV1, "completion_domain", raise_value_error
    )
    forest = build_completion_forest(tokenizer, prefix, remaining_tokens=8)
    assert forest.coverage == "none"
    assert forest.paths == ()


def test_compiler_ltr_decode_one_checks_deadline_each_step() -> None:
    model = _model()
    ctx, ctx_pad = model._encode_context(["card"])
    set_decode_deadline(0.05)
    try:
        time.sleep(0.08)
        with pytest.raises(TimeoutError, match="decode deadline exceeded"):
            model._compiler_ltr_decode_one(
                ctx, ctx_pad, 24, mode="tree", slot_contract=None
            )
    finally:
        clear_decode_deadline()
