"""Regression tests for the compiler-tree decode-deadline swallow finding
(docs/design/decode-compiler-tree-deadline-swallow-finding.md).

That finding showed two related gaps in the compiler-tree constrained-decode
path:

1. ``build_completion_forest`` (``compiler_draft.py``) wrapped its call into
   ``GrammarCapabilityAdapterV1.completion_domain`` in a bare
   ``except Exception``, so a cooperative-deadline ``TimeoutError`` raised
   inside that call (e.g. by the eval harness's SIGALRM re-fire landing
   there) was silently converted into an empty ``CompletionForest`` --
   recorded by callers as a legitimate grammar dead-end
   (``empty_completion_forest``) instead of a timeout.
2. ``_compiler_ltr_decode_one`` (``twotower.py``) never called
   ``check_decode_deadline()`` inside its per-position loop, unlike the
   LTR-repair and MaskGIT loops, which both got that cooperative check in
   PR #1167.

These tests prove both gaps are closed: a deadline ``TimeoutError`` now
propagates out of ``build_completion_forest`` instead of being masqueraded
as ``coverage="none"``, and the compiler-tree per-position loop now raises
``TimeoutError`` once the cooperative deadline has elapsed, exactly like its
sibling decode loops (I3: the constrained-decode safety net must fail
closed on a real timeout, not misreport it as "no legal continuation").
"""

from __future__ import annotations

import time

import pytest

from slm_training.dsl.grammar.fastpath.compiler_draft import build_completion_forest
from slm_training.models.decode_stats import clear_decode_deadline, set_decode_deadline
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

PREFIX = "root = Card([b1,"


def test_build_completion_forest_propagates_deadline_timeout(monkeypatch) -> None:
    """A ``TimeoutError`` raised inside ``completion_domain`` must propagate,
    not be caught by the bare ``except Exception`` and reported as an empty
    forest."""
    from slm_training.dsl import grammar_capabilities

    def _raise_timeout(self, request):
        raise TimeoutError("decode deadline exceeded")

    monkeypatch.setattr(
        grammar_capabilities.GrammarCapabilityAdapterV1,
        "completion_domain",
        _raise_timeout,
    )

    tok = DSLNativeTokenizer.build()
    prefix_ids = [tok.bos_id, *tok.encode(PREFIX, add_special=False)]

    with pytest.raises(TimeoutError, match="decode deadline exceeded"):
        build_completion_forest(
            tok, prefix_ids, remaining_tokens=24, max_path_tokens=8, min_content=0
        )


def test_build_completion_forest_still_fails_closed_on_other_errors(
    monkeypatch,
) -> None:
    """Non-timeout errors from ``completion_domain`` must keep failing closed
    to an empty forest -- only the deadline case changes behavior."""
    from slm_training.dsl import grammar_capabilities

    def _raise_value_error(self, request):
        raise ValueError("unrelated grammar-capability failure")

    monkeypatch.setattr(
        grammar_capabilities.GrammarCapabilityAdapterV1,
        "completion_domain",
        _raise_value_error,
    )

    tok = DSLNativeTokenizer.build()
    prefix_ids = [tok.bos_id, *tok.encode(PREFIX, add_special=False)]

    forest = build_completion_forest(
        tok, prefix_ids, remaining_tokens=24, max_path_tokens=8, min_content=0
    )
    assert forest.coverage == "none"
    assert forest.paths == ()


def test_compiler_ltr_decode_one_checks_decode_deadline() -> None:
    """The compiler-tree per-position loop must raise ``TimeoutError`` once
    the cooperative decode wall has elapsed, matching the LTR-repair and
    MaskGIT loops (both call ``check_decode_deadline()`` at the loop head)."""
    from slm_training.dsl.schema import ExampleRecord
    from slm_training.data.contract import canonicalize_example_template_markers
    from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

    record = canonicalize_example_template_markers(
        ExampleRecord(
            id="deadline-compiler",
            prompt="card",
            openui='root = Card([title])\ntitle = TextContent(":hero.title")\n',
            placeholders=[":hero.title"],
            split="train",
            source="fixture",
        )
    )
    model = TwoTowerModel.from_records(
        [record],
        config=TwoTowerConfig(
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
        ),
        device="cpu",
    )
    model.eval()
    ctx, ctx_pad = model._encode_context(["card"])

    set_decode_deadline(0.01)  # arm a tiny budget, then let it elapse
    time.sleep(0.03)
    try:
        with pytest.raises(TimeoutError, match="decode deadline exceeded"):
            model._compiler_ltr_decode_one(
                ctx, ctx_pad, 8, mode="tree", slot_contract=None
            )
    finally:
        clear_decode_deadline()
