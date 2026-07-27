"""Regression coverage for the tokenizer/grammar CI certificate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verify_tokenizer_grammar_invariants import (
    _layout_fingerprint,
    certify,
)
from slm_training.dsl.openui_tokens import (
    TOKEN_ID_NAMESPACE_RANGES,
    logical_token_id,
)
from slm_training.models.choice_tokenizer import ChoiceDecodeState, ChoiceTokenizer
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer
from slm_training.models.tokenizer import SPECIAL, validate_token_layout

LAYOUT_REGISTRY = Path(__file__).resolve().parents[2] / (
    "src/slm_training/resources/tokenizer_layout_registry.json"
)


def test_token_layout_rejects_id_and_normalized_text_collisions() -> None:
    with pytest.raises(ValueError, match="unique and contiguous"):
        validate_token_layout({token: 0 for token in SPECIAL})
    with pytest.raises(ValueError, match="NFC normalization"):
        validate_token_layout(
            {
                **{token: index for index, token in enumerate(SPECIAL)},
                "é": 5,
                "e\u0301": 6,
            }
        )


def test_logical_token_namespaces_are_disjoint() -> None:
    ranges = list(TOKEN_ID_NAMESPACE_RANGES.values())
    assert all(left.stop <= right.start for left, right in zip(ranges, ranges[1:]))
    assert logical_token_id("control", 1) != logical_token_id("openui", 1)


def test_choice_schema_keeps_closed_enum_strings_legal() -> None:
    assert ChoiceDecodeState._schema_accepts({"type": "string"}, "string")
    assert not ChoiceDecodeState._schema_accepts(
        {"type": "string", "x-openui-placeholder": True}, "string"
    )


@pytest.mark.parametrize("codec", ["dsl_native", "choice_codec"])
def test_tokenizer_layout_matches_the_checked_in_registry(codec: str) -> None:
    """Pin the vocabulary without needing the Node bridge.

    ``verify_tokenizer_grammar_invariants`` certifies the same layout, but it
    requires ``src/apps/openui_bridge/node_modules`` and so only runs in the
    node-enabled CI job. The layout is a checkpoint-bound contract; it must be
    caught by plain ``pytest`` and ``.githooks/check-changed`` too.
    """
    tokenizer = (
        DSLNativeTokenizer.build() if codec == "dsl_native" else ChoiceTokenizer.build()
    )
    expected = json.loads(LAYOUT_REGISTRY.read_text(encoding="utf-8"))["codecs"][codec]
    assert tokenizer.version == expected["version"]
    assert tokenizer.vocab_size == expected["vocab_size"]
    assert _layout_fingerprint(tokenizer) == expected["layout_sha256"]


def test_default_vocabulary_does_not_depend_on_the_node_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: an unavailable bridge must not change ``vocab_size``.

    The Lark fallback backend reports a different structural-token set than
    lang-core, and routing the vocabulary through the live backend let
    ``vocab_size`` swing 569 <-> 605 with ``DSL_TOKENIZER_VERSION`` unchanged —
    silently incompatible checkpoints. See
    ``docs/design/agent-harness-parity-audit.md``.
    """
    from slm_training.dsl.grammar.backends import get_backend
    from slm_training.dsl.openui_tokens import STRUCTURAL_TOKENS as CANONICAL
    from slm_training.models import dsl_tokenizer

    assert dsl_tokenizer.STRUCTURAL_TOKENS == CANONICAL
    monkeypatch.delenv("SLM_GRAMMAR_DSL", raising=False)
    assert dsl_tokenizer._active_structural_tokens() == CANONICAL

    # The fallback backend genuinely disagrees; that is exactly why it must not
    # be consulted for the frozen default-DSL vocabulary.
    assert get_backend("openui-lark").structural_tokens() != CANONICAL


def test_lark_structural_tokens_exclude_grammar_rule_names() -> None:
    """Only quoted literals are surface tokens.

    Scanning the grammar file for capitalised words swept up rule and terminal
    names and even character-class fragments (``[A-Za-z0-9_]`` yielded a bare
    ``Za``), none of which can appear in a program.
    """
    from slm_training.dsl.grammar.backends import get_backend

    tokens = get_backend("openui-lark").structural_tokens()
    assert not tokens & {"AST", "BOOL", "COMMENT", "NAME", "NUMBER", "STRING", "Za"}
    assert {"Stack", "Card", "TextContent", "root"} <= tokens


def _bridge_available() -> bool:
    from slm_training.dsl.grammar.backends import get_backend

    return bool(get_backend("openui-langcore").available())


@pytest.mark.skipif(
    not _bridge_available(),
    reason="needs the OpenUI lang-core bridge: npm --prefix src/apps/openui_bridge ci",
)
def test_certified_tokenizer_grammar_invariants() -> None:
    """Full certificate; CI runs the script itself in a bridge-enabled job.

    Skipped rather than failed locally because the layout pins above already
    cover the checkpoint-bound part of this contract without Node.
    """
    assert certify() == {"records": 19, "choice_records": 19}
