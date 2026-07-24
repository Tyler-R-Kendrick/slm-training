"""Regression coverage for the tokenizer/grammar CI certificate."""

from __future__ import annotations

import pytest

from scripts.verify_tokenizer_grammar_invariants import certify
from slm_training.dsl.openui_tokens import (
    TOKEN_ID_NAMESPACE_RANGES,
    logical_token_id,
)
from slm_training.models.choice_tokenizer import ChoiceDecodeState
from slm_training.models.tokenizer import SPECIAL, validate_token_layout


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


def test_certified_tokenizer_grammar_invariants() -> None:
    assert certify() == {"records": 19, "choice_records": 19}
