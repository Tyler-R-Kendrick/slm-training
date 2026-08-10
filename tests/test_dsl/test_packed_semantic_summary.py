"""Packed semantic summary (EXP-SR-7) unit checks."""

from __future__ import annotations

import pytest

from slm_training.dsl.grammar.fastpath.packed_semantic_summary import (
    assert_request_independent,
    build_packed_semantic_summary,
    load_checked_packed_semantic_summary,
    try_load_packed_semantic_summary,
)
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer


def test_build_and_load_checked_roundtrip() -> None:
    tok = DSLNativeTokenizer.build()
    summary = load_checked_packed_semantic_summary(tok)
    assert len(summary.kind_by_id) == tok.vocab_size
    assert tok.bos_id in summary.special_ids
    assert summary.has_semantic_effect(tok.bos_id) is False
    assert_request_independent(summary)


def test_try_load_fail_closed_on_divergence(monkeypatch: pytest.MonkeyPatch) -> None:
    from slm_training.dsl.grammar.fastpath import semantic_state as _ss

    tok = DSLNativeTokenizer.build()
    build_packed_semantic_summary(tok)

    def _broken(_tokenizer, _tid: int) -> str:
        return "broken_kind"

    monkeypatch.setattr(_ss, "_kind_of", _broken)
    loaded, reason = try_load_packed_semantic_summary(tok)
    assert loaded is None
    assert reason
