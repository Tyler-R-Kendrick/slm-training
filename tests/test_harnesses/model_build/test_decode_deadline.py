"""Cooperative decode deadline hard wall."""

from __future__ import annotations

import time

import pytest

from slm_training.models.decode_stats import (
    check_decode_deadline,
    clear_decode_deadline,
    decode_deadline_remaining,
    set_decode_deadline,
)
from slm_training.dsl.grammar.fastpath.compiler_draft import build_completion_forest


def test_check_decode_deadline_raises_after_budget() -> None:
    set_decode_deadline(0.05)
    try:
        time.sleep(0.08)
        with pytest.raises(TimeoutError, match="decode deadline exceeded"):
            check_decode_deadline()
    finally:
        clear_decode_deadline()


def test_check_decode_deadline_noop_when_unarmed() -> None:
    clear_decode_deadline()
    check_decode_deadline()  # must not raise
    assert decode_deadline_remaining() is None


def test_set_decode_deadline_remaining_positive() -> None:
    set_decode_deadline(1.0)
    try:
        remaining = decode_deadline_remaining()
        assert remaining is not None
        assert 0.0 < remaining <= 1.0
        check_decode_deadline()  # not expired yet
    finally:
        clear_decode_deadline()


def test_clear_decode_deadline_disarms() -> None:
    set_decode_deadline(0.01)
    time.sleep(0.03)
    clear_decode_deadline()
    check_decode_deadline()  # cleared — no raise


def test_build_completion_forest_propagates_deadline_timeout(monkeypatch) -> None:
    """The compiler-tree completion-domain lookup must not swallow a
    cooperative decode-deadline TimeoutError as a fake grammar dead-end.

    Regression test for docs/design/decode-compiler-tree-deadline-swallow-finding.md:
    compiler_draft.py's ``build_completion_forest`` used to catch a bare
    ``except Exception`` around the completion-domain lookup, converting any
    TimeoutError (including one raised by ``check_decode_deadline``) into an
    empty ``CompletionForest`` — recorded downstream as a legitimate grammar
    dead-end instead of a timeout.
    """
    from slm_training.dsl import grammar_capabilities as gc

    def _raise_timeout(self, request):  # noqa: ANN001 - test stub signature
        raise TimeoutError("decode deadline exceeded")

    monkeypatch.setattr(
        gc.GrammarCapabilityAdapterV1, "completion_domain", _raise_timeout
    )

    with pytest.raises(TimeoutError, match="decode deadline exceeded"):
        build_completion_forest(
            tokenizer=None,
            prefix_ids=[1],
            state=None,
            slot_contract=None,
            max_path_tokens=8,
            min_content=0,
            remaining_tokens=5,
        )


def test_build_completion_forest_still_fails_closed_on_other_errors(
    monkeypatch,
) -> None:
    """Non-timeout errors keep failing closed to an empty completion forest
    (the deadline fix must not weaken the existing fail-closed contract)."""
    from slm_training.dsl import grammar_capabilities as gc

    def _raise_value_error(self, request):  # noqa: ANN001 - test stub signature
        raise ValueError("boom")

    monkeypatch.setattr(
        gc.GrammarCapabilityAdapterV1, "completion_domain", _raise_value_error
    )

    forest = build_completion_forest(
        tokenizer=None,
        prefix_ids=[1],
        state=None,
        slot_contract=None,
        max_path_tokens=8,
        min_content=0,
        remaining_tokens=5,
    )
    assert forest.paths == ()
    assert forest.coverage == "none"
