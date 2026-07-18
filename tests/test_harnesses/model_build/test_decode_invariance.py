"""Paired decode-outcome attribution incl. the E288 decoder-defect regression."""

from __future__ import annotations

import pytest

from slm_training.harnesses.model_build.decode_invariance import (
    DecodeOutcome,
    classify_disagreement,
    is_invariant,
    pair_disagreement_summary,
)


def _o(eid: str, **kw: object) -> DecodeOutcome:
    base = dict(example_id=eid, parse_ok=True, meaningful=True, non_empty=True)
    base.update(kw)
    return DecodeOutcome(**base)  # type: ignore[arg-type]


def test_classify_each_class() -> None:
    assert classify_disagreement(_o("a"), _o("a")) == "agree"
    assert classify_disagreement(_o("a", non_empty=True), _o("a", non_empty=False)) == "empty_vs_populated"
    assert classify_disagreement(_o("a", meaningful=True), _o("a", meaningful=False)) == "semantic_binding"
    assert classify_disagreement(_o("a", parse_ok=True), _o("a", parse_ok=False)) == "syntax_placeholder"
    assert classify_disagreement(_o("a", timeout=False), _o("a", timeout=True)) == "timeout_fallback"
    assert (
        classify_disagreement(_o("a", choice_derivation="x"), _o("a", choice_derivation="y"))
        == "exact_choice_derivation"
    )
    assert (
        classify_disagreement(
            _o("a", canonical_output="C", raw_output="r1"),
            _o("a", canonical_output="C", raw_output="r2"),
        )
        == "surface_only"
    )


def test_classify_rejects_mismatched_examples() -> None:
    with pytest.raises(ValueError, match="different examples"):
        classify_disagreement(_o("a"), _o("b"))


def test_e288_decoder_defect_is_flagged_sensitive() -> None:
    """Byte-identical weights, parse 0 all-empty vs parse 1.0 populated.

    Regression for the E288 lesson: a decoder-path change (not weights) flipped
    the outcome, so the audit MUST classify this checkpoint decoder-sensitive.
    """
    declared = [
        DecodeOutcome(example_id=f"e{i}", parse_ok=False, meaningful=False, non_empty=False,
                      error_class="empty_root_stack")
        for i in range(19)
    ]
    exact = [
        DecodeOutcome(example_id=f"e{i}", parse_ok=True, meaningful=False, non_empty=True)
        for i in range(19)
    ]
    summary = pair_disagreement_summary("checkpoint_declared", "current_exact_or_compiler", declared, exact)
    assert summary["decoder_sensitive"] is True
    assert is_invariant(summary) is False
    assert summary["disagreement_counts"]["empty_vs_populated"] == 19
    assert summary["metric_deltas"]["parse_rate"] == 1.0


def test_identical_paths_are_invariant() -> None:
    outs = [_o(f"e{i}") for i in range(10)]
    summary = pair_disagreement_summary("a", "b", outs, outs)
    assert is_invariant(summary) is True
    assert summary["disagreement_counts"]["agree"] == 10
    assert summary["max_abs_metric_delta"] == 0.0


def test_surface_only_difference_stays_invariant() -> None:
    a = [_o("e0", canonical_output="C", raw_output="r1")]
    b = [_o("e0", canonical_output="C", raw_output="r2")]
    summary = pair_disagreement_summary("a", "b", a, b)
    assert summary["invariant"] is True
    assert summary["disagreement_counts"]["surface_only"] == 1


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="same length"):
        pair_disagreement_summary("a", "b", [_o("e0")], [])
