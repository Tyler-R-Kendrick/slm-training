"""SLM-303: decode-outcome taxonomy tests."""

from __future__ import annotations

import pytest

from slm_training.harnesses.model_build.decode_outcome import (
    DECODE_OUTCOMES,
    DecodeOutcomeRecordV1,
    classify_decode_outcome,
    fallback_counter_total,
    outcome_counts,
)


def test_taxonomy_has_six_outcomes() -> None:
    assert DECODE_OUTCOMES == (
        "model_valid",
        "model_invalid",
        "model_abstain",
        "runtime_timeout",
        "fallback_output",
        "harness_error",
    )


def test_precedence_harness_error_dominates() -> None:
    assert (
        classify_decode_outcome(
            parse_ok=True,
            fallback_counters=3,
            timed_out=True,
            abstained=True,
            harness_exception=True,
        )
        == "harness_error"
    )


def test_precedence_timeout_beats_fallback_and_parse() -> None:
    assert (
        classify_decode_outcome(
            parse_ok=True, fallback_counters=2, timed_out=True
        )
        == "runtime_timeout"
    )


def test_fallback_never_model_success() -> None:
    # Even a parseable output is a fallback artifact when a fallback fired.
    assert (
        classify_decode_outcome(parse_ok=True, fallback_counters=1)
        == "fallback_output"
    )
    assert (
        classify_decode_outcome(parse_ok=False, fallback_counters=1)
        == "fallback_output"
    )


def test_abstain_beats_parse_verdict() -> None:
    assert classify_decode_outcome(parse_ok=False, abstained=True) == "model_abstain"


def test_model_valid_and_invalid() -> None:
    assert classify_decode_outcome(parse_ok=True) == "model_valid"
    assert classify_decode_outcome(parse_ok=False) == "model_invalid"
    # parse not evaluated + produced output → model_valid with no parse claim
    assert classify_decode_outcome(parse_ok=None) == "model_valid"


def test_fallback_counter_total_mapping_and_object() -> None:
    assert fallback_counter_total(None) == 0
    assert fallback_counter_total({"compiler_fallbacks": 2, "seeded_fallbacks": 1}) == 3

    class _Stats:
        unconstrained_retries = 1
        template_fallback_count = 2

    assert fallback_counter_total(_Stats()) == 3


def test_outcome_counts_all_keys_present() -> None:
    counts = outcome_counts(["model_valid", "model_valid", "runtime_timeout"])
    assert counts["model_valid"] == 2
    assert counts["runtime_timeout"] == 1
    assert counts["harness_error"] == 0
    assert set(counts) == set(DECODE_OUTCOMES)
    with pytest.raises(ValueError):
        outcome_counts(["not_an_outcome"])


def test_record_roundtrip_and_guards() -> None:
    record = DecodeOutcomeRecordV1(
        request_id="r1",
        outcome="model_valid",
        stop_reason="completed",
        budget={"timeout_s": 10.0, "canvas_cap": 256},
        elapsed_ms=12.5,
    )
    clone = DecodeOutcomeRecordV1.from_dict(record.to_dict())
    assert clone == record
    with pytest.raises(ValueError):
        DecodeOutcomeRecordV1(request_id="r", outcome="bogus", stop_reason="x")
    with pytest.raises(ValueError):
        DecodeOutcomeRecordV1(
            request_id="r", outcome="fallback_output", stop_reason="x"
        )
    with pytest.raises(ValueError):
        DecodeOutcomeRecordV1(
            request_id="r",
            outcome="model_valid",
            stop_reason="x",
            fallback_used=True,
        )
