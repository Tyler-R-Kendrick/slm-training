"""Torch-free tests for codebook-only abstract-trace decoding (AP-017 / SLM-304).

Fixture logits drive :func:`capture_abstract_span` through: an ordinary (sampled)
end before the hard cap, a forced end exactly at the hard cap, an empty span (end
selected immediately), determinism across repeated calls, and the phase hand-off
into an ``answer`` decode. :func:`validate_abstract_span` is exercised directly
against hand-built malformed/nested-delimiter sequences a live decode can never
produce (the mask forbids it), proving the replay-time contract independently of
generation.
"""

from __future__ import annotations

import pytest

from slm_training.dsl.abstract_plan import AbstractPlanV1
from slm_training.models.abstract_decode import (
    AbstractDecodeError,
    capture_abstract_span,
    capture_phased_generation,
    validate_abstract_span,
)

_PROMPT_TOKEN = 999
_VOCAB_SIZE = 16_450  # covers the abstract_plan namespace (0x4000..) with headroom


def _plan(*, slot_count: int, max_slot_count: int) -> AbstractPlanV1:
    return AbstractPlanV1(slot_count=slot_count, max_slot_count=max_slot_count)


def _prefer(vocab_size: int, token_id: int) -> list[float]:
    logits = [0.0] * vocab_size
    logits[token_id] = 5.0
    return logits


# --- capture_abstract_span: sampled end, forced end, empty span ------------


def test_sampled_end_before_cap_is_not_forced() -> None:
    plan = _plan(slot_count=3, max_slot_count=3)
    begin_id, end_id = plan.delimiter_token_ids
    s0 = plan.slot_token_ids[0]
    initial_prefix = (_PROMPT_TOKEN, begin_id)

    def forward(prefix: tuple[int, ...]) -> list[float]:
        emitted = len(prefix) - len(initial_prefix)
        assert emitted in (0, 1), f"unexpected forward call at emitted={emitted}"
        return _prefer(_VOCAB_SIZE, s0 if emitted == 0 else end_id)

    result = capture_abstract_span(
        forward_logits=forward, plan=plan, initial_prefix=initial_prefix
    )
    assert result.slot_token_ids == (s0,)
    assert result.forced_end is False
    assert result.capture.stop_reason == "eos"
    assert result.capture.generated_token_ids == (s0, end_id)
    assert result.token_count == 2
    assert len(result.log_probs) == 2
    assert all(lp <= 0.0 for lp in result.log_probs)


def test_forced_end_exactly_at_the_hard_cap() -> None:
    plan = _plan(slot_count=2, max_slot_count=2)
    begin_id, end_id = plan.delimiter_token_ids
    s0 = plan.slot_token_ids[0]
    initial_prefix = (_PROMPT_TOKEN, begin_id)

    def forward(prefix: tuple[int, ...]) -> list[float]:
        emitted = len(prefix) - len(initial_prefix)
        # The cap is reached after 2 codebook tokens; forward_logits must never be
        # called once the legal set has collapsed to {<endabstract>} alone.
        assert emitted < plan.max_slot_count, "forward called past the hard cap"
        return _prefer(_VOCAB_SIZE, s0)

    result = capture_abstract_span(
        forward_logits=forward, plan=plan, initial_prefix=initial_prefix
    )
    assert result.slot_token_ids == (s0, s0)
    assert result.forced_end is True
    assert result.capture.stop_reason == "eos"
    assert result.capture.generated_token_ids == (s0, s0, end_id)
    # No out-of-codebook token ever appears in the span.
    assert set(result.slot_token_ids) <= set(plan.slot_token_ids)
    # The forced final step has no alternative: log(1.0) == 0.0.
    assert result.log_probs[-1] == 0.0


def test_empty_span_when_end_is_selected_immediately() -> None:
    plan = _plan(slot_count=2, max_slot_count=5)
    begin_id, end_id = plan.delimiter_token_ids
    initial_prefix = (_PROMPT_TOKEN, begin_id)

    def forward(prefix: tuple[int, ...]) -> list[float]:
        assert prefix == initial_prefix
        return _prefer(_VOCAB_SIZE, end_id)

    result = capture_abstract_span(
        forward_logits=forward, plan=plan, initial_prefix=initial_prefix
    )
    assert result.slot_token_ids == ()
    assert result.forced_end is False
    assert result.token_count == 1
    assert result.capture.generated_token_ids == (end_id,)


def test_capture_is_deterministic_across_repeated_calls() -> None:
    plan = _plan(slot_count=3, max_slot_count=4)
    begin_id, end_id = plan.delimiter_token_ids
    s1 = plan.slot_token_ids[1]
    initial_prefix = (_PROMPT_TOKEN, begin_id)

    def forward(prefix: tuple[int, ...]) -> list[float]:
        emitted = len(prefix) - len(initial_prefix)
        return _prefer(_VOCAB_SIZE, s1 if emitted < 2 else end_id)

    first = capture_abstract_span(
        forward_logits=forward, plan=plan, initial_prefix=initial_prefix
    )
    second = capture_abstract_span(
        forward_logits=forward, plan=plan, initial_prefix=initial_prefix
    )
    assert first.slot_token_ids == second.slot_token_ids
    assert first.log_probs == second.log_probs
    assert first.forced_end == second.forced_end


def test_requires_initial_prefix_to_already_end_with_begin_token() -> None:
    plan = _plan(slot_count=2, max_slot_count=2)
    with pytest.raises(AbstractDecodeError, match="beginabstract"):
        capture_abstract_span(
            forward_logits=lambda prefix: _prefer(_VOCAB_SIZE, 0),
            plan=plan,
            initial_prefix=(_PROMPT_TOKEN,),
        )


# --- validate_abstract_span: malformed / nested delimiters -----------------


def test_validate_accepts_a_well_formed_span() -> None:
    plan = _plan(slot_count=3, max_slot_count=3)
    begin_id, end_id = plan.delimiter_token_ids
    s0, s1, _s2 = plan.slot_token_ids
    assert validate_abstract_span((begin_id, s0, s1, end_id), plan) == (s0, s1)


@pytest.mark.parametrize(
    ("build_span", "match"),
    [
        (lambda plan: (plan.delimiter_token_ids[0],), "at least"),
        (
            lambda plan: (plan.slot_token_ids[0], plan.delimiter_token_ids[1]),
            "beginabstract",
        ),
        (
            lambda plan: (plan.delimiter_token_ids[0], plan.slot_token_ids[0]),
            "endabstract",
        ),
        (
            lambda plan: (
                plan.delimiter_token_ids[0],
                plan.delimiter_token_ids[0],
                plan.delimiter_token_ids[1],
            ),
            "nested",
        ),
        (
            lambda plan: (
                plan.delimiter_token_ids[0],
                plan.delimiter_token_ids[1],
                plan.delimiter_token_ids[1],
            ),
            "nested",
        ),
        (
            lambda plan: (
                plan.delimiter_token_ids[0],
                999_999,
                plan.delimiter_token_ids[1],
            ),
            "outside the reserved abstract codebook",
        ),
    ],
)
def test_validate_rejects_malformed_or_nested_delimiters(build_span, match) -> None:
    plan = _plan(slot_count=2, max_slot_count=2)
    with pytest.raises(AbstractDecodeError, match=match):
        validate_abstract_span(build_span(plan), plan)


def test_validate_rejects_a_body_longer_than_the_hard_cap() -> None:
    plan = _plan(slot_count=2, max_slot_count=2)
    begin_id, end_id = plan.delimiter_token_ids
    s0, s1 = plan.slot_token_ids
    with pytest.raises(AbstractDecodeError, match="hard cap"):
        validate_abstract_span((begin_id, s0, s1, s0, end_id), plan)


# --- capture_phased_generation: prompt -> abstract -> answer hand-off ------


def test_phased_generation_hands_off_to_the_answer_phase() -> None:
    plan = _plan(slot_count=2, max_slot_count=2)
    begin_id, end_id = plan.delimiter_token_ids
    s0 = plan.slot_token_ids[0]
    prompt_ids = (_PROMPT_TOKEN,)
    answer_eos_id = 1  # outside the abstract_plan namespace

    def forward(prefix: tuple[int, ...]) -> list[float]:
        emitted = len(prefix) - (len(prompt_ids) + 1)
        assert emitted < plan.max_slot_count, "forward called past the abstract cap"
        return _prefer(_VOCAB_SIZE, s0)

    def answer_allowed_ids(prefix: tuple[int, ...]) -> tuple[int, ...]:
        expected = (*prompt_ids, begin_id, s0, s0, end_id)
        assert prefix == expected
        return (answer_eos_id,)

    result = capture_phased_generation(
        forward_logits=forward,
        plan=plan,
        prompt_ids=prompt_ids,
        answer_allowed_ids=answer_allowed_ids,
        answer_eos_id=answer_eos_id,
        answer_max_new_tokens=4,
        abstract_seed=1,
        answer_seed=2,
    )
    assert result.prompt_token_count == 1
    assert result.abstract.slot_token_ids == (s0, s0)
    assert result.abstract.forced_end is True
    assert result.answer.generated_token_ids == (answer_eos_id,)
    assert result.answer.stop_reason == "eos"
