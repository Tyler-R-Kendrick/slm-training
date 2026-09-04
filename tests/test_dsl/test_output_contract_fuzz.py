"""Fuzz and property tests for the output contracts.

The contracts are stated as universals — "no target may make the model predict
free-form text", "a placeholder never occupies a non-content property" — so
they are checked by generating input rather than by listing cases.

The property that matters is **fail-closed totality**: for *any* string a
checker either returns a violation list or raises one of a small set of typed
errors. What it may never do is raise something unexpected, or return "no
violations" for text it could not actually parse. Admission relies on exactly
this: `partition_certified_corpus` wraps the checkers in `except Exception` and
records a refusal, so a typed parse failure is a correct refusal — but an
unexpected error type escaping to the caller would take the whole build down
instead of rejecting one record.

Writing these found that the checkers raise `ParseError` on malformed input
rather than returning a list, which is the fail-closed behavior; the first
draft of this file asserted plain totality and was wrong.

Regression context: admission checked one contract while the trainer checked
four, and 29 records that no trainer could accept reached the certified train
bucket. The generators deliberately include the shape that caused it.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from slm_training.dsl.analysis.templatize import (
    assert_role_safe_output,
    role_contract_violations,
)
from slm_training.dsl.lang_core import ParseError
from slm_training.dsl.language_contract import (
    OutputContractError,
    assert_symbol_only_output,
    output_contract_violations,
)

pytestmark = [pytest.mark.property, pytest.mark.fuzz]

#: Errors a checker is allowed to raise. Anything else is a defect: admission
#: catches broadly, but a surprise type here means the checker crashed rather
#: than judged, and the record would be refused for the wrong reason.
ALLOWED_ERRORS = (ParseError, OutputContractError, ValueError)

# Keep every case far inside MAX_RUN_MINUTES; these are pure string checkers.
_SETTINGS = settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)

#: Fragments drawn from real refusals plus generic hostile input.
_FRAGMENTS = [
    "root",
    "=",
    "Stack",
    "(",
    ")",
    "[",
    "]",
    ",",
    '"',
    ":slot_0",
    ":slot_99",
    ":sym30",
    "RadialChart",
    "labels",
    "TextContent",
    "Button",
    "\n",
    "\t",
    " ",
    "\\",
    "\x00",
    "😀",
    "Semantic roles:",
    "free form text",
    "0",
    "-1",
]

_soup = st.lists(st.sampled_from(_FRAGMENTS), min_size=0, max_size=40).map("".join)
_text = st.one_of(
    _soup,
    st.text(max_size=200),
    st.text(alphabet=st.characters(blacklist_categories=("Cs",)), max_size=80),
)
_kinds = st.sampled_from([None, "document", "statement", "expression", "lexical"])
_role_kinds = st.sampled_from(["document", "statement", "lexical"])


def _judge(fn, source: str, **kwargs):
    """Return the violation list, or None when the checker refused outright."""
    try:
        result = fn(source, **kwargs)
    except ALLOWED_ERRORS:
        return None
    return list(result)


@given(source=_text, kind=_kinds)
@_SETTINGS
def test_output_contract_judges_or_refuses_but_never_crashes(
    source: str, kind: str | None
) -> None:
    """Any string is judged or refused with a typed error — never a surprise."""
    try:
        result = output_contract_violations(source, output_kind=kind)
    except ALLOWED_ERRORS:
        return
    assert isinstance(result, (list, tuple))


@given(source=_text, kind=_role_kinds)
@_SETTINGS
def test_role_contract_judges_or_refuses_but_never_crashes(
    source: str, kind: str
) -> None:
    try:
        result = role_contract_violations(source, output_kind=kind)
    except ALLOWED_ERRORS:
        return
    assert isinstance(result, (list, tuple))


@given(source=_text, kind=_kinds)
@_SETTINGS
def test_output_contract_is_deterministic(source: str, kind: str | None) -> None:
    """A rebuild must not re-admit what a previous build refused."""
    first = _judge(output_contract_violations, source, output_kind=kind)
    second = _judge(output_contract_violations, source, output_kind=kind)
    assert first == second


@given(source=_text, kind=_role_kinds)
@_SETTINGS
def test_role_contract_is_deterministic(source: str, kind: str) -> None:
    first = _judge(role_contract_violations, source, output_kind=kind)
    second = _judge(role_contract_violations, source, output_kind=kind)
    assert first == second


@given(source=_text, kind=_kinds)
@_SETTINGS
def test_assert_agrees_with_its_violation_list(source: str, kind: str | None) -> None:
    """The raising wrapper and the list form never disagree.

    Admission uses the assert; reporting uses the list. If they could diverge a
    record could be refused with an empty explanation, or admitted while the
    report showed a violation.
    """
    violations = _judge(output_contract_violations, source, output_kind=kind)
    if violations is None:
        # The list form refused outright; the assert must refuse too.
        with pytest.raises(ALLOWED_ERRORS):
            assert_symbol_only_output(source, output_kind=kind)
    elif violations:
        with pytest.raises(ALLOWED_ERRORS):
            assert_symbol_only_output(source, output_kind=kind)
    else:
        assert_symbol_only_output(source, output_kind=kind)


@given(source=_text, kind=_role_kinds)
@_SETTINGS
def test_role_assert_agrees_with_its_violation_list(source: str, kind: str) -> None:
    violations = _judge(role_contract_violations, source, output_kind=kind)
    if violations is None or violations:
        with pytest.raises(ALLOWED_ERRORS):
            assert_role_safe_output(source, output_kind=kind)
    else:
        assert_role_safe_output(source, output_kind=kind)


@given(
    prefix=st.sampled_from(["", "root = ", "root = Stack(["]),
    junk=st.text(max_size=60),
)
@_SETTINGS
def test_junk_is_never_silently_admitted_as_a_document(prefix: str, junk: str) -> None:
    """Fails closed: unparseable text is refused, never accepted clean.

    The checker may reject valid-looking text; what it may never do is return
    "no violations" for something it could not parse.
    """
    source = prefix + junk
    role = _judge(role_contract_violations, source, output_kind="document")
    symbol = _judge(output_contract_violations, source, output_kind="document")
    if role == [] and symbol == []:
        # Anything passing both clean must at least carry a root binding.
        assert source.strip().startswith("root"), source


def test_the_regression_shape_is_refused_under_the_document_kind() -> None:
    """A placeholder in a non-content property: the exact admission miss.

    29 records of this shape reached the certified train bucket because
    admission ran a narrower contract than the trainer.
    """
    source = 'root = Stack([c1])\nc1 = RadialChart([":slot_0"], [1])'
    assert role_contract_violations(source, output_kind="document")
    with pytest.raises(ValueError):
        assert_role_safe_output(source, output_kind="document")


def test_the_regression_shape_is_invisible_without_an_explicit_kind() -> None:
    """Why the first scan missed it, pinned so the trap cannot be reset.

    Certified records carry no `target_kind`; passing that straight through
    resolves a laxer contract than the model does. Admission must resolve the
    document kind explicitly. This documents the difference — it does not
    assert the lax answer is the right one.
    """
    source = 'root = Stack([c1])\nc1 = RadialChart([":slot_0"], [1])'
    lax = role_contract_violations(source, output_kind=None)
    strict = role_contract_violations(source, output_kind="document")
    assert not lax
    assert strict
