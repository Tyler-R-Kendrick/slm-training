"""B1 (SLM-42): choice-sequence codec — only semantic decisions remain."""

from __future__ import annotations

import pytest

from slm_training.dsl.lang_core import ParseError
from slm_training.dsl.production_codec import (
    CHOICE_STOP,
    choice_stats,
    choices_to_productions,
    decode_choices,
    encode_choices,
    encode_openui,
)

DOCUMENT = (
    'root = Stack([hero, cta], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    "hero = Card([hero_title, hero_body])\n"
    'cta = Button(":cta.label")'
)
SIMPLE = 'root = Stack([t])\nt = TextContent(":a.b")'


def test_choice_stream_contains_only_semantic_decisions() -> None:
    choices = encode_choices(DOCUMENT)
    assert choices.tokens
    # No statement markers, no distinct close delimiters.
    assert "=" not in choices.tokens
    assert "-" not in choices.tokens
    assert "]" not in choices.tokens
    # Every token is a decision: component open, slot filler, statement ref,
    # direction, literal, list-shape, or the generic stop.
    for token in choices.tokens:
        assert (
            token.startswith(("+", "@", "&", "~", "^", "#", "n:", "p:", "*", "$@"))
            or token in ("[", CHOICE_STOP)
        ), token


def test_choices_round_trip_through_the_deterministic_detokenizer() -> None:
    for source in (SIMPLE, DOCUMENT):
        choices = encode_choices(source)
        decoded = decode_choices(
            list(choices.tokens), slot_contract=choices.slot_contract
        )
        # Round trip: choices(decode(choices)) is the identity.
        again = encode_choices(decoded, slot_contract=choices.slot_contract)
        assert again.tokens == choices.tokens
        # And the detokenizer output matches the production codec's decode.
        production = encode_openui(source)
        assert (
            choices_to_productions(choices.tokens) == list(production.tokens)
        )


def test_detokenizer_reconstructs_all_surface_syntax() -> None:
    choices = encode_choices(DOCUMENT)
    decoded = decode_choices(
        list(choices.tokens), slot_contract=choices.slot_contract
    )
    # The reconstructed program is real OpenUI (validated downstream by the
    # official serializer in decode_productions) with all placeholders routed.
    assert "root = Stack(" in decoded
    for placeholder in (":hero.title", ":hero.body", ":cta.label"):
        assert placeholder in decoded


def test_choice_stream_shrinks_the_decision_surface() -> None:
    stats = choice_stats(DOCUMENT)
    # Fewer choice tokens than production tokens (statement markers dropped),
    # and far fewer decisions than surface atoms.
    assert stats["choice_tokens"] < stats["production_tokens"]
    assert stats["choice_per_surface_atom"] < 0.5
    assert stats["choice_per_production"] < 1.0


def test_illegal_choice_streams_fail_closed() -> None:
    with pytest.raises(ParseError, match="closes more scopes"):
        choices_to_productions([CHOICE_STOP])
    with pytest.raises(ParseError, match="leaves scopes open"):
        choices_to_productions(["+Stack", "["])
    with pytest.raises(ParseError):
        # A truncated stream must not silently decode.
        decode_choices(["+Stack", "["])
