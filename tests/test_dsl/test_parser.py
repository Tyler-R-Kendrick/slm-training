"""DSL parser and schema tests."""

from __future__ import annotations

import pytest

from slm_training.dsl import (
    ExampleRecord,
    ParseError,
    extract_placeholders,
    parse,
    serialize,
    validate,
)


def test_parse_and_roundtrip() -> None:
    src = (
        'root = Stack(direction="vertical", children=hero)\n'
        "hero = Card(title=:hero.title, body=:hero.body)"
    )
    program = validate(src)
    assert program.placeholders == [":hero.title", ":hero.body"]
    again = serialize(program)
    assert validate(again).placeholders == program.placeholders


def test_reject_literal_content_prop() -> None:
    src = 'root = Button(label="Click me")'
    with pytest.raises(ParseError, match="placeholder"):
        validate(src)


def test_reject_unknown_component() -> None:
    src = "root = FancyWidget()"
    with pytest.raises(ParseError):
        validate(src)


def test_extract_placeholders_order() -> None:
    src = "a = Text(text=:b.x)\nc = Button(label=:a.y)\nd = Text(text=:b.x)"
    assert extract_placeholders(src) == [":b.x", ":a.y"]


def test_example_record_split_validation() -> None:
    with pytest.raises(ValueError, match="split"):
        ExampleRecord(
            id="x",
            prompt="p",
            openui="root = Text(text=:t)",
            split="nope",
        )


def test_parse_bool_and_number() -> None:
    src = "root = Stack(direction=\"horizontal\", gap=12, children=x)\nx = Text(text=:t)"
    program = parse(src)
    assert len(program.statements) == 2
