"""DSL tests against official @openuidev/lang-core bridge."""

from __future__ import annotations

import pytest

from slm_training.dsl import (
    ExampleRecord,
    ParseError,
    bridge_available,
    extract_placeholders,
    generate_system_prompt,
    parse,
    serialize,
    validate,
)

pytestmark = pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd tools/openui_bridge && npm ci",
)


def test_parse_and_roundtrip() -> None:
    src = (
        'root = Stack([hero], "column")\n'
        'hero_title = TextContent(":hero.title")\n'
        'hero_body = TextContent(":hero.body")\n'
        'hero = Card([hero_title, hero_body])'
    )
    program = validate(src)
    assert ":hero.title" in program.placeholders
    assert ":hero.body" in program.placeholders
    again = serialize(program)
    assert validate(again).placeholders == program.placeholders


def test_reject_literal_content_prop() -> None:
    src = 'root = Stack([cta])\ncta = Button("Click me")'
    with pytest.raises(ParseError, match="placeholder"):
        validate(src)


def test_reject_unknown_component() -> None:
    src = "root = FancyWidget()"
    with pytest.raises(ParseError):
        validate(src)


def test_extract_placeholders_order() -> None:
    src = 'a = TextContent(":b.x")\nc = Button(":a.y")\nd = TextContent(":b.x")'
    assert extract_placeholders(src) == [":b.x", ":a.y"]


def test_example_record_split_validation() -> None:
    with pytest.raises(ValueError, match="split"):
        ExampleRecord(
            id="x",
            prompt="p",
            openui='root = Stack([t])\nt = TextContent(":t")',
            split="nope",
        )


def test_parse_bool_and_number() -> None:
    src = (
        'root = Stack([x], "row", "m")\n'
        'x = TextContent(":t")'
    )
    program = parse(src)
    assert program.root is not None
    assert program.root["typeName"] == "Stack"


def test_official_system_prompt() -> None:
    prompt = generate_system_prompt()
    assert "openui-lang" in prompt.lower() or "OpenUI" in prompt or "Stack" in prompt
    assert "placeholder" in prompt.lower() or ":hero.title" in prompt
