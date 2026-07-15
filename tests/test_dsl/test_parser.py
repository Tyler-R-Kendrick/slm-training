"""DSL tests against official @openuidev/lang-core bridge."""

from __future__ import annotations

import pytest

from slm_training.dsl import (
    ExampleRecord,
    ParseError,
    bridge_available,
    extract_placeholders,
    generate_system_prompt,
    library_schema,
    parse,
    serialize,
    stream_check,
    validate,
)
from slm_training.dsl.language_contract import contract_id

pytestmark = pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)

V05_PROGRAM = (
    'root = Stack([form, card])\n'
    '$filter = "all"\n'
    'items = Query("get_items", {filter: $filter}, {rows: []})\n'
    'save = Mutation("save_item", {filter: $filter})\n'
    'submit = Action([@Run(save), @Run(items), @Set($filter, "all")])\n'
    'button = Button(":actions.save", submit)\n'
    'buttons = Buttons([button])\n'
    'input = Input("filter", ":filter.placeholder", "text", null, $filter)\n'
    'field = FormControl(":filter.label", input)\n'
    'form = Form("filters", buttons, [field])\n'
    'count = TextContent("" + @Count(items.rows))\n'
    'card = Card([count])'
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


def test_v05_whole_program_roundtrip() -> None:
    program = validate(V05_PROGRAM)
    assert program.meta["contract_id"] == contract_id()
    assert program.state_declarations == {"$filter": "all"}
    assert [q["statementId"] for q in program.query_statements] == ["items"]
    assert [m["statementId"] for m in program.mutation_statements] == ["save"]

    serialized = serialize(program)
    assert 'items = Query("get_items"' in serialized
    assert 'save = Mutation("save_item"' in serialized
    assert "submit = Action([@Run(save), @Run(items)" in serialized
    again = validate(serialized)
    assert again.state_declarations == program.state_declarations
    assert len(again.query_statements) == 1
    assert len(again.mutation_statements) == 1


def test_v05_prompt_schema_and_stream_metadata() -> None:
    prompt = generate_system_prompt(
        tools=[
            {
                "name": "get_items",
                "description": "List items",
                "inputSchema": {"type": "object"},
                "outputSchema": {"type": "object"},
            }
        ],
        toolCalls=True,
        bindings=True,
    )
    assert "Query(" in prompt
    assert "Mutation(" in prompt
    assert "$variables" in prompt

    schema = library_schema()
    assert schema["x-openui-lang"]["version"] == "0.5"
    assert "query" in schema["x-openui-lang"]["features"]

    streamed = stream_check(V05_PROGRAM)
    assert streamed["ok"] is True
    assert streamed["state_declarations"] == {"$filter": "all"}
    assert len(streamed["query_statements"]) == 1
    assert len(streamed["mutation_statements"]) == 1

    partial = stream_check('root = Stack([button])\nbutton = Button(":save"')
    assert partial["incomplete"] is True
    assert partial["serialized"] is None
