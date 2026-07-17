"""Grammar backend tests — Lark OpenUI AST + alternate DSL."""

from __future__ import annotations

import pytest

from slm_training.dsl import ParseError, bridge_available
from slm_training.dsl.grammar.backends import available_backends, get_backend
from slm_training.dsl.grammar.backends.ast_utils import (
    ast_fingerprint,
    component_multiset,
)


OPENUI_SRC = (
    'root = Stack([hero], "column")\n'
    'hero = TextContent(":hero.title")\n'
)

V05_SRC = (
    'root = Stack([button, count])\n'
    '$filter = "all"\n'
    'items = Query("get_items", {filter: $filter}, {rows: []})\n'
    'save = Mutation("save_item", {filter: $filter})\n'
    'submit = Action([@Run(save), @Run(items), @Set($filter, "all")])\n'
    'button = Button(":actions.save", submit)\n'
    'count = TextContent("" + @Count(items.rows))'
)

TOY_SRC = (
    'root = row(title, action)\n'
    'title = text(":hero.title")\n'
    'action = button(":cta.label")\n'
)


def test_backends_registered() -> None:
    names = set(available_backends())
    assert "openui" in names
    assert "openui-lark" in names
    assert "openui-langcore" in names
    assert "toy-layout" in names


def test_openui_lark_parses_element_ast() -> None:
    backend = get_backend("openui-lark")
    assert backend.available()
    program = backend.parse(OPENUI_SRC)
    assert program.root is not None
    assert program.root["type"] == "element"
    assert program.root["typeName"] == "Stack"
    assert program.root["props"]["direction"] == "column"
    children = program.root["props"]["children"]
    assert len(children) == 1
    assert children[0]["typeName"] == "TextContent"
    assert children[0]["props"]["text"] == ":hero.title"
    assert ":hero.title" in program.placeholders
    assert component_multiset(program.root)["Stack"] == 1
    assert component_multiset(program.root)["TextContent"] == 1


def test_openui_lark_requires_newlines_between_statements() -> None:
    backend = get_backend("openui-lark")
    with pytest.raises(ParseError):
        backend.parse(
            'root = Stack([hero])hero = TextContent(":hero.title")'
        )


def test_openui_lark_v05_program_metadata_and_roundtrip() -> None:
    backend = get_backend("openui-lark")
    program = backend.validate(V05_SRC)
    assert program.root is not None
    assert program.meta["state_declarations"] == ["$filter"]
    assert program.meta["query_statements"] == ["items"]
    assert program.meta["mutation_statements"] == ["save"]
    assert backend.validate(backend.serialize(program)).root is not None


def test_openui_lark_stream_check() -> None:
    backend = get_backend("openui-lark")
    ok = backend.stream_check(OPENUI_SRC)
    assert ok.ok and ok.has_root and not ok.incomplete
    partial = backend.stream_check("root = Stack([")
    assert partial.incomplete or (partial.ok and not partial.complete_ok)
    assert partial.has_root
    bad = backend.stream_check('root = !!!\n')
    assert not bad.ok or bad.hard_error


def test_toy_layout_backend() -> None:
    backend = get_backend("toy-layout")
    program = backend.validate(TOY_SRC)
    assert program.root is not None
    assert program.root["typeName"] == "row"
    kids = program.root["props"]["children"]
    assert {c["typeName"] for c in kids} == {"text", "button"}
    assert ":hero.title" in program.placeholders
    assert ":cta.label" in program.placeholders


@pytest.mark.skipif(not bridge_available(), reason="OpenUI bridge missing")
def test_lark_ast_matches_langcore_fingerprint() -> None:
    lark = get_backend("openui-lark").parse(OPENUI_SRC)
    official = get_backend("openui-langcore").parse(OPENUI_SRC)
    assert lark.root is not None and official.root is not None
    assert component_multiset(lark.root) == component_multiset(official.root)
    assert ast_fingerprint(lark.root) == ast_fingerprint(official.root)
    # Prop names for Stack / TextContent should align with official ElementNode.
    assert lark.root["typeName"] == official.root["typeName"]
    assert lark.root["props"]["direction"] == official.root["props"]["direction"]
    assert (
        lark.root["props"]["children"][0]["props"]["text"]
        == official.root["props"]["children"][0]["props"]["text"]
    )


@pytest.mark.skipif(not bridge_available(), reason="OpenUI bridge missing")
def test_hybrid_prefers_langcore() -> None:
    hybrid = get_backend("openui")
    program = hybrid.parse(OPENUI_SRC)
    assert program.meta.get("backend") == "openui-langcore" or program.meta.get(
        "kind"
    ) in {
        "lang-core",
        None,
    }


def test_grammar_module_uses_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    from slm_training.models import grammar as grammar_mod

    grammar_mod.set_active_dsl("openui-lark")
    status = grammar_mod.stream_check(OPENUI_SRC)
    assert status.ok and status.has_root
    grammar_mod.set_active_dsl("toy-layout")
    status2 = grammar_mod.stream_check(TOY_SRC)
    assert status2.ok and status2.has_root
    grammar_mod.set_active_dsl("openui")
