"""Grammar-generic AST scope extraction invariants."""

from __future__ import annotations

import pytest

from slm_training.data.scope_extract import (
    SCOPES,
    ScopeSlice,
    extract_scope_slices,
    typed_render,
    typed_terminal_value,
)
from slm_training.dsl.lang_core import ParseError

SOURCE = (
    'root = Stack([hero, cta], "column")\n'
    'hero = TextContent(":hero.title")\n'
    'cta = Button(":cta.label", true)\n'
    "count = 42"
)


def _slices(source: str = SOURCE, **kwargs) -> list[ScopeSlice]:
    return extract_scope_slices(source, **kwargs)


def test_every_span_is_exact() -> None:
    for item in _slices():
        assert SOURCE[item.span[0] : item.span[1]] == item.text


def test_all_four_scopes_present() -> None:
    scopes = {item.scope for item in _slices()}
    assert scopes == set(SCOPES)


def test_document_scope_is_whole_source() -> None:
    docs = [item for item in _slices() if item.scope == "document"]
    assert len(docs) == 1
    assert docs[0].text == SOURCE


def test_statement_scopes_carry_anchors() -> None:
    statements = [item for item in _slices() if item.scope == "statement"]
    assert [item.statement_anchor for item in statements] == [
        "root",
        "hero",
        "cta",
        "count",
    ]
    assert statements[3].text == "count = 42"


def test_lexical_scopes_type_literal_terminals() -> None:
    typed = {
        item.text: (item.category, item.typed_value)
        for item in _slices()
        if item.scope == "lexical" and item.typed
    }
    assert typed["true"] == ("BOOL", True)
    assert typed["42"] == ("NUMBER", 42)
    assert typed['"column"'] == ("STRING", "column")


def test_expression_scopes_nest_inside_statements() -> None:
    expressions = [item for item in _slices() if item.scope == "expression"]
    texts = {item.text for item in expressions}
    assert 'Stack([hero, cta], "column")' in texts
    assert "[hero, cta]" in texts
    # The statement itself is never an expression scope.
    assert "count = 42" not in texts


def test_scope_filter_is_honored() -> None:
    only_lexical = _slices(scopes=("lexical",))
    assert {item.scope for item in only_lexical} == {"lexical"}


def test_invalid_source_raises_parse_error() -> None:
    with pytest.raises(ParseError):
        extract_scope_slices("root = Stack([hero")


def test_generic_over_toy_grammar() -> None:
    toy = 'root = row(title)\ntitle = text(":hero.title")'
    slices = extract_scope_slices(toy, dsl="toy-layout")
    for item in slices:
        assert toy[item.span[0] : item.span[1]] == item.text
    assert {item.scope for item in slices} == set(SCOPES)


def test_typed_terminal_value_matches_transformer_semantics() -> None:
    assert typed_terminal_value("BOOL", "true") == (True, True)
    assert typed_terminal_value("BOOL", "false") == (True, False)
    assert typed_terminal_value("NUMBER", "42") == (True, 42)
    assert typed_terminal_value("NUMBER", "4.5") == (True, 4.5)
    assert typed_terminal_value("STRING", '"hi"') == (True, "hi")
    assert typed_terminal_value("NULL", "null") == (True, None)
    assert typed_terminal_value("NAME", "hero") == (False, None)


def test_typed_render_shapes() -> None:
    assert typed_render("BOOL", "true") == "Boolean(true)"
    assert typed_render("NUMBER", "42") == "Number(42)"
    assert typed_render("STRING", '"hi"') == 'String("hi")'
    assert typed_render("NULL", "null") == "Null(null)"
    assert typed_render("NAME", "hero") is None
