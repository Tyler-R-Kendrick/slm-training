"""Unit tests for lexer-native DSL output tokenizer (V5 Stage A)."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.models.dsl_tokenizer import (
    DSLNativeTokenizer,
    SymbolTable,
    TokenKind,
    NL,
)
from slm_training.models.tokenizer import OpenUITokenizer, tokenize_text

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    'hero = Card([hero_title, hero_body])'
)

CTA = (
    'root = Stack([copy, cta], "column")\n'
    'copy = TextContent(":copy.line")\n'
    'cta = Button(":cta.label")'
)


@pytest.fixture
def tok() -> DSLNativeTokenizer:
    return DSLNativeTokenizer.build()


def test_vocab_is_fixed_and_typed(tok: DSLNativeTokenizer) -> None:
    assert tok.vocab_size < 400
    assert tok.kind_of(tok.token_to_id["Stack"]) == TokenKind.COMPONENT
    assert tok.kind_of(tok.token_to_id["="]) == TokenKind.STRUCT
    assert tok.kind_of(tok.sym_id(0)) == TokenKind.SYM
    assert tok.kind_of(tok.bind_id(0)) == TokenKind.BIND
    assert "NL" in tok.token_to_id
    # No whitespace token in the fixed vocab.
    assert " " not in tok.token_to_id


def test_round_trip_with_symbol_table(tok: DSLNativeTokenizer) -> None:
    table = SymbolTable.from_placeholders([":hero.title", ":hero.body"])
    ids = tok.encode(HERO, table=table, use_symbol_table=True)
    text = tok.decode(ids, table=table)
    # Binders are alpha-renamed; placeholders restored from table.
    assert '":hero.title"' in text
    assert '":hero.body"' in text
    assert "Stack" in text and "Card" in text and "TextContent" in text
    assert "column" in text
    # Re-encode of decoded text with same placeholder inventory is stable.
    ids2 = tok.encode(text, table=SymbolTable.from_placeholders(table.placeholders))
    text2 = tok.decode(ids2, table=SymbolTable.from_placeholders(table.placeholders))
    assert '":hero.title"' in text2


def test_round_trip_without_symbol_table_uses_literal_channel(
    tok: DSLNativeTokenizer,
) -> None:
    ids = tok.encode(HERO, use_symbol_table=False)
    text = tok.decode(ids)
    assert ":hero.title" in text
    assert "Stack" in text


def test_length_reduction_vs_compositional(tok: DSLNativeTokenizer) -> None:
    table = SymbolTable.from_placeholders([":hero.title", ":hero.body"])
    dsl_len = len(tok.encode(HERO, add_special=False, table=table))
    old = OpenUITokenizer.build([HERO])
    old_len = len(old.encode(HERO, add_special=False))
    assert dsl_len < old_len
    # Expected band from the design doc (~40–60 vs ~100+ for hero).
    assert dsl_len <= 60
    assert old_len >= 40


def test_placeholder_is_single_sym_token(tok: DSLNativeTokenizer) -> None:
    table = SymbolTable.from_placeholders([":hero.title"])
    ids = tok.encode(
        'x = TextContent(":hero.title")',
        add_special=False,
        table=table,
    )
    kinds = [tok.kind_of(i) for i in ids]
    assert TokenKind.SYM in kinds
    assert kinds.count(TokenKind.SYM) == 1
    # No compositional ':' '.' spell-out of the placeholder.
    tokens = [tok.id_to_token[i] for i in ids]
    assert ":" not in tokens
    assert tokens.count("<SYM_0>") == 1


def test_statement_spans(tok: DSLNativeTokenizer) -> None:
    table = SymbolTable.from_placeholders([":hero.title", ":hero.body"])
    ids = tok.encode(HERO, add_special=True, table=table)
    spans = tok.statement_spans(ids)
    assert len(spans) == 4
    # Remask expansion around a mid-statement token stays inside the span.
    mid = spans[1][0] + 1
    hit = tok.spanning_statement(ids, mid)
    assert hit == spans[1]


def test_save_load_roundtrip(tmp_path: Path, tok: DSLNativeTokenizer) -> None:
    path = tmp_path / "dsl.tokenizer.json"
    tok.save(path)
    loaded = DSLNativeTokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.kind_of(loaded.token_to_id["Card"]) == TokenKind.COMPONENT
    table = SymbolTable.from_placeholders([":cta.label"])
    a = tok.encode(CTA, table=table)
    b = loaded.encode(CTA, table=table)
    assert a == b


def test_terminal_kind_alignment(tok: DSLNativeTokenizer) -> None:
    """Every grammar terminal class has at least one matching id (Stage A gate)."""
    assert tok.kind_ids(TokenKind.COMPONENT)
    assert tok.kind_ids(TokenKind.BIND)
    assert tok.kind_ids(TokenKind.SYM)
    assert tok.kind_ids(TokenKind.STRUCT)
    assert tok.token_to_id["="] == tok.token_to_id.get("=")
    for punct in ("=", "(", ")", "[", "]", ","):
        assert punct in tok.token_to_id
    assert NL in tok.token_to_id


def test_fixture_seeds_round_trip(tok: DSLNativeTokenizer) -> None:
    path = Path("fixtures/train_seeds.jsonl")
    if not path.is_file():
        pytest.skip("fixtures missing")
    import json

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        src = row["openui"]
        placeholders = row.get("placeholders") or []
        table = SymbolTable.from_placeholders(placeholders)
        ids = tok.encode(src, table=table)
        text = tok.decode(ids, table=table)
        for ph in placeholders:
            assert f'"{ph}"' in text
        assert "Stack" in text or "Card" in text or "TextContent" in text or "Button" in text


def test_compositional_still_longer_on_placeholders() -> None:
    """Sanity: v2 compositional tokenization still spells placeholders."""
    tokens = tokenize_text('TextContent(":smoke.hero.title")')
    assert ":" in tokens and "smoke" in tokens
