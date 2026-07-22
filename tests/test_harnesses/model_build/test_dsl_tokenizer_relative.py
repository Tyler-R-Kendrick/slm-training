"""C1 scope-as-relative-index (De Bruijn) binder-channel tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.dsl import bridge_available
from slm_training.dsl.language_contract import output_contract_violations
from slm_training.models.dsl_tokenizer import (
    BIND_DEF,
    DSLNativeTokenizer,
    SymbolTable,
    TokenKind,
)

requires_bridge = pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)

# root FORWARD-references hero (defined last) — the reason the relative
# encoding uses signed statement deltas, not most-recent-binder indices.
HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    "hero = Card([hero_title, hero_body])"
)

CTA = (
    'root = Stack([copy, cta], "column")\n'
    'copy = TextContent(":copy.line")\n'
    'cta = Button(":cta.label")'
)


@pytest.fixture
def tok() -> DSLNativeTokenizer:
    return DSLNativeTokenizer.build(bind_encoding="relative")


def _bind_tokens(tok: DSLNativeTokenizer, ids: list[int]) -> list[str]:
    return [tok.id_to_token[i] for i in ids if tok.kind_of(i) == TokenKind.BIND]


def test_definitions_are_nameless_and_references_relative(
    tok: DSLNativeTokenizer,
) -> None:
    table = SymbolTable.from_placeholders([":hero.title", ":hero.body"])
    ids = tok.encode(HERO, table=table)
    binds = _bind_tokens(tok, ids)
    # 4 definitions + 3 references; the model never emits an identity slot.
    assert binds.count(BIND_DEF) == 4
    assert not any(b.startswith("<BIND_") for b in binds)
    # root (stmt 0) forward-references hero (stmt 3): delta +3; hero (stmt 3)
    # back-references hero_title (stmt 1) and hero_body (stmt 2): -2, -1.
    rels = [b for b in binds if b.startswith("<BINDREL_")]
    assert rels == ["<BINDREL_+3>", "<BINDREL_-2>", "<BINDREL_-1>"]


@requires_bridge
def test_relative_round_trip_is_canonically_equal() -> None:
    from slm_training.dsl.canonicalize import canonicalize

    tok = DSLNativeTokenizer.build(bind_encoding="relative")
    for src in (HERO, CTA):
        table = SymbolTable.from_placeholders(
            [":hero.title", ":hero.body", ":copy.line", ":cta.label"]
        )
        decoded = tok.decode(tok.encode(src, table=table), table=table)
        assert canonicalize(decoded) == canonicalize(src)


@requires_bridge
def test_relative_fixture_corpus_property() -> None:
    """Every committed seed program round-trips to its canonical class."""
    from slm_training.dsl.canonicalize import canonicalize

    path = Path("src/slm_training/resources/train_seeds.jsonl")
    if not path.is_file():
        pytest.skip("fixtures missing")
    tok = DSLNativeTokenizer.build(bind_encoding="relative")
    checked = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        src = row["openui"]
        if output_contract_violations(src):
            with pytest.raises(ValueError, match="free-form output string"):
                tok.encode(src)
            continue
        table = SymbolTable.from_placeholders(row.get("placeholders") or [])
        decoded = tok.decode(tok.encode(src, table=table), table=table)
        assert canonicalize(decoded) == canonicalize(src), row.get("id")
        checked += 1
    assert checked > 0


def test_relative_requires_root_defined_first(tok: DSLNativeTokenizer) -> None:
    with pytest.raises(ValueError, match="root binder"):
        tok.encode('a = Button(":x")\nroot = Stack([a])')


@requires_bridge
def test_out_of_scope_offset_is_rejected_by_verifier(
    tok: DSLNativeTokenizer,
) -> None:
    from slm_training.dsl.parser import stream_check

    # root = Stack([<BINDREL_-5>]) — offset resolves before statement 0.
    ids = [
        tok.bos_id,
        tok.token_to_id[BIND_DEF],
        tok.token_to_id["="],
        tok.token_to_id["Stack"],
        tok.token_to_id["("],
        tok.token_to_id["["],
        tok.token_to_id["<BINDREL_-5>"],
        tok.token_to_id["]"],
        tok.token_to_id[")"],
        tok.eos_id,
    ]
    decoded = tok.decode(ids)
    assert "oob5" in decoded
    # Never silently repaired: the decode-path verifier flags the dangling
    # reference (lang-core validate() would drop it, so we assert on the
    # stream check the eval gates actually use).
    status = stream_check(decoded)
    assert not status.ok
    assert "oob5" in tuple(status.unresolved or ())


def test_grammar_gate_accepts_relative_bind_channel(
    tok: DSLNativeTokenizer,
) -> None:
    from slm_training.dsl.grammar.fastpath.token_map import allowed_id_set

    allowed = allowed_id_set(tok, frozenset({"NAME"}))
    assert allowed is not None
    assert tok.bind_def_id in allowed
    assert tok.token_to_id["<BINDREL_+1>"] in allowed


def test_save_load_preserves_bind_encoding(
    tok: DSLNativeTokenizer, tmp_path: Path
) -> None:
    path = tmp_path / "tok.json"
    tok.save(path)
    loaded = DSLNativeTokenizer.load(path)
    assert loaded.bind_encoding == "relative"
    assert loaded.token_to_id[BIND_DEF] == tok.token_to_id[BIND_DEF]
    # Absolute (default) tokenizers stay absolute after a round-trip.
    plain = DSLNativeTokenizer.build()
    plain.save(tmp_path / "plain.json")
    assert DSLNativeTokenizer.load(tmp_path / "plain.json").bind_encoding == "absolute"
