"""Unit tests for lexer-native DSL output tokenizer (V5 Stage A)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from slm_training.data.contract import GenerationRequest, RuntimeSymbol
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

V05_PROGRAM = (
    'root = Stack([button, count])\n'
    '$filter = "all"\n'
    'items = Query("get_items", {filter: $filter}, {rows: []})\n'
    'save = Mutation("save_item", {filter: $filter})\n'
    'submit = Action([@Run(save), @Run(items), @Set($filter, "all")])\n'
    'button = Button(":actions.save", submit)\n'
    'count = TextContent("" + @Count(items.rows))'
)


@pytest.fixture
def tok() -> DSLNativeTokenizer:
    return DSLNativeTokenizer.build()


def test_vocab_is_fixed_and_typed(tok: DSLNativeTokenizer) -> None:
    # Fixed corpus-independent vocabulary incl. 64 reserved <MACRO_i> rows (C3).
    assert tok.vocab_size <= 480
    assert tok.kind_of(tok.token_to_id["Stack"]) == TokenKind.COMPONENT
    assert tok.kind_of(tok.token_to_id["="]) == TokenKind.STRUCT
    assert tok.kind_of(tok.sym_id(0)) == TokenKind.SYM
    assert tok.kind_of(tok.bind_id(0)) == TokenKind.BIND
    assert tok.kind_of(tok.state_id(0)) == TokenKind.STATE
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


def test_prefix_decode_preserves_terminal_newline(tok: DSLNativeTokenizer) -> None:
    ids = [
        *tok.encode('root = Button(":cta.label")', add_special=False),
        tok.token_to_id["NL"],
    ]
    assert not tok.decode(ids).endswith("\n")
    assert tok.decode(ids, preserve_trailing_newline=True).endswith("\n")


def test_runtime_symbol_contract_and_v2_table_migration(tok: DSLNativeTokenizer) -> None:
    request = GenerationRequest(
        prompt="Hero",
        slot_contract=(":hero.title",),
        runtime_symbols=(
            RuntimeSymbol(
                surface=":hero.title",
                role="external_entity",
                semantic_type="copy",
            ),
            RuntimeSymbol(surface="$filter", role="state"),
        ),
    )
    assert GenerationRequest.from_dict(request.to_dict()) == request
    table = SymbolTable.from_dict(
        {
            "placeholders": [":hero.title"],
            "binders": {"root": 0, "hero": 1},
            "states": {"$filter": 0},
        }
    )
    assert table.symbol_for_surface(":hero.title").role == "external_entity"
    assert table.to_dict()["version"] == 3
    assert table.active_token_ids(tok) >= {tok.sym_id(0), tok.bind_id(0), tok.state_id(0)}


def test_symbol_permutation_preserves_root_and_surfaces() -> None:
    table = SymbolTable.from_dict(
        {
            "placeholders": [":a", ":b", ":c"],
            "binders": {"root": 0, "left": 1, "right": 2},
            "states": {"$one": 0, "$two": 1},
        }
    )
    shuffled = table.permuted(7)
    assert shuffled.binders["root"] == 0
    assert set(shuffled.placeholders) == set(table.placeholders)
    assert set(shuffled.binders) == set(table.binders)
    assert set(shuffled.states) == set(table.states)


def test_request_features_change_only_active_rows(tok: DSLNativeTokenizer) -> None:
    from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

    model = TwoTowerModel(
        tokenizer=tok,
        config=TwoTowerConfig(
            d_model=16,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            max_prompt_len=16,
            max_target_len=16,
            runtime_symbol_features="role_gated",
        ),
    )
    table = SymbolTable.from_runtime_symbols(
        [
            RuntimeSymbol(surface=":hero.title", role="external_entity"),
            RuntimeSymbol(surface="local", role="alpha_binder"),
        ]
    )
    features = model._runtime_feature_tensor([table])
    assert features is not None
    assert torch.count_nonzero(features[0, tok.sym_id(0)]) > 0
    assert torch.count_nonzero(features[0, tok.bind_id(0)]) == 0
    assert torch.count_nonzero(features[0, tok.sym_id(1)]) == 0


def test_disabled_features_are_exact_and_semantic_mask_keeps_gold(
    tok: DSLNativeTokenizer,
) -> None:
    from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

    model = TwoTowerModel(
        tokenizer=tok,
        config=TwoTowerConfig(
            d_model=16,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            max_prompt_len=16,
            max_target_len=16,
            runtime_symbol_features="none",
            semantic_candidate_masks=True,
        ),
    )
    noisy = torch.tensor([[tok.bos_id, tok.mask_id]])
    context = torch.zeros((1, 2, 16))
    baseline = model.denoiser(noisy, context, tok.pad_id)
    model.denoiser.set_runtime_symbol_features(None)
    assert torch.equal(baseline, model.denoiser(noisy, context, tok.pad_id))

    table = SymbolTable.from_placeholders([":hero.title"])
    model._current_runtime_table = table
    masked = model._mask_inactive_dynamic_logits(torch.zeros((1, 1, tok.vocab_size)))
    assert torch.isfinite(masked[0, 0, tok.sym_id(0)])
    assert torch.isneginf(masked[0, 0, tok.sym_id(1)])
    assert torch.isfinite(masked[0, 0, tok.bind_id(1)])


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


def test_macro_induction_round_trip_and_persistence(tmp_path) -> None:
    """C3 (SLM-27): mined macros are deterministic, fixed-vocabulary only,
    shorten the corpus, and expand back to canonical-equal programs."""
    import json

    from slm_training.data.macro_induction import (
        MacroInductionConfig,
        induce_macros,
    )
    from slm_training.dsl.canonicalize import canonical_equal, canonicalize
    from slm_training.models.dsl_tokenizer import MACRO_EXPANDABLE_KINDS

    path = Path("src/slm_training/resources/train_seeds.jsonl")
    if not path.is_file():
        pytest.skip("fixtures missing")
    sources = [
        json.loads(line)["openui"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][:12]

    tok = DSLNativeTokenizer.build()
    first = induce_macros(sources, tok, MacroInductionConfig())
    second = induce_macros(sources, DSLNativeTokenizer.build(), MacroInductionConfig())
    assert first.expansions == second.expansions  # deterministic
    assert first.expansions, "expected macros on the fixture corpus"
    for expansion in first.expansions:
        assert len(expansion) >= 2
        for token in expansion:
            tid = tok.token_to_id[token]
            assert tok.id_to_kind[tid] in MACRO_EXPANDABLE_KINDS
    assert (
        first.stats["tokens_after_with_table"] < first.stats["tokens_before"]
    )

    tok.set_macro_expansions(first.expansions)
    plain = DSLNativeTokenizer.build()
    for source in sources:
        canon = canonicalize(source)
        table = SymbolTable()
        ids = tok.encode(canon, add_special=False, table=table)
        plain_ids = plain.encode(canon, add_special=False, table=SymbolTable())
        assert len(ids) <= len(plain_ids)
        assert canonical_equal(tok.decode(ids, table=table), canon)

    saved = tmp_path / "macro_tok.json"
    tok.save(saved)
    loaded = DSLNativeTokenizer.load(saved)
    assert loaded.macro_expansions == tok.macro_expansions
    table = SymbolTable()
    canon = canonicalize(sources[0])
    assert loaded.encode(canon, add_special=False, table=table) == tok.encode(
        canon, add_special=False, table=SymbolTable()
    )


def test_macro_expansions_fail_closed_on_dynamic_tokens(
    tok: DSLNativeTokenizer,
) -> None:
    with pytest.raises(ValueError, match="non-fixed kind"):
        tok.set_macro_expansions([("<SYM_0>", "=")])
    with pytest.raises(ValueError, match="too short"):
        tok.set_macro_expansions([("=",)])
    # An orphaned macro row renders as nothing rather than fake content.
    assert tok.decode([tok.macro_id(5)]) == ""


def test_surface_identifiers_round_trip_and_isolate_the_lever(
    tok: DSLNativeTokenizer,
) -> None:
    """C4 (SLM-28): symbol_anonymization=False keeps binder/state names as
    byte-channel surface text (exact round trip, no table needed at decode)
    while placeholders still ride <SYM_i> — the one-lever comparison arm."""
    for program in (HERO, CTA, V05_PROGRAM):
        table = SymbolTable()
        ids = tok.encode(program, table=table, symbol_anonymization=False)
        assert not any(tok.is_bind_id(i) for i in ids)
        assert not any(
            tok.kind_of(i) == TokenKind.STATE for i in ids
        )
        assert tok.decode(ids, table=table) == program
    # Placeholder channel is untouched by the flag.
    table = SymbolTable()
    ids = tok.encode(HERO, table=table, symbol_anonymization=False)
    assert any(tok.is_sym_id(i) for i in ids)
    # Nameless relative refs cannot carry surface names: fail closed.
    rel = DSLNativeTokenizer.build(bind_encoding="relative")
    with pytest.raises(ValueError, match="relative"):
        rel.encode(HERO, symbol_anonymization=False)


def test_fixture_seeds_round_trip(tok: DSLNativeTokenizer) -> None:
    path = Path("src/slm_training/resources/train_seeds.jsonl")
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


def test_canonicalize_is_idempotent_per_example(tok: DSLNativeTokenizer) -> None:
    """B2 (SLM-22): the decode collapse is a fixed point per fresh symbol table."""
    path = Path("src/slm_training/resources/train_seeds.jsonl")
    if not path.is_file():
        pytest.skip("fixtures missing")
    import json

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        src = json.loads(line)["openui"]
        first = tok.canonicalize(src, SymbolTable())
        second = tok.canonicalize(first, SymbolTable())
        assert second == first
        assert tok.encode(first, table=SymbolTable()) == tok.encode(
            second, table=SymbolTable()
        )


def test_compositional_still_longer_on_placeholders() -> None:
    """Sanity: v2 compositional tokenization still spells placeholders."""
    tokens = tokenize_text('TextContent(":smoke.hero.title")')
    assert ":" in tokens and "smoke" in tokens


def test_v05_typed_state_and_builtin_roundtrip(tok: DSLNativeTokenizer) -> None:
    table = SymbolTable.from_placeholders([":actions.save"])
    ids = tok.encode(V05_PROGRAM, table=table)
    assert any(tok.kind_of(i) == TokenKind.STATE for i in ids)
    assert any(tok.kind_of(i) == TokenKind.BUILTIN for i in ids)
    decoded = tok.decode(ids, table=table)
    assert '$s0 = "all"' in decoded
    assert "Query(" in decoded and "Mutation(" in decoded
    assert "Action([@Run(" in decoded and "@Set($s0" in decoded
    assert "{filter: $s0}" in decoded


def test_v05_tokenizer_ignores_line_comments(tok: DSLNativeTokenizer) -> None:
    source = 'root = Stack([]) // trailing comment\n# full-line comment\n'
    decoded = tok.decode(tok.encode(source))
    assert decoded == "root = Stack([])"


def test_v05_numbers_and_single_quotes_use_typed_literals(
    tok: DSLNativeTokenizer,
) -> None:
    source = "root = TextContent('hello')\nsmall = .5\nlarge = -1.25e+3"
    ids = tok.encode(source)
    assert ids.count(tok.token_to_id["LIT_NUM"]) == 2
    decoded = tok.decode(ids)
    assert 'TextContent("hello")' in decoded
    assert ".5" in decoded
    assert "-1.25e+3" in decoded


def test_v05_factorized_embeddings_have_distinct_new_kinds(
    tok: DSLNativeTokenizer,
) -> None:
    from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

    model = TwoTowerModel(
        tokenizer=tok,
        config=TwoTowerConfig(
            d_model=32,
            n_heads=4,
            context_layers=1,
            denoiser_layers=1,
            max_prompt_len=32,
            max_target_len=64,
            factorized_embeddings=True,
        ),
        device="cpu",
    )
    assert model.denoiser.kind is not None
    assert model.denoiser.kind.num_embeddings == 9
    lookup = model.denoiser.kind_lookup
    assert int(lookup[tok.state_id(0)]) == 8
    assert int(lookup[tok.token_to_id["@Run"]]) == 7
