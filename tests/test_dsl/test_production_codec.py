"""Production codec roundtrip and slot-pointer contract tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from slm_training.data.contract import (
    GenerationRequest,
    canonical_slot_contract,
    normalize_example_record,
)
from slm_training.data.leakage import normalize_openui_structure
from slm_training.data.splits import clustered_train_val_split
from slm_training.dsl.production_codec import (
    ACTION_STMT,
    CLOSE,
    EOL,
    MUTATION_STMT,
    OPEN_PREFIX,
    QUERY_STMT,
    REF_PREFIX,
    REL_REF_PREFIX,
    ROOT_STMT,
    SLOT_PREFIX,
    STATE_STMT,
    STMT,
    V05,
    ProductionCodec,
    build_vocab_from_corpus,
    decode_productions,
    encode_openui,
    from_choice_stream,
    from_relative_refs,
    roundtrip_openui,
    to_choice_stream,
    to_relative_refs,
)
from slm_training.dsl.lang_core import ParseError, bridge_available
from slm_training.dsl.lang_core import parse as lang_core_parse
from slm_training.dsl.schema import ExampleRecord, load_jsonl

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    'hero = Card([hero_title, hero_body])'
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


def test_production_codec_parallel_slot_pointers() -> None:
    codec = ProductionCodec.build([HERO])
    inventory = [":hero.title", ":hero.body"]
    prod, slot = codec.encode(HERO, inventory)
    assert prod[0] == codec.bos_id
    assert prod[-1] == codec.eos_id
    assert any(s > 0 for s in slot)
    decoded = codec.decode(prod, slot, inventory)
    assert normalize_openui_structure(decoded) == normalize_openui_structure(HERO)


def test_production_codec_stop_at_mask() -> None:
    codec = ProductionCodec.build([HERO])
    inventory = [":hero.title", ":hero.body"]
    prod, slot = codec.encode(HERO, inventory)
    masked = list(prod)
    masked[len(masked) // 2] = codec.mask_id
    partial = codec.decode(masked, slot, inventory, stop_at_mask=True)
    assert "<mask>" in partial


def test_production_codec_preserves_eos_on_truncate() -> None:
    codec = ProductionCodec.build([HERO])
    inventory = [":hero.title", ":hero.body"]
    prod, slot = codec.encode(HERO, inventory, max_len=8)
    assert len(prod) == 8
    assert prod[-1] == codec.eos_id
    assert len(slot) == len(prod)


def test_encode_uses_slot_indices_not_namespaces() -> None:
    program = encode_openui(HERO)
    assert program.slot_contract == (":hero.title", ":hero.body")
    joined = " ".join(program.tokens)
    assert ":hero.title" not in joined
    assert ":hero.body" not in joined
    assert f"{SLOT_PREFIX}0" in program.tokens
    assert f"{SLOT_PREFIX}1" in program.tokens


def test_roundtrip_structure_matches() -> None:
    program, decoded = roundtrip_openui(HERO)
    assert normalize_openui_structure(decoded) == normalize_openui_structure(HERO)


def test_decode_with_explicit_contract() -> None:
    program = encode_openui(HERO)
    alt_contract = (":smoke.title", ":smoke.body")
    decoded = decode_productions(program.tokens, alt_contract)
    assert ":smoke.title" in decoded
    assert ":smoke.body" in decoded
    assert ":hero.title" not in decoded


def test_production_vocab_is_grammar_closed() -> None:
    sources = [HERO, 'root = Stack([cta], "column")\ncta = Button(":cta")']
    vocab = build_vocab_from_corpus(sources)
    for src in sources:
        program = encode_openui(src)
        ids = vocab.encode(program.tokens, add_special=False)
        assert all(i < vocab.vocab_size for i in ids)
        roundtrip_tokens = vocab.decode_ids(ids, skip_special=True)
        decoded = decode_productions(roundtrip_tokens, program.slot_contract)
        assert normalize_openui_structure(decoded) == normalize_openui_structure(src)


def test_open_close_direction_tokens_present() -> None:
    program = encode_openui(HERO)
    assert any(t.startswith(OPEN_PREFIX) for t in program.tokens)
    assert CLOSE in program.tokens
    assert "^column" in program.tokens


def test_generation_request_from_record() -> None:
    record = ExampleRecord(
        id="t",
        prompt="hero",
        openui=HERO,
        placeholders=[":hero.title", ":hero.body"],
    )
    req = GenerationRequest.from_record(record)
    assert req.prompt == "hero"
    assert req.slot_contract == (":hero.title", ":hero.body")


def test_normalize_switchitem_and_slider_signatures() -> None:
    raw = (
        'root = Stack([notify, volume], "column")\n'
        'notify = SwitchItem(":held.settings.notify", ":held.settings.notify.desc", false)\n'
        'volume = Slider(":held.settings.volume", 0, 100, 40)'
    )
    record = ExampleRecord(
        id="held_out_settings_01",
        prompt="settings",
        openui=raw,
        placeholders=[
            ":held.settings.notify",
            ":held.settings.notify.desc",
            ":held.settings.volume",
        ],
        split="held_out",
    )
    normalized = normalize_example_record(record)
    assert 'SwitchItem(":held.settings.notify", ":held.settings.notify.desc", "notify")' in (
        normalized.openui
    )
    assert (
        'Slider("volume", "continuous", 0, 100, 1, [40], ":held.settings.volume")'
        in normalized.openui
    )


def test_fixture_settings_schema_consistency() -> None:
    test_line = Path("src/slm_training/resources/test_seeds.jsonl").read_text(encoding="utf-8").splitlines()[7]
    train_line = Path("src/slm_training/resources/train_seeds.jsonl").read_text(encoding="utf-8").splitlines()[15]
    test_rec = normalize_example_record(ExampleRecord.from_dict(json.loads(test_line)))
    train_rec = normalize_example_record(ExampleRecord.from_dict(json.loads(train_line)))
    assert 'SwitchItem(' in test_rec.openui and '"notify"' in test_rec.openui
    assert 'Slider("volume", "continuous"' in test_rec.openui
    assert 'Slider("volume", "continuous"' in train_rec.openui


def test_normalize_full_slider_signature_from_generated_schema() -> None:
    record = ExampleRecord(
        id="slider-drift",
        prompt="slider",
        openui='root = Slider("volume", "default", 0, 100, 1, 40, ":label")',
        placeholders=[":label"],
    )
    normalized = normalize_example_record(record)
    assert 'Slider("volume", "continuous", 0, 100, 1, [40], ":label")' in (
        normalized.openui
    )


def test_clustered_split_keeps_structures_disjoint() -> None:
    records = [
        ExampleRecord(id="a1", prompt="p", openui=HERO, split="train"),
        ExampleRecord(
            id="a2",
            prompt="p2",
            openui='root = Stack([hero], "column")\nhero = Card([t])\nt = TextContent(":x")',
            split="train",
        ),
        ExampleRecord(
            id="b1",
            prompt="p3",
            openui='root = Stack([cta], "row")\ncta = Button(":y")',
            split="train",
        ),
        ExampleRecord(
            id="b2",
            prompt="p4",
            openui='root = Stack([cta], "row")\ncta = Button(":z")',
            split="train",
        ),
    ]
    split = clustered_train_val_split(records, val_fraction=0.25, seed=7)
    train_fps = {normalize_openui_structure(r.openui) for r in split.train}
    val_fps = {normalize_openui_structure(r.openui) for r in split.val}
    assert train_fps.isdisjoint(val_fps)
    assert len(split.train) + len(split.val) == len(records)


def test_train_fixture_roundtrips() -> None:
    records = load_jsonl("src/slm_training/resources/train_seeds.jsonl")
    for record in records[:5]:
        program, decoded = roundtrip_openui(
            record.openui,
            slot_contract=canonical_slot_contract(
                record.openui,
                declared=record.placeholders,
            ),
        )
        assert program.slot_contract
        assert normalize_openui_structure(decoded) == normalize_openui_structure(
            record.openui
        )


def test_v05_production_sigils_and_roundtrip() -> None:
    program, decoded = roundtrip_openui(V05_PROGRAM)
    assert program.tokens[0] == V05
    assert ROOT_STMT in program.tokens
    assert STATE_STMT in program.tokens
    assert QUERY_STMT in program.tokens
    assert MUTATION_STMT in program.tokens
    assert ACTION_STMT in program.tokens
    parsed = __import__("slm_training.dsl", fromlist=["parse"]).parse(decoded)
    assert parsed.root is not None
    assert parsed.state_declarations == {"$s0": "all"}
    assert len(parsed.query_statements) == 1
    assert len(parsed.mutation_statements) == 1


def test_v05_codec_handles_arithmetic_without_runtime_statements() -> None:
    source = 'root = TextContent("" + (1 + 2 * 3))'
    program, decoded = roundtrip_openui(source)
    assert program.tokens[0] == V05
    assert "1 + 2 * 3" in decoded
    assert __import__("slm_training.dsl", fromlist=["parse"]).parse(decoded).root


# --- B2 (SLM-22): training targets are fixed points of the decode collapse ---

DEAD_BINDING = 'root = Card(":card.title")\norphan = Button(":cta.label")'

MODAL = (
    'root = Stack([dialog], "column")\n'
    'body = TextContent(":modal.body")\n'
    'dialog = Modal(":modal.title", true, [body])'
)

# Binder "hero" also appears inside placeholder string literals; statement
# order must come from the parsed AST, not from name matches in surface text.
NAME_IN_LITERAL = (
    'root = Stack([title, hero], "column")\n'
    'title = TextContent(":smoke.hero.kicker")\n'
    'head = CardHeader(":smoke.hero.title", ":smoke.hero.subtitle")\n'
    'hero = Card([head])'
)

V05_STATE_FIRST = (
    '$count = 0\n'
    'root = Stack([hello], "column")\n'
    'hello = TextContent(@Count($count) > 0 ? ":a" : ":b")'
)


def _fixture_records() -> list[ExampleRecord]:
    records: list[ExampleRecord] = []
    for path in (
        "src/slm_training/resources/train_seeds.jsonl",
        "src/slm_training/resources/test_seeds.jsonl",
    ):
        records.extend(load_jsonl(path))
    return records


def test_dead_binding_keeps_root_binding() -> None:
    _, decoded = roundtrip_openui(DEAD_BINDING)
    bindings = dict(line.split(" = ", 1) for line in decoded.splitlines())
    assert bindings["root"].startswith("Card(")


def test_positional_prop_order_survives_roundtrip() -> None:
    _, decoded = roundtrip_openui(MODAL)
    assert 'Modal(":modal.title", true, [v0])' in decoded


def test_forward_reference_to_root_fails_closed() -> None:
    source = 'root = Card(":card.title")\necho = Stack([root], "column")'
    with pytest.raises(ParseError, match="forward reference"):
        encode_openui(source)


def test_token_stream_is_fixed_point_of_decode() -> None:
    for source in (HERO, DEAD_BINDING, MODAL, NAME_IN_LITERAL, V05_STATE_FIRST):
        program = encode_openui(source)
        decoded = decode_productions(program.tokens, program.slot_contract)
        reencoded = encode_openui(decoded, slot_contract=program.slot_contract)
        assert reencoded.tokens == program.tokens, source


def test_fixture_corpus_token_streams_are_canonical_fixed_points() -> None:
    for record in _fixture_records():
        contract = canonical_slot_contract(
            record.openui, declared=record.placeholders
        )
        program = encode_openui(record.openui, slot_contract=contract)
        decoded = decode_productions(program.tokens, program.slot_contract)
        reencoded = encode_openui(decoded, slot_contract=program.slot_contract)
        assert reencoded.tokens == program.tokens, record.id


def _strip_statement_ids(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            key: _strip_statement_ids(value)
            for key, value in node.items()
            if key != "statementId"
        }
    if isinstance(node, list):
        return [_strip_statement_ids(item) for item in node]
    return node


@pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)
def test_fixture_corpus_roundtrip_is_langcore_canonical_equal() -> None:
    """encode→decode must preserve the official lang-core resolved structure.

    The official serializer keeps binder names, so equality is checked on the
    resolved root AST with statementId erased — the codec's deterministic
    alpha-renaming must never change what the official parser resolves.
    """
    for record in _fixture_records():
        contract = canonical_slot_contract(
            record.openui, declared=record.placeholders
        )
        program = encode_openui(record.openui, slot_contract=contract)
        decoded = decode_productions(program.tokens, program.slot_contract)
        expected = _strip_statement_ids(lang_core_parse(record.openui).root)
        actual = _strip_statement_ids(lang_core_parse(decoded).root)
        assert actual == expected, record.id


# --- C1: relative-index (De Bruijn) references -----------------------------


def test_relative_refs_emit_debruijn_deltas() -> None:
    absolute = encode_openui(HERO)
    relative = encode_openui(HERO, relative_refs=True)
    # Same length, refs converted from absolute (&i) to relative (~delta).
    assert len(relative.tokens) == len(absolute.tokens)
    assert any(t.startswith(REF_PREFIX) for t in absolute.tokens)
    assert not any(t.startswith(REF_PREFIX) for t in relative.tokens)
    rel_refs = [t for t in relative.tokens if t.startswith(REL_REF_PREFIX)]
    assert rel_refs
    # HERO's refs are all backward in canonical order → positive deltas.
    assert all(int(t[len(REL_REF_PREFIX):]) >= 1 for t in rel_refs)


def test_relative_refs_roundtrip_document() -> None:
    _, decoded = roundtrip_openui(HERO, relative_refs=True)
    assert normalize_openui_structure(decoded) == normalize_openui_structure(HERO)


def test_relative_refs_roundtrip_v05() -> None:
    program, decoded = roundtrip_openui(V05_PROGRAM, relative_refs=True)
    assert program.tokens[0] == V05
    assert any(t.startswith(REL_REF_PREFIX) for t in program.tokens)
    assert not any(t.startswith(REF_PREFIX) for t in program.tokens)
    assert normalize_openui_structure(decoded) == normalize_openui_structure(
        V05_PROGRAM
    )


def test_relative_and_absolute_decode_identically() -> None:
    absolute = encode_openui(HERO)
    relative = to_relative_refs(absolute.tokens)
    assert from_relative_refs(relative) == absolute.tokens
    assert decode_productions(relative, absolute.slot_contract) == decode_productions(
        absolute.tokens, absolute.slot_contract
    )


def test_relative_refs_are_translation_invariant() -> None:
    # Prepending an unrelated leaf renumbers absolute slots but not the local
    # def→use distance for refs that do not cross the inserted statement.
    base = encode_openui(HERO, relative_refs=True)
    shifted_src = (
        'root = Stack([extra, hero], "column")\n'
        'extra = TextContent(":extra")\n'
        'hero_title = TextContent(":hero.title")\n'
        'hero_body = TextContent(":hero.body")\n'
        'hero = Card([hero_title, hero_body])'
    )
    shifted = encode_openui(shifted_src, relative_refs=True)
    # The Card→children distances (hero_title, hero_body) survive the insertion.
    base_deltas = {t for t in base.tokens if t.startswith(REL_REF_PREFIX)}
    shifted_deltas = {t for t in shifted.tokens if t.startswith(REL_REF_PREFIX)}
    assert base_deltas & shifted_deltas


def test_production_codec_build_preserves_relative_refs() -> None:
    # build(relative_refs=True) must return an instance whose encode() actually
    # emits relative refs — not silently fall back to the absolute default.
    codec = ProductionCodec.build([HERO], relative_refs=True)
    assert codec.relative_refs is True
    prod, _ = codec.encode(HERO, [":hero.title", ":hero.body"])
    surface = {codec.id_to_production.get(pid, "") for pid in prod}
    assert any(tok.startswith(REL_REF_PREFIX) for tok in surface)
    assert not any(tok.startswith(REF_PREFIX) for tok in surface)


def test_relative_ref_illegal_delta_rejected() -> None:
    # A delta that resolves before the start of scope is not legal binding —
    # enforced here, not learned.
    # Stream: one statement (cur=0) whose ref points 5 statements back → idx -5.
    try:
        from_relative_refs(("=", f"{REL_REF_PREFIX}5", ";"))
    except ParseError:
        pass
    else:  # pragma: no cover - guard
        raise AssertionError("expected ParseError for out-of-scope relative ref")


# --- B1 (SLM-42): choice-sequence stream (semantic decisions only) ----------


def test_choice_stream_drops_document_statement_markers() -> None:
    program = encode_openui(HERO)
    choices = to_choice_stream(program.tokens)
    assert STMT in program.tokens
    assert STMT not in choices
    # Exactly one marker elided per statement; everything else survives.
    assert len(choices) == len(program.tokens) - 4
    assert from_choice_stream(choices) == program.tokens


def test_choice_stream_v05_drops_eol_and_inverts() -> None:
    program = encode_openui(V05_PROGRAM)
    choices = to_choice_stream(program.tokens)
    assert EOL in program.tokens
    assert EOL not in choices
    # Typed statement markers stay: statement kind is a semantic choice.
    assert ROOT_STMT in choices and STATE_STMT in choices
    assert from_choice_stream(choices) == program.tokens


def test_choice_stream_is_fixed_point_through_reencode() -> None:
    # The issue's verify clause: choices → serialize → parse → choices.
    for source in (HERO, MODAL, NAME_IN_LITERAL, V05_PROGRAM, V05_STATE_FIRST):
        program = encode_openui(source)
        choices = to_choice_stream(program.tokens)
        decoded = decode_productions(
            from_choice_stream(choices), program.slot_contract
        )
        reencoded = encode_openui(decoded, slot_contract=program.slot_contract)
        assert to_choice_stream(reencoded.tokens) == choices, source


def test_choice_stream_composes_with_relative_refs() -> None:
    program = encode_openui(HERO, relative_refs=True)
    choices = to_choice_stream(program.tokens)
    assert any(t.startswith(REL_REF_PREFIX) for t in choices)
    decoded = decode_productions(from_choice_stream(choices), program.slot_contract)
    assert normalize_openui_structure(decoded) == normalize_openui_structure(HERO)


def test_fixture_corpus_choice_streams_invert() -> None:
    for record in _fixture_records():
        contract = canonical_slot_contract(
            record.openui, declared=record.placeholders
        )
        program = encode_openui(record.openui, slot_contract=contract)
        choices = to_choice_stream(program.tokens)
        assert from_choice_stream(choices) == program.tokens, record.id


def test_production_codec_choice_stream_end_to_end() -> None:
    inventory = [":hero.title", ":hero.body"]
    codec = ProductionCodec.build([HERO], relative_refs=True, choice_stream=True)
    assert codec.choice_stream is True
    prod, slot = codec.encode(HERO, inventory)
    surface = [codec.id_to_production.get(pid, "") for pid in prod]
    assert STMT not in surface
    decoded = codec.decode(prod, slot, inventory)
    assert normalize_openui_structure(decoded) == normalize_openui_structure(HERO)


def test_choice_stream_unbalanced_frame_fails_closed() -> None:
    with pytest.raises(ParseError):
        from_choice_stream((f"{OPEN_PREFIX}Card", '#"x"'))
