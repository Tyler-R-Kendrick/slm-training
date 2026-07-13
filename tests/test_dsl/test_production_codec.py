"""Production codec roundtrip and slot-pointer contract tests."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.data.contract import (
    GenerationRequest,
    canonical_slot_contract,
    normalize_example_record,
)
from slm_training.data.leakage import normalize_openui_structure
from slm_training.data.splits import clustered_train_val_split
from slm_training.dsl.production_codec import (
    CLOSE,
    OPEN_PREFIX,
    SLOT_PREFIX,
    ProductionCodec,
    ProductionVocab,
    build_vocab_from_corpus,
    decode_productions,
    encode_openui,
    roundtrip_openui,
)
from slm_training.dsl.schema import ExampleRecord, load_jsonl

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    'hero = Card([hero_title, hero_body])'
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
        'Slider("volume", "default", 0, 100, 1, 40, ":held.settings.volume")'
        in normalized.openui
    )


def test_fixture_settings_schema_consistency() -> None:
    test_line = Path("fixtures/test_seeds.jsonl").read_text(encoding="utf-8").splitlines()[7]
    train_line = Path("fixtures/train_seeds.jsonl").read_text(encoding="utf-8").splitlines()[15]
    test_rec = normalize_example_record(ExampleRecord.from_dict(json.loads(test_line)))
    train_rec = normalize_example_record(ExampleRecord.from_dict(json.loads(train_line)))
    assert 'SwitchItem(' in test_rec.openui and '"notify"' in test_rec.openui
    assert 'Slider("volume", "default"' in test_rec.openui
    assert 'Slider("volume", "default"' in train_rec.openui


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
    records = load_jsonl("fixtures/train_seeds.jsonl")
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
