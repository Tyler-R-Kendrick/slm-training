"""ChoiceTokenizer (B1 / SLM-42): vocab closure, id round trips, model wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.dsl.lang_core import bridge_available
from slm_training.dsl.parser import validate
from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.models.choice_tokenizer import (
    CHOICE_TOKENIZER_KIND,
    DIR_PREFIX,
    LIT_PREFIX,
    LIT_STR,
    NAME_PREFIX,
    NAME_STR,
    TERNARY_OP,
    ChoiceDecodeState,
    ChoiceTokenizer,
    _component_contracts,
    is_choice_tokenizer,
)
from slm_training.models.dsl_tokenizer import SymbolTable

HERO = (
    'root = Stack([hero], "column")\n'
    'hero_title = TextContent(":hero.title")\n'
    'hero_body = TextContent(":hero.body")\n'
    "hero = Card([hero_title, hero_body])"
)

needs_bridge = pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)


@pytest.fixture(scope="module")
def tok() -> ChoiceTokenizer:
    return ChoiceTokenizer.build()


def test_vocab_is_deterministic_and_grammar_closed(tok: ChoiceTokenizer) -> None:
    again = ChoiceTokenizer.build()
    assert again.token_to_id == tok.token_to_id
    assert again.id_to_kind == tok.id_to_kind
    # Specials pinned at the front.
    assert tok.pad_id == 0 and tok.bos_id == 1 and tok.eos_id == 2
    assert tok.mask_id == 3 and tok.unk_id == 4
    assert tok.vocab_size < 1024


def test_save_load_roundtrip(tok: ChoiceTokenizer, tmp_path: Path) -> None:
    path = tmp_path / "choice.tokenizer.json"
    tok.save(path)
    assert f'"kind": "{CHOICE_TOKENIZER_KIND}"' in path.read_text(encoding="utf-8")
    loaded = ChoiceTokenizer.load(path)
    assert loaded.token_to_id == tok.token_to_id
    assert loaded.id_to_kind == tok.id_to_kind
    assert loaded.sym_slots == tok.sym_slots


def test_load_rejects_foreign_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "other.json"
    path.write_text('{"kind": "dsl_native", "token_to_id": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="choice_codec"):
        ChoiceTokenizer.load(path)


def test_kind_ids_partition(tok: ChoiceTokenizer) -> None:
    all_ids = set(range(tok.vocab_size))
    covered: set[int] = set()
    for kind in ("special", "struct", "component", "builtin", "lit", "byte", "sym", "bind", "state"):
        ids = tok.kind_ids(kind)
        assert not (ids & covered)
        covered |= ids
    assert covered == all_ids


@needs_bridge
def test_fixture_corpus_ids_are_decode_fixed_points(tok: ChoiceTokenizer) -> None:
    records: list[ExampleRecord] = []
    for path in (
        "src/slm_training/resources/train_seeds.jsonl",
        "src/slm_training/resources/test_seeds.jsonl",
    ):
        records.extend(load_jsonl(path))
    for record in records:
        table = SymbolTable.from_placeholders(
            list(record.placeholders or []), max_slots=tok.sym_slots
        )
        ids = tok.encode(record.openui, add_special=True, table=table)
        assert tok.unk_id not in ids, record.id
        decoded = tok.decode(ids, table=table)
        assert decoded, record.id
        assert validate(decoded).serialized == decoded, record.id
        fresh = SymbolTable.from_placeholders(
            list(record.placeholders or []), max_slots=tok.sym_slots
        )
        assert tok.encode(decoded, add_special=True, table=fresh) == ids, record.id


@needs_bridge
def test_free_literals_use_byte_channel_not_unk(tok: ChoiceTokenizer) -> None:
    source = 'root = Tabs([tab])\ntab = TabItem("one", ":tabs.one", [c])\nc = TextContent(":tabs.body")'
    table = SymbolTable.from_placeholders([":tabs.one", ":tabs.body"], max_slots=64)
    ids = tok.encode(source, add_special=False, table=table)
    assert tok.unk_id not in ids
    decoded = tok.decode(ids, table=table)
    assert '"one"' in decoded


@needs_bridge
def test_unknown_component_fails_closed_at_encode(tok: ChoiceTokenizer) -> None:
    from slm_training.dsl.lang_core import ParseError

    with pytest.raises(ParseError):
        tok.encode('root = FancyWidget(":x")')


def test_decode_fails_closed_on_mask_unk_and_garbage(tok: ChoiceTokenizer) -> None:
    assert tok.decode([tok.bos_id, tok.mask_id, tok.eos_id]) == ""
    assert tok.decode([tok.bos_id, tok.unk_id, tok.eos_id]) == ""
    assert tok.decode([tok.bos_id, tok.vocab_size + 5, tok.eos_id]) == ""
    assert tok.decode([]) == ""


def test_choice_state_reserves_a_complete_root_and_forces_singletons(
    tok: ChoiceTokenizer,
) -> None:
    state = ChoiceDecodeState(tok)
    assert tok.eos_id not in state.allowed_ids(3)

    component = tok.token_to_id["+CardHeader"]
    assert component in state.allowed_ids(3)
    assert state.advance_id(component)

    close = tok.token_to_id["-"]
    assert state.allowed_ids(2) == {close}
    assert state.advance_id(close)
    assert state.allowed_ids(1) == {tok.eos_id}


def test_choice_state_rejects_unavailable_slots_and_forward_refs(
    tok: ChoiceTokenizer,
) -> None:
    state = ChoiceDecodeState(tok, slot_count=1)
    assert tok.token_to_id["@0"] in state.allowed_ids(8)
    assert tok.token_to_id["@1"] not in state.allowed_ids(8)
    assert tok.token_to_id["&0"] not in state.allowed_ids(8)


def test_choice_state_counts_completed_variadic_items(tok: ChoiceTokenizer) -> None:
    state = ChoiceDecodeState(tok, slot_count=1)
    for token in ("+Card", "[", "+TextContent", "@0", "-"):
        assert state.advance_id(tok.token_to_id[token])

    assert state.frames[-1].kind == "variadic"
    assert state.frames[-1].item_count == 1


def test_choice_state_rejects_unbound_names_as_expressions(
    tok: ChoiceTokenizer,
) -> None:
    state = ChoiceDecodeState(tok)
    name_id = next(
        token_id
        for token_id, token in tok.id_to_token.items()
        if token.startswith(NAME_PREFIX)
    )
    assert not state.advance_id(name_id)
    assert name_id not in state.allowed_ids(8)

    object_state = ChoiceDecodeState(tok)
    assert object_state.advance_id(tok.token_to_id["{"])
    assert object_state.advance_id(name_id)


def test_choice_state_derives_placeholder_fields_from_dsl_policy(
    tok: ChoiceTokenizer,
) -> None:
    component, _ = next(
        (name, contract)
        for name, contract in _component_contracts().items()
        if contract[1] == 1
        and contract[0]
        and contract[0][0].get("x-openui-placeholder")
    )
    open_id = tok.token_to_id[f"+{component}"]
    slot_id = tok.token_to_id["@0"]
    literal_id = next(
        token_id
        for token_id, token in tok.id_to_token.items()
        if token.startswith(DIR_PREFIX)
    )

    without_contract = ChoiceDecodeState(tok)
    assert open_id not in without_contract.allowed_ids(8)

    with_contract = ChoiceDecodeState(tok, slot_count=1)
    assert with_contract.advance_id(open_id)
    assert slot_id in with_contract.allowed_ids(3)
    assert literal_id not in with_contract.allowed_ids(3)
    assert with_contract.advance_id(slot_id)
    assert with_contract.advance_id(tok.token_to_id["-"])
    assert with_contract.section_slot_ids == [frozenset({0})]
    assert with_contract.current_section_slot_ids == set()
    assert with_contract.clone().section_slot_ids == [frozenset({0})]
    assert with_contract.advance_id(tok.eos_id)


def test_choice_state_enforces_component_array_item_schema(
    tok: ChoiceTokenizer,
) -> None:
    state = ChoiceDecodeState(tok, slot_count=3)
    assert state.advance_id(tok.token_to_id["+Modal"])
    assert state.advance_id(tok.token_to_id["@0"])
    assert state.advance_id(tok.token_to_id[f"{LIT_PREFIX}true"])
    assert state.advance_id(tok.token_to_id["["])

    assert tok.token_to_id["@1"] not in state.allowed_ids(16)
    assert tok.token_to_id["+TextContent"] in state.allowed_ids(16)
    assert tok.token_to_id["+Button"] not in state.allowed_ids(16)
    assert state.advance_id(tok.token_to_id["+TextContent"])
    assert state.advance_id(tok.token_to_id["@1"])
    assert state.advance_id(tok.token_to_id["-"])
    assert state.advance_id(tok.token_to_id["+Buttons"])
    assert state.advance_id(tok.token_to_id["["])
    assert state.advance_id(tok.token_to_id["+Button"])
    assert state.advance_id(tok.token_to_id["@2"])
    assert state.advance_id(tok.token_to_id["-"])
    assert state.advance_id(tok.token_to_id["]"])
    assert state.advance_id(tok.token_to_id["-"])
    assert state.advance_id(tok.token_to_id["]"])
    assert state.advance_id(tok.token_to_id["-"])
    assert state.advance_id(tok.eos_id)


def test_choice_state_enforces_closed_typed_object_properties(
    tok: ChoiceTokenizer,
) -> None:
    state = ChoiceDecodeState(tok, slot_count=1)
    for token in ("+ImageGallery", "[", "{"):
        assert state.advance_id(tok.token_to_id[token])

    frame = state.frames[-1]
    assert frame.kind == "object"
    assert frame.property_names == ("src", "alt", "details")
    assert frame.required_properties == ("src",)

    close = tok.token_to_id["}"]
    src = tok.token_to_id[f"{NAME_PREFIX}src"]
    unknown = next(
        token_id
        for token_id, token in tok.id_to_token.items()
        if token.startswith(NAME_PREFIX)
        and token[len(NAME_PREFIX) :] not in frame.property_names
    )
    allowed = state.allowed_ids(8)
    assert src in allowed
    assert close not in allowed
    assert unknown not in allowed
    assert tok.token_to_id[NAME_STR] not in allowed

    assert state.advance_id(src)
    assert state.advance_id(tok.token_to_id["@0"])
    assert src not in state.allowed_ids(6)
    assert close in state.allowed_ids(6)
    for token in ("}", "]", "-", "<eos>"):
        assert state.advance_id(tok.token_to_id[token])


def test_choice_state_signature_tracks_variadic_item_count(
    tok: ChoiceTokenizer,
) -> None:
    empty = ChoiceDecodeState(tok, slot_count=1)
    for token in ("+Card", "["):
        assert empty.advance_id(tok.token_to_id[token])
    nonempty = empty.clone()
    for token in ("+TextContent", "@0", "-"):
        assert nonempty.advance_id(tok.token_to_id[token])

    assert empty.frames[-1].item_count == 0
    assert nonempty.frames[-1].item_count == 1
    assert empty.signature() != nonempty.signature()


def test_choice_state_caches_exact_legal_sets(tok: ChoiceTokenizer) -> None:
    tok.allowed_cache.clear()
    tok.allowed_cache_hits = 0
    tok.allowed_cache_misses = 0
    state = ChoiceDecodeState(tok)
    first = state.allowed_ids(8)
    second = state.allowed_ids(8)
    assert second == first
    assert tok.allowed_cache_misses == 1
    assert tok.allowed_cache_hits == 1


def test_direct_candidates_match_exhaustive_oracle_on_reachable_states(
    tok: ChoiceTokenizer,
) -> None:
    def advanced(
        token: str, parent: ChoiceDecodeState | None = None
    ) -> ChoiceDecodeState:
        state = (parent or ChoiceDecodeState(tok, slot_count=3)).clone()
        assert state.advance_id(tok.token_to_id[token])
        return state

    component = next(
        token for token in tok.token_to_id if token.startswith("+")
    )
    object_state = advanced("{")
    states = [
        ChoiceDecodeState(tok, slot_count=3),
        advanced("r="),
        advanced(component),
        advanced("["),
        object_state,
        advanced(NAME_STR, object_state),
        advanced(LIT_STR),
        advanced(TERNARY_OP),
    ]
    for state in states:
        accepted = set()
        for token_id in tok.id_to_token:
            probe = state.clone()
            if probe.advance_id(token_id):
                accepted.add(token_id)
        assert accepted <= state._candidate_ids()

    initial = states[0]
    assert initial._filter_allowed(
        initial._candidate_ids(), 8
    ) == initial.exhaustive_allowed_ids(8)


def test_direct_candidates_avoid_most_vocabulary_ids(tok: ChoiceTokenizer) -> None:
    tok.allowed_cache.clear()
    tok.candidates_considered = 0
    tok.vocab_candidates_avoided = 0
    initial = ChoiceDecodeState(tok, slot_count=2)
    initial.allowed_ids(8)
    assert tok.candidates_considered < tok.vocab_size
    assert tok.vocab_candidates_avoided > 0

    literal = initial.clone()
    assert literal.advance_id(tok.token_to_id[LIT_STR])
    literal_candidates = literal._candidate_ids()
    assert len(literal_candidates) < tok.vocab_size // 2


def test_choice_allowed_ids_with_evidence_matches_allowed_ids(
    tok: ChoiceTokenizer,
) -> None:
    # VSS0-02: the explain companion returns the identical allowed set plus
    # reason-coded evidence for every considered candidate, without disturbing
    # the default hot path.
    import json

    from slm_training.dsl.grammar.fastpath.compiler_draft import (
        ConstraintEvidence,
        ConstraintStage,
    )

    for slot_count, remaining in ((0, 3), (1, 8), (3, 8), (2, 4)):
        state = ChoiceDecodeState(tok, slot_count=slot_count)
        expected = state.allowed_ids(remaining)
        allowed, evidence = state.allowed_ids_with_evidence(remaining)
        assert allowed == expected
        assert evidence
        admitted = {e.candidate_id for e in evidence if e.admitted}
        assert admitted == set(allowed)
        assert all(
            e.reason_code and e.stage is ConstraintStage.GRAMMAR for e in evidence
        )
        excluded = {e.candidate_id for e in evidence if not e.admitted}
        assert excluded.isdisjoint(allowed)
        _, evidence_again = state.allowed_ids_with_evidence(remaining)
        assert evidence_again == evidence
        for record in evidence:
            assert (
                ConstraintEvidence.from_dict(json.loads(json.dumps(record.as_dict())))
                == record
            )


def test_expression_candidates_are_reused_and_request_scoped(
    tok: ChoiceTokenizer,
) -> None:
    tok.expression_candidate_cache.clear()
    first = tok.expression_candidates(slot_count=2, available_ref_count=1)
    assert tok.expression_candidates(slot_count=2, available_ref_count=1) is first
    assert tok.token_to_id["@1"] in first
    assert tok.token_to_id["@2"] not in first
    assert tok.token_to_id["&0"] in first
    assert tok.token_to_id["&1"] not in first


def test_minimum_completion_cache_collapses_equivalent_literal_states(
    tok: ChoiceTokenizer,
) -> None:
    literals = [
        token
        for token in tok.token_to_id
        if token.startswith(LIT_PREFIX + '"')
    ][:2]
    assert len(literals) == 2
    states = []
    for literal in literals:
        state = ChoiceDecodeState(tok)
        assert state.advance_id(tok.token_to_id[literal])
        states.append(state)
    assert states[0].signature() == states[1].signature()

    tok.completion_cache.clear()
    tok.completion_cache_hits = 0
    tok.completion_cache_misses = 0
    completion_length = states[0].minimal_completion_length()
    assert completion_length < 1025
    assert states[1].minimal_completion_length() == completion_length
    assert tok.completion_cache_misses == 1
    assert tok.completion_cache_hits == 1


@needs_bridge
def test_twotower_choice_wiring(tmp_path: Path) -> None:
    """from_records builds the choice tokenizer; train/save/load round trip."""
    import torch  # noqa: F401 - environment guard

    from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

    records = [
        ExampleRecord(
            id="t1",
            prompt="Hero card",
            openui=HERO,
            placeholders=[":hero.title", ":hero.body"],
        ),
        ExampleRecord(
            id="t2",
            prompt="CTA button",
            openui='root = Stack([cta], "column")\ncta = Button(":cta.label")',
            placeholders=[":cta.label"],
        ),
    ]
    cfg = TwoTowerConfig(
        output_tokenizer="choice",
        context_backend="scratch",
        grammar_constrained=True,
        d_model=64,
        n_heads=4,
        context_layers=1,
        denoiser_layers=2,
        max_prompt_len=64,
        max_target_len=64,
    )
    model = TwoTowerModel.from_records(records, config=cfg, device="cpu")
    assert is_choice_tokenizer(model.tokenizer)
    assert model.context_tokenizer is not model.tokenizer
    loss = model.training_loss(records)
    assert float(loss.detach()) >= 0.0
    # The deterministic choice state owns syntax even for an untrained model.
    text = model.generate("Hero card", gold=records[0])
    assert text
    assert validate(text).serialized == text

    ckpt = tmp_path / "choice.pt"
    model.save(ckpt)
    sidecar = ckpt.with_suffix(".tokenizer.json")
    assert sidecar.is_file()
    assert CHOICE_TOKENIZER_KIND in sidecar.read_text(encoding="utf-8")
    reloaded = TwoTowerModel.from_checkpoint(ckpt, device="cpu")
    assert is_choice_tokenizer(reloaded.tokenizer)
    assert reloaded.tokenizer.token_to_id == model.tokenizer.token_to_id
