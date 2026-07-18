"""ChoiceTokenizer (B1 / SLM-42): vocab closure, id round trips, model wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.dsl.lang_core import bridge_available
from slm_training.dsl.parser import validate
from slm_training.dsl.schema import ExampleRecord, load_jsonl
from slm_training.models.choice_tokenizer import (
    CHOICE_TOKENIZER_KIND,
    LIT_PREFIX,
    LIT_STR,
    NAME_STR,
    TERNARY_OP,
    ChoiceDecodeState,
    ChoiceTokenizer,
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


def test_choice_state_counts_bound_components(tok: ChoiceTokenizer) -> None:
    state = ChoiceDecodeState(tok, slot_count=1)
    for token in ("=", "+TextContent", '@0', "-"):
        assert state.advance_id(tok.token_to_id[token])
    assert state.bound_component_count == 1

    for token in ("r=", "+Stack", "[", "]", "-"):
        assert state.advance_id(tok.token_to_id[token])
    assert state.bound_component_count == 1
    assert state.valid_root_seen


def test_choice_state_does_not_count_bound_stack_as_content(
    tok: ChoiceTokenizer,
) -> None:
    state = ChoiceDecodeState(tok)
    for token in ("=", "+Stack", "[", "]", "-"):
        assert state.advance_id(tok.token_to_id[token])
    assert state.bound_component_count == 0


def test_choice_state_identifies_slot_content_components(
    tok: ChoiceTokenizer,
) -> None:
    state = ChoiceDecodeState(tok)
    assert state.is_slot_content_component_id(tok.token_to_id["+TextContent"])
    assert tok.required_slot_count(tok.token_to_id["+TextContent"]) == 1
    assert tok.slot_content_count(tok.token_to_id["+TextContent"]) == 1
    assert state.is_slot_content_component_type("element:TextContent")
    assert not state.is_slot_content_component_id(tok.token_to_id["+Stack"])
    assert not state.is_slot_content_component_id(tok.token_to_id["+Separator"])
    assert tok.required_slot_count(tok.token_to_id["+CardHeader"]) == 0
    assert tok.slot_content_count(tok.token_to_id["+CardHeader"]) == 2
    assert tok.slot_content_count(tok.token_to_id["+Callout"]) == 2
    assert tok.slot_content_count(tok.token_to_id["+Input"]) == 2


def test_choice_state_initial_bind_path_can_satisfy_content_floor(
    tok: ChoiceTokenizer,
) -> None:
    state = ChoiceDecodeState(tok, slot_count=1)
    assert tok.token_to_id["="] in state.allowed_ids(32)
    assert state.mode is None

    assert state.advance_id(tok.token_to_id["="])
    assert state.mode == "v05"


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
        component_plan_loss_weight=1.0,
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
    model.config.decode_min_content = -1
    model.config.component_plan_decode_weight = 2.0
    state = ChoiceDecodeState(model.tokenizer, slot_count=2)
    initial = model._choice_min_content_legal_ids(
        state,
        state.allowed_ids(63),
        [":hero.title", ":hero.body"],
    )
    assert model.tokenizer.token_to_id["="] not in initial
    assert model.tokenizer.token_to_id["+TextContent"] in initial
    assert all(state.is_slot_content_component_id(token_id) for token_id in initial)
    state = ChoiceDecodeState(model.tokenizer, slot_count=2)
    for token in ("=", "+TextContent"):
        assert state.advance_id(model.tokenizer.token_to_id[token])
    slot = model._choice_min_content_legal_ids(
        state,
        state.allowed_ids(61),
        [":hero.title", ":hero.body"],
    )
    assert slot == {model.tokenizer.token_to_id["@0"]}
    assert state.advance_id(model.tokenizer.token_to_id["@0"])
    bound_close = model._choice_min_content_legal_ids(
        state,
        state.allowed_ids(60),
        [":hero.title", ":hero.body"],
    )
    assert bound_close == {model.tokenizer.token_to_id["-"]}
    assert state.advance_id(model.tokenizer.token_to_id["-"])
    root_marker = model._choice_min_content_legal_ids(
        state,
        state.allowed_ids(59),
        [":hero.title", ":hero.body"],
    )
    assert root_marker == {model.tokenizer.token_to_id["="]}
    for token in ("=", "+TextContent"):
        assert state.advance_id(model.tokenizer.token_to_id[token])
    second_bound_slot = model._choice_min_content_legal_ids(
        state,
        state.allowed_ids(57),
        [":hero.title", ":hero.body"],
        [model.tokenizer.token_to_id["@0"]],
    )
    assert second_bound_slot == {model.tokenizer.token_to_id["@1"]}
    for token in ("@1", "-"):
        assert state.advance_id(model.tokenizer.token_to_id[token])
    root_marker = model._choice_min_content_legal_ids(
        state,
        state.allowed_ids(55),
        [":hero.title", ":hero.body"],
    )
    assert root_marker == {model.tokenizer.token_to_id["r="]}
    assert state.advance_id(model.tokenizer.token_to_id["r="])
    root = model._choice_min_content_legal_ids(
        state,
        state.allowed_ids(54),
        [":hero.title", ":hero.body"],
    )
    assert root == {model.tokenizer.token_to_id["+Stack"]}
    assert state.advance_id(model.tokenizer.token_to_id["+Stack"])
    stack_arg = model._choice_min_content_legal_ids(
        state,
        state.allowed_ids(53),
        [":hero.title", ":hero.body"],
    )
    assert stack_arg == {model.tokenizer.token_to_id["["]}
    assert state.advance_id(model.tokenizer.token_to_id["["])
    child = model._choice_min_content_legal_ids(
        state,
        state.allowed_ids(52),
        [":hero.title", ":hero.body"],
    )
    assert child == {model.tokenizer.token_to_id["&0"]}
    assert state.advance_id(model.tokenizer.token_to_id["&0"])
    list_close = model._choice_min_content_legal_ids(
        state,
        state.allowed_ids(51),
        [":hero.title", ":hero.body"],
    )
    assert list_close == {model.tokenizer.token_to_id["&1"]}
    assert state.advance_id(model.tokenizer.token_to_id["&1"])
    list_close = model._choice_min_content_legal_ids(
        state,
        state.allowed_ids(50),
        [":hero.title", ":hero.body"],
    )
    assert list_close == {model.tokenizer.token_to_id["]"]}
    assert state.advance_id(model.tokenizer.token_to_id["]"])
    component_close = model._choice_min_content_legal_ids(
        state,
        state.allowed_ids(49),
        [":hero.title", ":hero.body"],
    )
    assert component_close == {model.tokenizer.token_to_id["-"]}

    structural = ChoiceDecodeState(model.tokenizer, slot_count=2)
    for token in ("+TextContent", "@0", "-", "+TextContent", "@1", "-"):
        legal = model._choice_min_content_legal_ids(
            structural,
            structural.allowed_ids(64),
            [":hero.title", ":hero.body"],
            [model.tokenizer.token_to_id["@0"]]
            if token == "@1"
            else None,
        )
        assert model.tokenizer.token_to_id[token] in legal, token
        assert structural.advance_id(model.tokenizer.token_to_id[token]), token
    after_floor = model._choice_min_content_legal_ids(
        structural,
        structural.allowed_ids(58),
        [":hero.title", ":hero.body"],
        [
            model.tokenizer.token_to_id["@0"],
            model.tokenizer.token_to_id["@1"],
        ],
    )
    assert model.tokenizer.eos_id not in after_floor
    assert model.tokenizer.token_to_id["+Card"] in after_floor
    exhausted = model._choice_min_content_legal_ids(
        structural,
        structural.allowed_ids(58),
        [":hero.title", ":hero.body"],
        [
            model.tokenizer.token_to_id["@0"],
            model.tokenizer.token_to_id["@1"],
        ],
    )
    assert model.tokenizer.token_to_id["+Button"] not in exhausted
    assert model.tokenizer.token_to_id["+Card"] in exhausted
    with torch.no_grad():
        assert model.component_plan_head is not None
        model.component_plan_head.weight.zero_()
        model.component_plan_head.bias.fill_(-10.0)
        card_id = model.tokenizer.token_to_id["+Card"]
        model.component_plan_head.bias[model.tokenizer.vocab_size + card_id] = 1.25
    ctx = torch.zeros(1, 1, model.config.d_model)
    ctx_pad = torch.zeros(1, 1, dtype=torch.bool)
    planned = model._choice_structural_plan_legal_ids(
        structural, after_floor, ctx, ctx_pad
    )
    assert planned == {card_id}
    for token in ("+Card", "["):
        assert structural.advance_id(model.tokenizer.token_to_id[token]), token
    card_children = model._choice_min_content_legal_ids(
        structural,
        structural.allowed_ids(56),
        [":hero.title", ":hero.body"],
    )
    assert card_children == {
        model.tokenizer.token_to_id["&0"],
        model.tokenizer.token_to_id["&1"],
    }
    for token in ("&0", "]", "-"):
        assert structural.advance_id(model.tokenizer.token_to_id[token]), token
    root = model._choice_structural_plan_legal_ids(
        structural,
        model._choice_min_content_legal_ids(
            structural,
            structural.allowed_ids(52),
            [":hero.title", ":hero.body"],
            [
                model.tokenizer.token_to_id["@0"],
                model.tokenizer.token_to_id["@1"],
            ],
        ),
        ctx,
        ctx_pad,
    )
    assert root == {model.tokenizer.token_to_id["+Stack"]}
    for token in ("+Stack", "["):
        assert structural.advance_id(model.tokenizer.token_to_id[token]), token
    root_children = model._choice_min_content_legal_ids(
        structural,
        structural.allowed_ids(49),
        [":hero.title", ":hero.body"],
        [
            model.tokenizer.token_to_id["&0"],
            model.tokenizer.token_to_id["&0"],
        ],
    )
    assert model.tokenizer.token_to_id["]"] not in root_children
    assert model.tokenizer.token_to_id["&1"] in root_children
    assert model.tokenizer.token_to_id["&2"] in root_children
    assert structural.advance_id(model.tokenizer.token_to_id["&1"])
    root_last_child = model._choice_min_content_legal_ids(
        structural,
        structural.allowed_ids(48),
        [":hero.title", ":hero.body"],
        [
            model.tokenizer.token_to_id["&0"],
            model.tokenizer.token_to_id["&1"],
        ],
    )
    assert root_last_child == {model.tokenizer.token_to_id["&2"]}
    for token in ("&2", "]", "-"):
        assert structural.advance_id(model.tokenizer.token_to_id[token]), token
    complete = model._choice_min_content_legal_ids(
        structural,
        structural.allowed_ids(47),
        [":hero.title", ":hero.body"],
    )
    assert complete == {model.tokenizer.eos_id}

    multi = ChoiceDecodeState(model.tokenizer, slot_count=3)
    prefix = [model.tokenizer.bos_id]
    for token in ("=", "+AccordionItem"):
        token_id = model.tokenizer.token_to_id[token]
        assert multi.advance_id(token_id)
        prefix.append(token_id)
    first_slot = model._choice_min_content_legal_ids(
        multi,
        multi.allowed_ids(52),
        [":faq.title", ":faq.body", ":faq.extra"],
        prefix,
    )
    assert first_slot == {model.tokenizer.token_to_id["@0"]}
    assert multi.advance_id(model.tokenizer.token_to_id["@0"])
    prefix.append(model.tokenizer.token_to_id["@0"])
    second_slot = model._choice_min_content_legal_ids(
        multi,
        multi.allowed_ids(51),
        [":faq.title", ":faq.body", ":faq.extra"],
        prefix,
    )
    assert second_slot == {model.tokenizer.token_to_id["@1"]}

    ckpt = tmp_path / "choice.pt"
    model.save(ckpt)
    sidecar = ckpt.with_suffix(".tokenizer.json")
    assert sidecar.is_file()
    assert CHOICE_TOKENIZER_KIND in sidecar.read_text(encoding="utf-8")
    reloaded = TwoTowerModel.from_checkpoint(ckpt, device="cpu")
    assert is_choice_tokenizer(reloaded.tokenizer)
    assert reloaded.tokenizer.token_to_id == model.tokenizer.token_to_id
