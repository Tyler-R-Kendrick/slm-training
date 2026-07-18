"""Compiler-drafted constrained decoding tests."""

from __future__ import annotations

import pytest
import torch

from slm_training.dsl.grammar.fastpath.compiler_draft import (
    CompilerDecision,
    CompletionPath,
    active_declaration_binder_id,
    active_declaration_reference_count,
    active_parent_component_ids,
    binder_reference_arities,
    build_completion_forest,
    gold_compiler_decisions,
    gold_compiler_decision_positions,
    semantic_component_edges,
)
from slm_training.dsl.schema import ExampleRecord
from slm_training.data.contract import GenerationRequest
from slm_training.models.blocks import DenoiserTower
from slm_training.models.decode_stats import collect_decode_stats
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer
from slm_training.models.grammar import make_grammar_state
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel
from slm_training.harnesses.distill.trace_store import DecodeTraceRecorder
from slm_training.harnesses.preference.local_decisions import events_from_trace


def _model(**config_overrides) -> TwoTowerModel:
    record = ExampleRecord(
        id="compiler",
        prompt="card",
        openui='root = Card([title])\ntitle = TextContent(":hero.title")\n',
        placeholders=[":hero.title"],
        split="train",
        source="fixture",
    )
    config = TwoTowerConfig(
        context_backend="scratch",
        output_tokenizer="lexer",
        d_model=32,
        n_heads=2,
        context_layers=1,
        denoiser_layers=1,
        max_prompt_len=32,
        max_target_len=32,
        grammar_ltr_max_tokens=32,
        gen_steps=1,
        seed=0,
        **config_overrides,
    )
    model = TwoTowerModel.from_records([record], config=config, device="cpu")
    model.eval()
    return model


def test_gathered_projection_matches_full_lm_head() -> None:
    denoiser = DenoiserTower(vocab_size=17, d_model=8, n_layers=1, n_heads=2)
    hidden = torch.randn(2, 8)
    candidates = torch.tensor([1, 5, 11])
    gathered = denoiser.project(hidden, candidates)
    full = denoiser.project(hidden)
    assert torch.allclose(gathered, full.index_select(-1, candidates))


def test_choice_gold_decisions_classify_component_roles() -> None:
    from slm_training.models.choice_tokenizer import ChoiceTokenizer

    tokenizer = ChoiceTokenizer.build()
    source = (
        'root = Card([title])\n'
        'title = TextContent(":hero.title")'
    )
    decisions = gold_compiler_decisions(
        tokenizer,
        tokenizer.encode(source, placeholders=[":hero.title"]),
        slot_contract=[":hero.title"],
    )
    component_roles = [
        decision.kind
        for decision in decisions
        if decision.token_kind == "component"
    ]
    # Structural choice streams emit dependencies first and the root last.
    assert component_roles == ["component_bound", "component_root"]
    assert all(len(decision.candidate_ids) > 1 for decision in decisions)


def test_choice_component_plan_trains_without_surface_compiler() -> None:
    record = ExampleRecord(
        id="choice-plan",
        prompt="card with title",
        openui='root = Card([title])\ntitle = TextContent(":hero.title")',
        placeholders=[":hero.title"],
        split="train",
        source="fixture",
    )
    model = TwoTowerModel.from_records(
        [record],
        config=TwoTowerConfig(
            context_backend="scratch",
            output_tokenizer="choice",
            d_model=32,
            n_heads=2,
            context_layers=1,
            denoiser_layers=1,
            max_prompt_len=32,
            max_target_len=64,
            component_plan_loss_weight=1.0,
            component_plan_decode_weight=1.0,
            component_plan_attention_pool=True,
            seed=0,
        ),
        device="cpu",
    )
    loss = model.training_loss([record])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.component_plan_head is not None
    assert model.component_plan_head.weight.grad is not None
    assert model.component_plan_head.weight.grad.abs().sum() > 0
    assert model.component_plan_query is not None
    assert model.component_plan_query.grad is not None
    assert model.component_plan_query.grad.abs().sum() > 0
    assert model.last_training_metrics["component_plan_root_accuracy"] >= 0.0


def test_choice_component_plan_token_pool_trains_component_specific_evidence() -> None:
    record = ExampleRecord(
        id="choice-token-plan",
        prompt="card with title",
        openui='root = Card([title])\ntitle = TextContent(":hero.title")',
        placeholders=[":hero.title"],
        split="train",
        source="fixture",
    )
    model = TwoTowerModel.from_records(
        [record],
        config=TwoTowerConfig(
            context_backend="scratch",
            output_tokenizer="choice",
            d_model=32,
            n_heads=2,
            context_layers=1,
            denoiser_layers=1,
            max_prompt_len=32,
            max_target_len=64,
            component_plan_loss_weight=1.0,
            component_plan_decode_weight=1.0,
            component_plan_token_pool=True,
            seed=0,
        ),
        device="cpu",
    )

    loss = model.training_loss([record])
    loss.backward()

    assert model.component_plan_head is not None
    assert model.component_plan_head.weight.grad is not None
    assert model.component_plan_head.weight.grad.abs().sum() > 0
    assert model.component_plan_query is None
    assert model.last_training_metrics["component_plan_root_accuracy"] >= 0.0


def test_projection_with_features_accepts_sliced_hidden() -> None:
    """Compiler/tree scorers project [D] and [N,D] slices; with one request's
    runtime symbol features active the projection must match the [B,T,D]
    path instead of crashing on the feature einsum."""
    denoiser = DenoiserTower(vocab_size=17, d_model=8, n_layers=1, n_heads=2)
    features = torch.zeros((1, 17, 8))
    features[:, 3, :] = 0.5
    denoiser.set_runtime_symbol_features(features)
    hidden3 = torch.randn(1, 4, 8)
    full = denoiser.project(hidden3)
    single = denoiser.project(hidden3[0, 2])
    assert torch.allclose(single, full[0, 2], atol=1e-5)
    candidates = torch.tensor([3, 5])
    gathered = denoiser.project(hidden3[0, 2], candidates)
    assert torch.allclose(gathered, full[0, 2].index_select(-1, candidates), atol=1e-5)
    # Multi-request features cannot be attributed to a sliced row: fail closed.
    denoiser.set_runtime_symbol_features(torch.zeros((2, 17, 8)))
    with pytest.raises(ValueError, match="B,T,D"):
        denoiser.project(hidden3[0, 2])


def test_completion_forest_uses_active_binder_and_symbol_spaces(monkeypatch) -> None:
    from slm_training.dsl.grammar.fastpath import compiler_draft

    tokenizer = DSLNativeTokenizer.build()
    start = build_completion_forest(tokenizer, [])
    assert start.coverage == "complete"
    assert tokenizer.bind_id(0) in start.candidate_ids
    assert tokenizer.bind_id(1) not in start.candidate_ids
    bos_start = build_completion_forest(tokenizer, [tokenizer.bos_id])
    assert tokenizer.bind_id(0) in bos_start.candidate_ids
    assert tokenizer.bind_id(1) not in bos_start.candidate_ids
    assert tokenizer.eos_id not in bos_start.candidate_ids
    after_newline = build_completion_forest(
        tokenizer, [tokenizer.bos_id, tokenizer.token_to_id["NL"]]
    )
    assert tokenizer.token_to_id["NL"] not in after_newline.candidate_ids
    assert tokenizer.bind_id(0) in after_newline.candidate_ids
    root_value = build_completion_forest(
        tokenizer,
        [
            tokenizer.bos_id,
            tokenizer.bind_id(0),
            tokenizer.token_to_id["="],
        ],
        slot_contract=[":hero.title"],
    )
    assert root_value.candidate_ids
    assert all(
        compiler_draft._semantic_kind(tokenizer, token_id) == "component"
        for token_id in root_value.candidate_ids
    )
    child_value = build_completion_forest(
        tokenizer,
        [
            tokenizer.bos_id,
            *tokenizer.encode("root=Stack([b1])", add_special=False),
            tokenizer.token_to_id["NL"],
            tokenizer.bind_id(1),
            tokenizer.token_to_id["="],
        ],
    )
    assert child_value.candidate_ids
    assert all(
        compiler_draft._semantic_kind(tokenizer, token_id) == "component"
        for token_id in child_value.candidate_ids
    )
    children = build_completion_forest(
        tokenizer,
        [
            tokenizer.bos_id,
            *tokenizer.encode("root=Stack([", add_special=False),
        ],
        slot_contract=[":hero.title"],
    )
    assert not (set(children.candidate_ids) & set(tokenizer.kind_ids("sym")))
    assert not (set(children.candidate_ids) & set(tokenizer.kind_ids("lit")))
    assert tokenizer.bind_id(1) in children.candidate_ids
    assert tokenizer.token_to_id["Stack"] in children.candidate_ids

    nested_children = build_completion_forest(
        tokenizer,
        [
            tokenizer.bos_id,
            *tokenizer.encode("root=Stack([Stack([", add_special=False),
        ],
    )
    assert not (set(nested_children.candidate_ids) & set(tokenizer.kind_ids("lit")))

    monkeypatch.setattr(
        compiler_draft,
        "_official_schema",
        lambda: {
            "properties": {"TextContent": {}},
            "$defs": {"TextContent": {"properties": {"text": {"type": "string"}}}},
        },
    )
    prefix = tokenizer.encode("root=TextContent(", add_special=False)
    forest = build_completion_forest(
        tokenizer, prefix, slot_contract=[":hero.title"]
    )
    assert forest.coverage == "complete"
    assert tokenizer.sym_id(0) in forest.candidate_ids
    assert tokenizer.sym_id(1) not in forest.candidate_ids
    assert set(forest.candidate_ids) == {tokenizer.sym_id(0)}
    # Every emitted edge is accepted by the grammar and has a reachable
    # continuation; candidate policy must come from parser state, not from
    # a list of forbidden punctuation or component names.
    assert forest.paths
    assert all(path.token_ids for path in forest.paths)

    complete = build_completion_forest(
        tokenizer,
        [tokenizer.bos_id, *tokenizer.encode(
            'root=TextContent(":hero.title")', add_special=False
        )],
        slot_contract=[":hero.title"],
    )
    continuation_ids = set(
        compiler_draft.allowed_id_set(
            tokenizer,
            frozenset({"$END", "_NL", "NAME", "STATE_NAME", "COMMENT", "WS_INLINE"}),
        )
        or set()
    ) | {tokenizer.eos_id}
    assert set(complete.candidate_ids) <= continuation_ids
    assert tokenizer.eos_id in complete.candidate_ids


def test_min_content_contract_withholds_eos_until_components_emitted() -> None:
    # A4: an empty/underfull but grammatically complete layout must not be a
    # legal completion while the grammar still offers a way to add content.
    from slm_training.dsl.grammar.fastpath.compiler_draft import (
        emitted_component_count,
    )

    tokenizer = DSLNativeTokenizer.build()
    contract = [":hero.title"]
    prefix = [
        tokenizer.bos_id,
        *tokenizer.encode('root=TextContent(":hero.title")', add_special=False),
    ]
    assert emitted_component_count(tokenizer, prefix) == 1

    # Contract off (default): EOS admitted for the complete document.
    off = build_completion_forest(tokenizer, prefix, slot_contract=contract)
    assert tokenizer.eos_id in off.candidate_ids

    # Floor met (1 component >= 1): EOS still admitted.
    met = build_completion_forest(
        tokenizer, prefix, slot_contract=contract, min_content=1
    )
    assert tokenizer.eos_id in met.candidate_ids

    # Floor unmet (1 component < 2): EOS withheld, but a non-EOS continuation
    # remains so decode is forced to add content rather than dead-end.
    unmet = build_completion_forest(
        tokenizer, prefix, slot_contract=contract, min_content=2
    )
    assert tokenizer.eos_id not in unmet.candidate_ids
    assert unmet.paths
    assert any(path.token_ids for path in unmet.paths)


def test_effective_min_content_auto_from_inventory() -> None:
    # decode_min_content == -1 derives the floor from distinct declared slots.
    model = TwoTowerModel.from_records(
        [
            ExampleRecord(
                id="a",
                prompt="Hero",
                openui='root = Stack([t])\nt = TextContent(":hero.title")',
                placeholders=[":hero.title"],
            )
        ],
        config=TwoTowerConfig(
            d_model=32, n_heads=4, context_layers=1, denoiser_layers=1, seed=0,
            decode_min_content=-1,
        ),
        device="cpu",
    )
    assert model._effective_min_content([":hero.title", ":hero.body"]) == 2
    assert model._effective_min_content([":a.x", ":b.y"]) == 2
    assert model._effective_min_content([":hero.title", ":hero.title"]) == 1
    assert model._effective_min_content(None) == 0
    model.config.decode_min_content = 3
    assert model._effective_min_content(None) == 3
    model.config.decode_min_content = 0
    assert model._effective_min_content([":a.x", ":b.y"]) == 0


def test_complete_ast_uses_lark_when_official_parser_is_unavailable(
    monkeypatch,
) -> None:
    from slm_training.dsl import lang_core
    from slm_training.dsl.grammar.fastpath import compiler_draft

    compiler_draft._generated_ast_is_complete.cache_clear()
    monkeypatch.setattr(
        lang_core,
        "parse",
        lambda _source: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    assert compiler_draft._generated_ast_is_complete(
        'root = TextContent(":hero.title")'
    )
    assert not compiler_draft._generated_ast_is_complete("root = TextContent(")


def test_completion_forest_uses_schema_property_order_for_enums(monkeypatch) -> None:
    from slm_training.dsl.grammar.fastpath import compiler_draft

    tokenizer = DSLNativeTokenizer.build()
    schema = {
        "properties": {"Stack": {}},
        "$defs": {
            "Stack": {
                "properties": {
                    "children": {"type": "array"},
                    "direction": {"enum": ["row", "column"]},
                }
            }
        },
    }
    monkeypatch.setattr(compiler_draft, "_official_schema", lambda: schema)
    prefix = tokenizer.encode("root=Stack([],", add_special=False)
    forest = build_completion_forest(tokenizer, prefix)
    assert forest.coverage == "complete"
    assert set(forest.candidate_ids) == {
        tokenizer.token_to_id["STR:row"],
        tokenizer.token_to_id["STR:column"],
    }
    completed_enum = build_completion_forest(
        tokenizer,
        tokenizer.encode('root=Stack([],"column"', add_special=False),
    )
    assert set(completed_enum.candidate_ids) == {tokenizer.token_to_id[")"]}


def test_completion_forest_enforces_generated_schema_arity(monkeypatch) -> None:
    from slm_training.dsl.grammar.fastpath import compiler_draft

    tokenizer = DSLNativeTokenizer.build()
    monkeypatch.setattr(
        compiler_draft,
        "_official_schema",
        lambda: {
            "properties": {"SelectItem": {}},
            "$defs": {
                "SelectItem": {
                    "properties": {
                        "value": {"type": "string"},
                        "label": {"type": "string"},
                    },
                    "required": ["value", "label"],
                }
            },
        },
    )

    first = build_completion_forest(
        tokenizer, tokenizer.encode('root=SelectItem(":value"', add_special=False)
    )
    assert set(first.candidate_ids) == {tokenizer.token_to_id[","]}

    complete = build_completion_forest(
        tokenizer,
        tokenizer.encode('root=SelectItem(":value",":label"', add_special=False),
    )
    assert set(complete.candidate_ids) == {tokenizer.token_to_id[")"]}


def test_completion_forest_enforces_lexer_literal_frame() -> None:
    tokenizer = DSLNativeTokenizer.build()
    contract = [":value", ":label"]
    prefix = tokenizer.encode(
        'root=RadioItem(":value",":label",', add_special=False
    )
    outside = build_completion_forest(tokenizer, prefix, slot_contract=contract)
    assert tokenizer.token_to_id["LIT_STR"] in outside.candidate_ids
    assert tokenizer.token_to_id["LIT_END"] not in outside.candidate_ids

    inside = build_completion_forest(
        tokenizer,
        [*prefix, tokenizer.token_to_id["LIT_STR"]],
        slot_contract=contract,
    )
    from slm_training.models.dsl_tokenizer import TokenKind

    expected = tokenizer.kind_ids(TokenKind.BYTE) | {
        tokenizer.token_to_id["LIT_END"]
    }
    assert set(inside.candidate_ids) <= expected
    assert tokenizer.token_to_id["LIT_END"] in inside.candidate_ids
    assert set(inside.candidate_ids) & tokenizer.kind_ids(TokenKind.BYTE)
    assert not (set(inside.candidate_ids) & tokenizer.kind_ids(TokenKind.SYM))

    closed = build_completion_forest(
        tokenizer,
        [
            *prefix,
            tokenizer.token_to_id["LIT_STR"],
            tokenizer.token_to_id["LIT_END"],
        ],
        slot_contract=contract,
    )
    assert set(closed.candidate_ids) == {tokenizer.token_to_id[")"]}


def test_completion_forest_encodes_generated_enum_with_literal_channel(
    monkeypatch,
) -> None:
    from slm_training.dsl.grammar.fastpath import compiler_draft

    tokenizer = DSLNativeTokenizer.build()
    schema = {
        "$defs": {
            "Stack": {
                "properties": {
                    "children": {"type": "array"},
                    "align": {"type": "string", "enum": ["schema-only"]},
                },
                "required": ["children"],
            }
        },
        "properties": {"Stack": {}},
    }
    monkeypatch.setattr(compiler_draft, "_official_schema", lambda: schema)
    prefix = [
        tokenizer.bos_id,
        *tokenizer.encode("root=Stack([] ,", add_special=False),
    ]
    forest = build_completion_forest(tokenizer, prefix)
    expected = tuple(tokenizer.encode('"schema-only"', add_special=False))
    assert [path.token_ids[: len(expected)] for path in forest.paths] == [expected]


def test_active_call_ignores_nested_array_and_call_commas() -> None:
    from slm_training.dsl.grammar.fastpath import compiler_draft
    from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine

    engine = OpenUIIncrementalEngine()
    assert engine.set_prefix('root=Stack([TextContent(":a"),TextContent(":b")')
    assert compiler_draft._active_call(engine) == ("Stack", 0, 1)

    assert engine.set_prefix(
        'root=Form("name",Buttons([]),[FormControl(":label",":input")'
    )
    assert compiler_draft._active_call(engine) == ("Form", 2, 3)


def test_completion_forest_tracks_forward_binder_scope() -> None:
    tokenizer = DSLNativeTokenizer.build()
    first = [
        tokenizer.bos_id,
        *tokenizer.encode("root=Stack([", add_special=False),
    ]
    forest = build_completion_forest(tokenizer, first)
    assert tokenizer.bind_id(1) in forest.candidate_ids
    assert tokenizer.bind_id(2) not in forest.candidate_ids
    assert tokenizer.bind_id(0) not in forest.candidate_ids

    second = [*first, tokenizer.bind_id(1), tokenizer.token_to_id[","]]
    forest = build_completion_forest(tokenizer, second)
    assert tokenizer.bind_id(1) in forest.candidate_ids
    assert tokenizer.bind_id(2) in forest.candidate_ids

    declaration = [
        tokenizer.bos_id,
        *tokenizer.encode("root=Stack([b1,b2])", add_special=False),
        tokenizer.token_to_id["NL"],
    ]
    forest = build_completion_forest(tokenizer, declaration)
    assert set(forest.candidate_ids) & set(tokenizer.kind_ids("bind")) == {
        tokenizer.bind_id(1)
    }

    unresolved = [
        tokenizer.bos_id,
        *tokenizer.encode('root=Stack([b1],"column")', add_special=False),
    ]
    forest = build_completion_forest(tokenizer, unresolved)
    assert tokenizer.eos_id not in forest.candidate_ids
    postfix_ids = {
        tokenizer.token_to_id[token]
        for token in ("[", ".", "?", "+", "-", "*", "/", "%", ">", "<")
    }
    assert not (set(forest.candidate_ids) & postfix_ids)

    resolved = [
        tokenizer.bos_id,
        *tokenizer.encode(
            'root=Stack([b1],"column")\nb1=TextContent(":hero.title")',
            add_special=False,
        ),
    ]
    forest = build_completion_forest(
        tokenizer, resolved, slot_contract=[":hero.title"]
    )
    assert tokenizer.eos_id in forest.candidate_ids


def test_gold_decisions_follow_compiler_forest() -> None:
    tokenizer = DSLNativeTokenizer.build()
    target = tokenizer.encode(
        'root=Card([title,body])\ntitle=TextContent(":hero.title")\n'
        'body=TextContent(":hero.body")',
        add_special=True,
    )
    positions = gold_compiler_decision_positions(
        tokenizer, target, slot_contract=[":hero.title", ":hero.body"]
    )
    selected = {tokenizer.id_to_token[target[position]] for position in positions}
    assert {"<BIND_0>", "Card", "<BIND_1>", "TextContent"} <= selected
    kinds = {decision.kind for decision in gold_compiler_decisions(tokenizer, target)}
    assert {
        "bind_declaration_root",
        "bind_reference_root_children",
        "component_root",
        "component_bound",
    } <= kinds
    assert "grammar_rsqb_root_populated" in kinds
    assert "grammar_comma" in kinds
    decisions = gold_compiler_decisions(tokenizer, target)
    assert all(len(decision.candidate_ids) > 1 for decision in decisions)
    assert all(
        target[decision.position] in decision.candidate_ids for decision in decisions
    )
    assert all(
        decision.is_semantic_role
        for decision in decisions
        if decision.kind.startswith(("component_", "bind_"))
    )
    assert all(
        not decision.is_semantic_role
        for decision in decisions
        if decision.kind.startswith("grammar_")
    )
    empty = tokenizer.encode("root=Stack([])", add_special=True)
    assert "grammar_rsqb_root_empty" in {
        decision.kind for decision in gold_compiler_decisions(tokenizer, empty)
    }
    bound_empty = tokenizer.encode(
        "root=Stack([child])\nchild=Stack([])", add_special=True
    )
    assert "grammar_rsqb_bound_empty" in {
        decision.kind
        for decision in gold_compiler_decisions(tokenizer, bound_empty)
    }


def test_grammar_state_advances_lexer_literals_as_source() -> None:
    tokenizer = DSLNativeTokenizer.build()
    state = make_grammar_state()
    prefix = [
        tokenizer.bos_id,
        *tokenizer.encode('root=Stack([Form("m"', add_special=False),
    ]
    for token_id in prefix[1:]:
        state.advance_token(tokenizer, token_id)
    forest = build_completion_forest(tokenizer, prefix, state=state)
    assert state.prefix_text.endswith('Form("m"')
    assert set(forest.candidate_ids) == {tokenizer.token_to_id[","]}


def test_compiler_alignment_loss_trains_gold_semantic_states() -> None:
    model = _model()
    model.config.compiler_alignment_loss_weight = 1.0
    record = ExampleRecord(
        id="alignment",
        prompt="card",
        openui='root = Card([title])\ntitle = TextContent(":hero.title")',
        placeholders=[":hero.title"],
        split="train",
        source="fixture",
    )
    loss = model.training_loss([record])
    assert torch.isfinite(loss)
    assert model.last_training_metrics["compiler_alignment_rows"] == 1
    assert model.last_training_metrics["compiler_alignment_loss"] > 0.0
    assert model.last_training_metrics["compiler_alignment_candidate_count_mean"] > 1


def test_compiler_alignment_margin_reports_legal_branch_violations() -> None:
    model = _model()
    model.config.compiler_alignment_loss_weight = 1.0
    model.config.compiler_alignment_margin = 100.0
    record = ExampleRecord(
        id="alignment-margin",
        prompt="card",
        openui='root = Card([title])\ntitle = TextContent(":hero.title")',
        placeholders=[":hero.title"],
        split="train",
        source="fixture",
    )
    loss = model.training_loss([record])
    assert torch.isfinite(loss)
    assert model.last_training_metrics["compiler_alignment_margin_loss"] > 0.0
    assert model.last_training_metrics["compiler_alignment_margin_violation_rate"] == 1.0


def test_compiler_alignment_can_stratify_grammar_decision_kinds() -> None:
    model = _model()
    model.config.compiler_alignment_loss_weight = 1.0
    model.config.compiler_alignment_stratified = True
    record = ExampleRecord(
        id="alignment-stratified",
        prompt="card",
        openui=(
            'root = Card([title, body])\n'
            'title = TextContent(":hero.title")\n'
            'body = TextContent(":hero.body")'
        ),
        placeholders=[":hero.title", ":hero.body"],
        split="train",
        source="fixture",
    )
    loss = model.training_loss([record])
    assert torch.isfinite(loss)
    metrics = model.last_training_metrics
    assert metrics["compiler_alignment_rows"] > 1
    assert metrics["compiler_alignment_bind_declaration_root_rows"] == 1
    assert "compiler_alignment_bind_declaration_bound_rows" not in metrics
    assert metrics["compiler_alignment_bind_reference_root_children_rows"] == 1
    assert metrics["compiler_alignment_component_root_rows"] == 1
    assert metrics["compiler_alignment_component_bound_rows"] == 1
    assert metrics["compiler_alignment_grammar_rsqb_root_populated_rows"] == 1
    assert metrics["compiler_alignment_grammar_comma_rows"] == 1
    assert metrics["compiler_alignment_grammar_rsqb_root_populated_loss"] >= 0.0


def test_compiler_alignment_skips_gold_outside_legal_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    model.config.compiler_alignment_loss_weight = 1.0
    model.config.compiler_alignment_stratified = True
    model.config.compiler_alignment_semantic_exhaustive = True
    record = ExampleRecord(
        id="alignment-invalid-candidate",
        prompt="card",
        openui='root = Card([title])\ntitle = TextContent(":hero.title")',
        placeholders=[":hero.title"],
        split="train",
        source="fixture",
    )
    monkeypatch.setattr(
        "slm_training.dsl.grammar.fastpath.compiler_draft.gold_compiler_decisions",
        lambda *_args, **_kwargs: (
            CompilerDecision(
                position=1,
                kind="component_root",
                token_kind="component",
                candidate_ids=(model.tokenizer.mask_id,),
            ),
        ),
    )

    loss = model.training_loss([record])

    assert torch.isfinite(loss)
    assert model.last_training_metrics["compiler_alignment_rows"] == 0
    assert (
        model.last_training_metrics[
            "compiler_alignment_gold_outside_candidate_rows"
        ]
        == 1
    )


def test_component_inventory_supervision_trains_prompt_level_component_set() -> None:
    model = _model(component_inventory_loss_weight=1.0)
    model.train()
    record = ExampleRecord(
        id="inventory",
        prompt="card with a title",
        openui='root = Card([title])\ntitle = TextContent(":hero.title")',
        placeholders=[":hero.title"],
        split="train",
        source="fixture",
    )

    loss = model.training_loss([record])
    loss.backward()

    assert torch.isfinite(loss)
    assert model.component_inventory_head is not None
    assert model.component_inventory_head.weight.grad is not None
    assert model.component_inventory_head.weight.grad.abs().sum() > 0
    metrics = model.last_training_metrics
    assert metrics["component_inventory_loss"] > 0.0
    assert 0.0 <= metrics["component_inventory_topk_recall"] <= 1.0
    assert metrics["component_inventory_positive_count_mean"] == 2.0


def test_component_inventory_bias_only_scores_compiler_legal_components() -> None:
    model = _model(component_inventory_decode_weight=2.0)
    assert model.component_inventory_head is not None
    tokenizer = model.tokenizer
    card = tokenizer.token_to_id["Card"]
    stack = tokenizer.token_to_id["Stack"]
    lparen = tokenizer.token_to_id["("]
    with torch.no_grad():
        model.component_inventory_head.weight.zero_()
        model.component_inventory_head.bias.zero_()
        model.component_inventory_head.bias[card] = 3.0
        model.component_inventory_head.bias[stack] = -1.0
    ctx, ctx_pad = model._encode_context(["card"])

    bias = model._component_inventory_bias(ctx, ctx_pad, (stack, lparen, card))

    assert bias is not None
    assert bias.tolist() == [-2.0, 0.0, 6.0]


def test_component_plan_supervises_root_role_and_bound_counts() -> None:
    model = _model(component_plan_loss_weight=1.0)
    model.train()
    record = ExampleRecord(
        id="component-plan",
        prompt="card with two labels",
        openui=(
            'root = Card([title, body])\n'
            'title = TextContent(":hero.title")\n'
            'body = TextContent(":hero.body")'
        ),
        placeholders=[":hero.title", ":hero.body"],
        split="train",
        source="fixture",
    )

    loss = model.training_loss([record])
    loss.backward()

    assert torch.isfinite(loss)
    assert model.component_plan_head is not None
    assert model.component_plan_head.weight.grad is not None
    metrics = model.last_training_metrics
    assert metrics["component_plan_loss"] > 0.0
    assert metrics["component_plan_bound_loss"] >= 0.0
    assert 0.0 <= metrics["component_plan_root_accuracy"] <= 1.0
    assert 0.0 <= metrics["component_plan_bound_topk_recall"] <= 1.0
    assert metrics["component_plan_bound_count_mae"] >= 0.0


def test_component_plan_bias_is_role_conditioned_and_count_aware() -> None:
    model = _model(component_plan_decode_weight=2.0)
    assert model.component_plan_head is not None
    tokenizer = model.tokenizer
    card = tokenizer.token_to_id["Card"]
    text = tokenizer.token_to_id["TextContent"]
    stack = tokenizer.token_to_id["Stack"]
    with torch.no_grad():
        model.component_plan_head.weight.zero_()
        model.component_plan_head.bias.zero_()
        vocab = tokenizer.vocab_size
        model.component_plan_head.bias[card] = 3.0
        model.component_plan_head.bias[vocab + text] = 4.0
    ctx, ctx_pad = model._encode_context(["card with labels"])

    root_bias = model._component_plan_bias(
        ctx,
        ctx_pad,
        [],
        (stack, card),
        ("component_root", "component_root"),
    )
    bound_bias_before = model._component_plan_bias(
        ctx,
        ctx_pad,
        [card],
        (card, text),
        ("component_bound", "component_bound"),
    )
    bound_bias_after = model._component_plan_bias(
        ctx,
        ctx_pad,
        [card, text],
        (card, text),
        ("component_bound", "component_bound"),
    )

    assert root_bias is not None and root_bias.tolist() == [0.0, 6.0]
    assert bound_bias_before is not None
    assert bound_bias_after is not None
    assert bound_bias_before[1] > bound_bias_before[0]
    assert bound_bias_after[1] < bound_bias_before[1]


def test_slot_component_supervises_each_visible_slot_owner() -> None:
    model = _model(
        slot_component_loss_weight=1.0,
        slot_component_focal_gamma=2.0,
    )
    model.train()
    record = ExampleRecord(
        id="slot-components",
        prompt="email field and submit action",
        openui=(
            'root = Stack([field, submit])\n'
            'field = Input("email", ":form.email")\n'
            'submit = Button(":form.submit")'
        ),
        placeholders=[":form.email", ":form.submit"],
        split="train",
        source="fixture",
    )

    loss = model.training_loss([record])
    loss.backward()

    assert torch.isfinite(loss)
    assert model.slot_component_head is not None
    assert model.slot_component_head.weight.grad is not None
    assert model.slot_component_head.weight.grad.abs().sum() > 0
    assert model.last_training_metrics["slot_component_rows"] == 2
    assert model.last_training_metrics["slot_component_loss"] > 0
    assert 0 <= model.last_training_metrics["slot_component_accuracy"] <= 1
    assert 0 < model.last_training_metrics["slot_component_majority_baseline"] <= 1


def test_slot_component_class_weights_come_from_training_owners() -> None:
    records = [
        ExampleRecord(
            id=f"text-{index}",
            prompt="text",
            openui=f'root = TextContent(":text.{index}")',
            placeholders=[f":text.{index}"],
            split="train",
            source="fixture",
        )
        for index in range(2)
    ]
    records.append(
        ExampleRecord(
            id="input",
            prompt="input",
            openui='root = Input("email", ":form.email")',
            placeholders=[":form.email"],
            split="train",
            source="fixture",
        )
    )
    model = TwoTowerModel.from_records(
        records,
        config=TwoTowerConfig(
            context_backend="scratch",
            output_tokenizer="lexer",
            d_model=32,
            n_heads=2,
            context_layers=1,
            denoiser_layers=1,
            slot_component_loss_weight=1.0,
            slot_component_class_balance_power=0.5,
        ),
        device="cpu",
    )
    component_index = model._component_name_index()
    weights = model.config.slot_component_class_weights

    assert len(weights) == len(component_index)
    assert weights[component_index["Input"]] > weights[component_index["TextContent"]]


def test_slot_component_can_exclude_whole_prompt_context() -> None:
    model = _model(
        slot_component_loss_weight=1.0,
        slot_component_prompt_context=False,
    )
    model.eval()
    first, first_pad = model._encode_context(["email field"])
    second, second_pad = model._encode_context(["unrelated dashboard"])

    with torch.no_grad():
        first_logits = model._slot_component_logits(
            [":form.email"], first, first_pad, torch.tensor([0])
        )
        second_logits = model._slot_component_logits(
            [":form.email"], second, second_pad, torch.tensor([0])
        )

    assert torch.equal(first_logits, second_logits)


def test_slot_component_next_context_preserves_order() -> None:
    model = _model(slot_component_next_context=True)

    assert model._slot_component_texts(
        [":hint.title", ":hint.body", ":submit"]
    ) == [
        ":hint.title\n:hint.body",
        ":hint.body\n:submit",
        ":submit",
    ]


def test_slot_component_pair_interaction_is_explicit() -> None:
    model = _model(
        slot_component_loss_weight=1.0,
        slot_component_prompt_context=False,
        slot_component_pair_interaction=True,
    )
    model.eval()
    ctx, ctx_pad = model._encode_context(["form"])
    rows = torch.tensor([0])

    with torch.no_grad():
        baseline = model._slot_component_logits(
            [":hint.title"], ctx, ctx_pad, rows
        )
        missing = model._slot_component_logits(
            [":hint.title"], ctx, ctx_pad, rows, next_slots=[None]
        )
        paired = model._slot_component_logits(
            [":hint.title"], ctx, ctx_pad, rows, next_slots=[":hint.body"]
        )

    assert torch.equal(baseline, missing)
    assert not torch.equal(baseline, paired)


def test_slot_component_lexeme_prior_changes_matching_slot_only() -> None:
    model = _model(
        slot_component_loss_weight=1.0,
        slot_component_prompt_context=False,
    )
    model.eval()
    ctx, ctx_pad = model._encode_context(["form"])
    rows = torch.tensor([0])
    classes = len(model._component_name_index())
    scores = [0.0] * classes
    scores[model._component_name_index()["Input"]] = 2.0

    with torch.no_grad():
        baseline = model._slot_component_logits(
            [":form.email"], ctx, ctx_pad, rows
        )
        model.config.slot_component_lexeme_prior_weight = 1.0
        model.config.slot_component_lexeme_priors = (("email", tuple(scores)),)
        matching = model._slot_component_logits(
            [":form.email"], ctx, ctx_pad, rows
        )
        unrelated = model._slot_component_logits(
            [":form.title"], ctx, ctx_pad, rows
        )

    assert matching[0, model._component_name_index()["Input"]] == (
        baseline[0, model._component_name_index()["Input"]] + 2.0
    )
    assert not torch.equal(matching, unrelated)


def test_slot_component_bias_uses_next_unfilled_slot() -> None:
    from types import MethodType

    model = _model(
        slot_component_decode_weight=2.0,
        slot_component_content_arity=True,
    )
    assert model.slot_component_head is not None
    tokenizer = model.tokenizer
    component_ids = model._component_inventory_token_ids()
    component_index = {
        tokenizer.id_to_token[token_id]: index
        for index, token_id in enumerate(component_ids)
    }

    def logits(self, slots, context, pad_mask, context_rows):
        rows = torch.zeros(
            (len(slots), len(component_ids)),
            dtype=context.dtype,
            device=context.device,
        )
        for index, slot in enumerate(slots):
            target = "Input" if slot == ":form.email" else "Button"
            rows[index, component_index[target]] = 3.0
        return rows

    model._slot_component_logits = MethodType(logits, model)
    ctx, ctx_pad = model._encode_context(
        ["email field and submit action; slots :form.email, :form.submit"]
    )
    stack = tokenizer.token_to_id["Stack"]
    input_id = tokenizer.token_to_id["Input"]
    button = tokenizer.token_to_id["Button"]
    candidates = (input_id, button)
    kinds = ("component_bound", "component_bound")

    email_bias = model._slot_component_bias(
        ctx,
        ctx_pad,
        [stack],
        candidates,
        kinds,
        [":form.email", ":form.submit"],
    )
    submit_bias = model._slot_component_bias(
        ctx,
        ctx_pad,
        [stack, tokenizer.sym_id(0)],
        candidates,
        kinds,
        [":form.email", ":form.submit"],
    )

    assert email_bias is not None and email_bias[0] > email_bias[1]
    assert submit_bias is not None and submit_bias[1] > submit_bias[0]


def test_slot_component_bias_combines_consumed_slots_and_span_prior() -> None:
    from types import MethodType

    model = _model(slot_component_decode_weight=2.0)
    tokenizer = model.tokenizer
    component_ids = model._component_inventory_token_ids()
    def logits(self, slots, context, pad_mask, context_rows):
        rows = torch.zeros(
            (len(slots), len(component_ids)),
            dtype=context.dtype,
            device=context.device,
        )
        rows[0, 0] = 4.0
        rows[1, 0] = 2.0
        return rows

    model._slot_component_logits = MethodType(logits, model)
    ctx, ctx_pad = model._encode_context(["faq title and body"])
    accordion, text = component_ids[:2]

    def slot_content_count(self, token_id):
        return 2 if token_id == accordion else 1

    tokenizer.slot_content_count = MethodType(slot_content_count, tokenizer)
    scores = [0.0] * len(component_ids)
    scores[0] = 2.0
    model.config.slot_component_span_prior_weight = 1.0
    model.config.slot_component_span_priors = (
        ("title\x1fbody", tuple(scores)),
    )
    bias = model._slot_component_bias(
        ctx,
        ctx_pad,
        [tokenizer.bos_id],
        (accordion, text),
        ("component_bound", "component_bound"),
        [":faq.title", ":faq.body"],
    )

    assert tokenizer.slot_content_count(accordion) == 2
    assert tokenizer.slot_content_count(text) == 1
    assert bias is not None
    assert bias.tolist() == [8.0, 0.0]


def test_component_edges_come_from_ast_and_partial_reference_graph() -> None:
    tokenizer = DSLNativeTokenizer.build()
    root = {
        "type": "element",
        "typeName": "Card",
        "props": {
            "children": [
                {
                    "type": "element",
                    "typeName": "TextContent",
                    "props": {"text": ":title"},
                }
            ]
        },
    }
    card = tokenizer.token_to_id["Card"]
    text = tokenizer.token_to_id["TextContent"]
    assert semantic_component_edges(root, tokenizer) == ((card, text),)
    prefix = tokenizer.encode("root = Card([title])\ntitle =", add_special=False)
    assert active_parent_component_ids(tokenizer, prefix) == (card,)
    assert active_parent_component_ids(tokenizer, [tokenizer.bos_id, *prefix]) == (
        card,
    )
    assert active_declaration_binder_id(tokenizer, prefix) == tokenizer.bind_id(1)


def test_binder_reference_arities_follow_grammar_token_roles() -> None:
    tokenizer = DSLNativeTokenizer.build()
    ids = tokenizer.encode(
        'root = Card([title, body])\ntitle = TextContent(":hero.title")\n'
        'body = Stack([copy], "column")\ncopy = TextContent(":hero.body")',
        add_special=False,
    )
    assert binder_reference_arities(tokenizer, ids) == (
        (tokenizer.bind_id(0), 2),
        (tokenizer.bind_id(1), 0),
        (tokenizer.bind_id(2), 1),
        (tokenizer.bind_id(3), 0),
    )
    prefix = tokenizer.encode("root = Card([title,", add_special=False)
    assert active_declaration_reference_count(tokenizer, prefix) == 1


def test_component_edge_supervision_and_parent_conditioned_bias() -> None:
    model = _model(
        component_edge_loss_weight=1.0,
        component_edge_alignment_loss_weight=1.0,
        component_edge_decode_weight=2.0,
    )
    model.train()
    record = ExampleRecord(
        id="component-edge",
        prompt="card with a title",
        openui='root = Card([title])\ntitle = TextContent(":hero.title")',
        placeholders=[":hero.title"],
        split="train",
        source="fixture",
    )
    loss = model.training_loss([record])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.component_edge_head is not None
    assert model.component_edge_head.weight.grad is not None
    assert model.component_edge_head.weight.grad.abs().sum() > 0
    assert model.last_training_metrics["component_edge_loss"] > 0
    assert 0 <= model.last_training_metrics["component_edge_topk_recall"] <= 1
    assert model.last_training_metrics["component_edge_positive_count_mean"] == 1
    assert model.last_training_metrics["component_edge_alignment_rows"] == 1
    assert model.last_training_metrics["component_edge_alignment_loss"] > 0
    assert (
        model.last_training_metrics["component_edge_alignment_unknown_parent_rows"]
        == 0
    )
    assert 0 <= model.last_training_metrics["component_edge_alignment_accuracy"] <= 1

    tokenizer = model.tokenizer
    components = model._component_inventory_token_ids()
    component_index = {token_id: i for i, token_id in enumerate(components)}
    card = tokenizer.token_to_id["Card"]
    text = tokenizer.token_to_id["TextContent"]
    with torch.no_grad():
        model.component_edge_head.weight.zero_()
        model.component_edge_head.bias.zero_()
        model.component_edge_head.bias[
            component_index[card] * len(components) + component_index[text]
        ] = 3.0
    ctx, ctx_pad = model._encode_context(["card with a title"])
    prefix = tokenizer.encode("root = Card([title])\ntitle =", add_special=False)
    bias = model._component_edge_bias(
        ctx,
        ctx_pad,
        prefix,
        (card, text),
        ("component_bound", "component_bound"),
    )
    assert bias is not None
    assert bias[1] > bias[0]


def test_binder_component_plan_supervises_instances_and_biases_legal_choices() -> None:
    model = _model(
        binder_component_plan_loss_weight=1.0,
        binder_component_plan_decode_weight=2.0,
    )
    model.train()
    record = ExampleRecord(
        id="binder-component-plan",
        prompt="card with title and body",
        openui=(
            'root = Card([title, body])\n'
            'title = TextContent(":hero.title")\n'
            'body = TextContent(":hero.body")'
        ),
        placeholders=[":hero.title", ":hero.body"],
        split="train",
        source="fixture",
    )
    loss = model.training_loss([record])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.binder_component_plan_head is not None
    assert model.binder_component_plan_head.weight.grad is not None
    assert model.binder_component_plan_head.weight.grad.abs().sum() > 0
    assert model.last_training_metrics["binder_component_plan_rows"] == 2
    assert model.last_training_metrics["binder_component_plan_loss"] > 0

    tokenizer = model.tokenizer
    binders = model._binder_component_token_ids()
    components = model._component_inventory_token_ids()
    card = tokenizer.token_to_id["Card"]
    text = tokenizer.token_to_id["TextContent"]
    binder = binders.index(tokenizer.bind_id(1))
    child = components.index(text)
    with torch.no_grad():
        model.binder_component_plan_head.weight.zero_()
        model.binder_component_plan_head.bias.zero_()
        model.binder_component_plan_head.bias[binder * len(components) + child] = 3.0
    ctx, ctx_pad = model._encode_context(["card with title and body"])
    prefix = tokenizer.encode("root = Card([title])\ntitle =", add_special=False)
    bias = model._binder_component_plan_bias(
        ctx,
        ctx_pad,
        prefix,
        (card, text),
        ("component_bound", "component_bound"),
    )
    assert bias is not None
    assert bias[1] > bias[0]


def test_binder_topology_supervises_and_biases_legal_references() -> None:
    model = _model(
        binder_topology_loss_weight=1.0,
        binder_topology_decode_weight=2.0,
    )
    model.train()
    record = ExampleRecord(
        id="binder-topology",
        prompt="card with title and body",
        openui=(
            'root = Card([title, body])\n'
            'title = TextContent(":hero.title")\n'
            'body = TextContent(":hero.body")'
        ),
        placeholders=[":hero.title", ":hero.body"],
        split="train",
        source="fixture",
    )
    loss = model.training_loss([record])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.binder_topology_head is not None
    assert model.binder_topology_head.weight.grad is not None
    assert model.binder_topology_head.weight.grad.abs().sum() > 0
    assert model.last_training_metrics["binder_topology_rows"] > 0
    assert model.last_training_metrics["binder_topology_loss"] > 0

    tokenizer = model.tokenizer
    binders = model._binder_component_token_ids()
    root = binders.index(tokenizer.bind_id(0))
    child = binders.index(tokenizer.bind_id(2))
    with torch.no_grad():
        model.binder_topology_head.weight.zero_()
        model.binder_topology_head.bias.zero_()
        model.binder_topology_head.bias[root * len(binders) + child] = 3.0
    ctx, ctx_pad = model._encode_context(["card with title and body"])
    prefix = tokenizer.encode("root = Card([title,", add_special=False)
    candidates = (tokenizer.bind_id(1), tokenizer.bind_id(2))
    bias = model._binder_topology_bias(
        ctx,
        ctx_pad,
        prefix,
        candidates,
        ("bind_reference_root_children", "bind_reference_root_children"),
    )
    assert bias is not None
    assert bias[1] > bias[0]


def test_binder_arity_supervises_and_biases_continue_stop_paths() -> None:
    model = _model(
        binder_arity_loss_weight=1.0,
        binder_arity_decode_weight=2.0,
    )
    model.train()
    record = ExampleRecord(
        id="binder-arity",
        prompt="card with title and body",
        openui=(
            'root = Card([title, body])\n'
            'title = TextContent(":hero.title")\n'
            'body = TextContent(":hero.body")'
        ),
        placeholders=[":hero.title", ":hero.body"],
        split="train",
        source="fixture",
    )
    loss = model.training_loss([record])
    loss.backward()
    auxiliary_loss = model.take_detached_auxiliary_loss()
    assert auxiliary_loss is not None
    auxiliary_loss.backward()
    assert torch.isfinite(loss)
    assert model.binder_arity_head is not None
    assert model.binder_arity_head.weight.grad is not None
    assert model.binder_arity_head.weight.grad.abs().sum() > 0
    assert model.last_training_metrics["binder_arity_rows"] == 3
    assert model.last_training_metrics["binder_arity_loss"] > 0

    tokenizer = model.tokenizer
    binders = model._binder_component_token_ids()
    root = binders.index(tokenizer.bind_id(0))
    buckets = len(binders) + 1
    with torch.no_grad():
        model.binder_arity_head.weight.zero_()
        model.binder_arity_head.bias.zero_()
        model.binder_arity_head.bias[root * buckets + 2] = 3.0
    ctx, ctx_pad = model._encode_context(["card with title and body"])
    prefix = tokenizer.encode("root = Card([title", add_special=False)
    paths = (
        CompletionPath(
            (tokenizer.token_to_id[","], tokenizer.bind_id(2)),
            "grammar_comma",
        ),
        CompletionPath(
            (tokenizer.token_to_id["]"], tokenizer.token_to_id[")"]),
            "grammar_rsqb",
        ),
    )
    bias = model._binder_arity_path_bias(ctx, ctx_pad, prefix, paths)
    assert bias is not None
    assert bias[0] > bias[1]


def test_tree_verifier_packs_prefix_nodes_and_avoids_full_projection() -> None:
    model = _model()
    model.config.structural_bias = 0.0
    tokenizer = model.tokenizer
    prefix = [tokenizer.bos_id, *tokenizer.encode("root=", add_special=False)]
    paths = (
        CompletionPath(
            (tokenizer.token_to_id["Card"], tokenizer.token_to_id["("]),
            "component",
        ),
        CompletionPath(
            (tokenizer.token_to_id["Stack"], tokenizer.token_to_id["("]),
            "component",
        ),
    )
    ctx, ctx_pad = model._encode_context(["card"])
    canvas = model._compiler_canvas(prefix, 24)
    full = model.denoiser(
        canvas,
        ctx,
        pad_id=tokenizer.pad_id,
        ctx_pad_mask=ctx_pad,
    )[0, len(prefix)]
    expected = max(paths, key=lambda path: float(full[path.token_ids[0]].item()))
    with collect_decode_stats() as stats:
        selected = model._select_compiler_path(
            prefix, paths, ctx, ctx_pad, 24, tree=True
        )
    assert selected in {path.token_ids for path in paths}
    assert selected == expected.token_ids
    assert stats.trie_nodes >= 3
    assert stats.restricted_projections >= 1
    assert stats.full_projections == 0
    assert stats.constrained_selection_traces[0]["phase"] == "compiler_tree"
    assert stats.constrained_selection_traces[0]["top_candidates"]
    assert "first_edge_score" in stats.constrained_selection_traces[0][
        "top_candidates"
    ][0]


def test_tree_verifier_records_exact_branch_support_for_event_mining(
    monkeypatch,
) -> None:
    model = _model()
    model.config.structural_bias = 0.0
    tokenizer = model.tokenizer
    prefix = [tokenizer.bos_id, *tokenizer.encode("root=", add_special=False)]
    paths = (
        CompletionPath(
            (tokenizer.token_to_id["Card"], tokenizer.token_to_id["("]),
            "component",
        ),
        CompletionPath(
            (tokenizer.token_to_id["Stack"], tokenizer.token_to_id["("]),
            "component",
        ),
    )
    ctx, ctx_pad = model._encode_context(["card"])
    original_project = model.denoiser.project

    def project(hidden, candidate_ids=None):
        scores = original_project(hidden, candidate_ids)
        if candidate_ids is None:
            scores = scores.clone()
            scores[tokenizer.mask_id] = scores.max() + 100.0
        return scores

    monkeypatch.setattr(model.denoiser, "project", project)
    recorder = DecodeTraceRecorder(record_support=True)
    model.trace_recorder = recorder
    selected = model._select_compiler_path(prefix, paths, ctx, ctx_pad, 24, tree=True)
    model.trace_recorder = None

    trace = recorder.finalize(
        record_id="record-1",
        context_text="card",
        policy_checkpoint_sha="checkpoint-sha",
        tokenizer_sha="tokenizer-sha",
        decode_config_hash="decode-sha",
        seed=0,
    )
    trace["trajectory_id"] = "trajectory-1"
    commits = [commit for step in trace["steps"] for commit in step["commits"]]
    assert commits
    assert commits[0]["id"] == selected[0]
    assert commits[0]["raw_id"] == tokenizer.mask_id
    assert set(commits[0]["allowed_id_set"]) == {
        tokenizer.token_to_id["Card"],
        tokenizer.token_to_id["Stack"],
    }
    assert commits[0]["pre_canvas"][: len(prefix)] == prefix
    events = events_from_trace(trace)
    assert len(events) == 1
    assert events[0].good_token_ids == (selected[0],)
    assert events[0].bad_token_ids == (tokenizer.mask_id,)


def test_maskgit_fallback_keeps_compiler_prefix_visible() -> None:
    model = _model()
    tokenizer = model.tokenizer
    seed = [tokenizer.bos_id, *tokenizer.encode("root=Stack(", add_special=False)]
    ctx, ctx_pad = model._encode_context(["card"])
    text = model._generate_maskgit_one(
        ctx,
        ctx_pad,
        24,
        use_grammar=False,
        seed_ids=seed,
    )
    # The lexer stores root as binding zero internally, but public output is
    # canonical OpenUI and must preserve the seeded root component surface.
    assert text.startswith("root = Stack(")


def test_compiler_decode_is_opt_in() -> None:
    assert TwoTowerConfig().compiler_decode_mode == "off"
    assert TwoTowerConfig().compiler_search_mode == "greedy"


def test_compiler_decode_reserves_room_beyond_predicted_length(monkeypatch) -> None:
    model = _model()
    model.config.compiler_decode_mode = "tree"
    model.config.grammar_ltr_primary = True
    model.config.grammar_draft_window = 5
    monkeypatch.setattr(model, "_predict_target_lengths", lambda *_: [12])
    observed: list[int] = []

    def decode(_ctx, _ctx_pad, length: int) -> torch.Tensor:
        observed.append(length)
        return torch.full((1, length), model.tokenizer.eos_id, dtype=torch.long)

    monkeypatch.setattr(model, "_greedy_ltr_decode_batch", decode)
    monkeypatch.setattr(model, "_decode_ids", lambda _ids: "root = Stack([])")
    monkeypatch.setattr(model, "_ensure_valid_openui", lambda text, *_a, **_k: text)
    assert model.generate_batch_requests([GenerationRequest(prompt="card")]) == [
        "root = Stack([])"
    ]
    assert observed == [17]


def test_lattice_search_rejects_unknown_mode() -> None:
    model = _model()
    model.config.compiler_search_mode = "unknown"
    ctx, ctx_pad = model._encode_context(["card"])
    with pytest.raises(ValueError, match="must be greedy, lattice, ptrm, or gram"):
        model._compiler_ltr_decode_one(
            ctx, ctx_pad, 24, mode="tree", slot_contract=None
        )


def test_lattice_search_matches_greedy_without_conflict() -> None:
    greedy = _model()
    ctx, ctx_pad = greedy._encode_context(["card"])
    expected = greedy._compiler_ltr_decode_one(
        ctx, ctx_pad, 24, mode="tree", slot_contract=None
    )
    model = _model()
    model.config.compiler_search_mode = "lattice"
    ctx, ctx_pad = model._encode_context(["card"])
    with collect_decode_stats() as stats:
        ids = model._compiler_ltr_decode_one(
            ctx, ctx_pad, 24, mode="tree", slot_contract=None
        )
    assert torch.equal(ids, expected)
    assert stats.compiler_lattice_states > 0
    assert stats.compiler_lattice_candidates >= stats.compiler_lattice_states


def test_ptrm_trajectory_policy_is_seed_reproducible() -> None:
    model = _model()
    model.config.compiler_search_mode = "ptrm"
    model.config.compiler_search_trigger = "always"
    model.config.compiler_search_width = 4
    model.config.compiler_search_noise = 1.0
    ctx, ctx_pad = model._encode_context(["card"])
    with collect_decode_stats() as left_stats:
        left = model._compiler_ltr_decode_one(
            ctx, ctx_pad, 24, mode="tree", slot_contract=None
        )
    with collect_decode_stats() as right_stats:
        right = model._compiler_ltr_decode_one(
            ctx, ctx_pad, 24, mode="tree", slot_contract=None
        )
    assert torch.equal(left, right)
    assert left_stats.compiler_lattice_trajectory_triggers > 0
    assert left_stats.compiler_lattice_trajectories == (
        right_stats.compiler_lattice_trajectories
    )
