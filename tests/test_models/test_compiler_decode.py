"""Compiler-drafted constrained decoding tests."""

from __future__ import annotations

import torch

from slm_training.dsl.grammar.fastpath.compiler_draft import (
    CompletionPath,
    build_completion_forest,
    gold_compiler_decisions,
    gold_compiler_decision_positions,
)
from slm_training.dsl.schema import ExampleRecord
from slm_training.models.blocks import DenoiserTower
from slm_training.models.decode_stats import collect_decode_stats
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer
from slm_training.models.grammar import make_grammar_state
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel


def _model() -> TwoTowerModel:
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
        'root=Card([title])\ntitle=TextContent(":hero.title")',
        add_special=True,
    )
    positions = gold_compiler_decision_positions(
        tokenizer, target, slot_contract=[":hero.title"]
    )
    selected = {tokenizer.id_to_token[target[position]] for position in positions}
    assert {"<BIND_0>", "Card", "<BIND_1>", "TextContent"} <= selected
    kinds = {decision.kind for decision in gold_compiler_decisions(tokenizer, target)}
    assert {"bind", "component_root", "component_bound", "struct"} <= kinds


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


def test_compiler_alignment_can_stratify_grammar_decision_kinds() -> None:
    model = _model()
    model.config.compiler_alignment_loss_weight = 1.0
    model.config.compiler_alignment_stratified = True
    record = ExampleRecord(
        id="alignment-stratified",
        prompt="card",
        openui='root = Card([title])\ntitle = TextContent(":hero.title")',
        placeholders=[":hero.title"],
        split="train",
        source="fixture",
    )
    loss = model.training_loss([record])
    assert torch.isfinite(loss)
    metrics = model.last_training_metrics
    assert metrics["compiler_alignment_rows"] > 1
    assert metrics["compiler_alignment_bind_rows"] == 1
    assert metrics["compiler_alignment_component_root_rows"] == 1
    assert metrics["compiler_alignment_component_bound_rows"] == 1


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
