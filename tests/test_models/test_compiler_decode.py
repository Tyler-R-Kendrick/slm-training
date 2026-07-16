"""Compiler-drafted constrained decoding tests."""

from __future__ import annotations

import pytest
import torch

from slm_training.dsl.grammar.fastpath.compiler_draft import (
    CompletionPath,
    build_completion_forest,
    gold_compiler_decisions,
    gold_compiler_decision_positions,
)
from slm_training.dsl.schema import ExampleRecord
from slm_training.data.contract import GenerationRequest
from slm_training.models.blocks import DenoiserTower
from slm_training.models.decode_stats import collect_decode_stats
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer
from slm_training.models.grammar import make_grammar_state
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel


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
