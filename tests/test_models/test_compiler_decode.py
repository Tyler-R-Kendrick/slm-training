"""Compiler-drafted constrained decoding tests."""

from __future__ import annotations

import torch

from slm_training.dsl.grammar.fastpath.compiler_draft import (
    CompletionPath,
    build_completion_forest,
)
from slm_training.dsl.schema import ExampleRecord
from slm_training.models.blocks import DenoiserTower
from slm_training.models.decode_stats import collect_decode_stats
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer
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
