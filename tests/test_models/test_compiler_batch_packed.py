"""Greedy compiler rows share forwards without weakening exact bypasses."""

from __future__ import annotations

import pytest
import torch

from slm_training.data.contract import canonicalize_example_template_markers
from slm_training.dsl.grammar.fastpath.compiler_draft import (
    CompletionForest,
    CompletionPath,
)
from slm_training.dsl.schema import ExampleRecord
from slm_training.models.decode_stats import collect_decode_stats
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel


@pytest.fixture
def model() -> TwoTowerModel:
    record = ExampleRecord(
        id="compiler-batch",
        prompt="card",
        openui='root = TextContent(":slot_0")',
        placeholders=[":slot_0"],
        split="train",
        source="fixture",
    )
    config = TwoTowerConfig(
        context_backend="scratch",
        output_tokenizer="lexer",
        compiler_decode_mode="restricted",
        d_model=32,
        n_heads=2,
        context_layers=1,
        denoiser_layers=1,
        max_prompt_len=32,
        max_target_len=16,
        grammar_ltr_max_tokens=16,
        grammar_draft_window=4,
        gen_steps=1,
        seed=0,
        slot_contract_constrained_decode=True,
    )
    result = TwoTowerModel.from_records(
        [canonicalize_example_template_markers(record)],
        config=config,
        device="cpu",
    )
    result.eval()
    return result


def _forests(model: TwoTowerModel) -> tuple[CompletionForest, CompletionForest]:
    eos = int(model.tokenizer.eos_id)
    component = min(int(token_id) for token_id in model.tokenizer.kind_ids("component"))
    forced = CompletionForest((CompletionPath((eos,), "eos"),), "complete")
    ambiguous = CompletionForest(
        (
            CompletionPath((eos,), "eos"),
            CompletionPath((component,), "component"),
        ),
        "complete",
    )
    return forced, ambiguous


def test_greedy_compiler_compacts_only_ambiguous_rows(
    model: TwoTowerModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slm_training.dsl.grammar.fastpath import compiler_draft

    forced, ambiguous = _forests(model)
    monkeypatch.setattr(
        compiler_draft,
        "build_completion_forest",
        lambda *_args, slot_contract=None, **_kwargs: (
            ambiguous if slot_contract else forced
        ),
    )
    model._slot_contracts = [None, [":slot_0"]]
    batch_sizes: list[int] = []
    original = model._denoiser_hidden

    def hidden(ids, *args, **kwargs):
        batch_sizes.append(int(ids.size(0)))
        return original(ids, *args, **kwargs)

    monkeypatch.setattr(model, "_denoiser_hidden", hidden)
    ctx, ctx_pad = model._encode_context(["forced", "ambiguous"])
    with collect_decode_stats() as stats:
        result = model._compiler_ltr_decode_batch(ctx, ctx_pad, 5, mode="restricted")

    assert batch_sizes == [1]
    assert result[:, 1].tolist() == [model.tokenizer.eos_id] * 2
    assert stats.ambiguous_rows_forwarded == 1
    assert stats.forwards_count == 1


def test_all_exact_compiler_rows_bypass_every_forward(
    model: TwoTowerModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slm_training.dsl.grammar.fastpath import compiler_draft

    forced, _ = _forests(model)
    monkeypatch.setattr(
        compiler_draft, "build_completion_forest", lambda *_args, **_kwargs: forced
    )
    monkeypatch.setattr(
        model,
        "_denoiser_hidden",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("exact complete singleton must not run the model")
        ),
    )
    model._slot_contracts = [None, None]
    ctx, ctx_pad = model._encode_context(["one", "two"])
    with collect_decode_stats() as stats:
        result = model._compiler_ltr_decode_batch(ctx, ctx_pad, 5, mode="restricted")

    assert result[:, 1].tolist() == [model.tokenizer.eos_id] * 2
    assert stats.forwards_count == 0


def test_batch_rows_share_cache_but_keep_distinct_decode_states(
    model: TwoTowerModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slm_training.dsl.grammar.fastpath import compiler_draft

    forced, _ = _forests(model)
    cache_ids: list[int] = []
    state_ids: list[int] = []

    def forest(*_args, state=None, **_kwargs):
        cache_ids.append(id(state.completion_batch_cache))
        state_ids.append(id(state))
        return forced

    monkeypatch.setattr(compiler_draft, "build_completion_forest", forest)
    model._slot_contracts = [None, None]
    ctx, ctx_pad = model._encode_context(["one", "two"])
    model._compiler_ltr_decode_batch(ctx, ctx_pad, 5, mode="restricted")

    assert len(set(cache_ids)) == 1
    assert len(set(state_ids)) == 2


@pytest.mark.parametrize("mode", ["restricted", "tree"])
def test_batched_compiler_matches_sequential_row_scoring(
    model: TwoTowerModel,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    from slm_training.dsl.grammar.fastpath import compiler_draft

    _, ambiguous = _forests(model)
    monkeypatch.setattr(
        compiler_draft,
        "build_completion_forest",
        lambda *_args, **_kwargs: ambiguous,
    )
    contracts = [[":slot_0"], [":slot_0"]]
    model._slot_contracts = contracts
    ctx, ctx_pad = model._encode_context(["one", "two"])
    expected = torch.stack(
        [
            model._compiler_ltr_decode_one(
                ctx[row : row + 1],
                ctx_pad[row : row + 1],
                5,
                mode=mode,
                slot_contract=contracts[row],
                _plan_row=row,
            )
            for row in range(2)
        ]
    )
    forwards: list[int] = []
    original = model._denoiser_hidden

    def hidden(ids, *args, **kwargs):
        forwards.append(int(ids.size(0)))
        return original(ids, *args, **kwargs)

    monkeypatch.setattr(model, "_denoiser_hidden", hidden)
    actual = model._compiler_ltr_decode_batch(ctx, ctx_pad, 5, mode=mode)

    assert torch.equal(actual, expected)
    assert forwards == [2]


def test_empty_hard_domains_fail_closed_without_forward(
    model: TwoTowerModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    from slm_training.dsl.grammar.fastpath import compiler_draft

    empty = CompletionForest((), "none")
    monkeypatch.setattr(
        compiler_draft, "build_completion_forest", lambda *_args, **_kwargs: empty
    )
    monkeypatch.setattr(
        model,
        "_denoiser_hidden",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("an empty legal domain must not run the model")
        ),
    )
    model._slot_contracts = [None, None]
    ctx, ctx_pad = model._encode_context(["one", "two"])
    with collect_decode_stats() as stats:
        result = model._compiler_ltr_decode_batch(ctx, ctx_pad, 5, mode="restricted")

    assert result[:, 1:].eq(model.tokenizer.pad_id).all()
    assert stats.constrained_dead_ends == 2
    assert stats.compiler_fallbacks == 2
    assert stats.forwards_count == 0


@pytest.mark.parametrize("fallback", ["search", "speculative"])
def test_non_greedy_authority_uses_sequential_decoder(
    model: TwoTowerModel,
    monkeypatch: pytest.MonkeyPatch,
    fallback: str,
) -> None:
    calls: list[int] = []

    def decode_one(ctx, _ctx_pad, length, **_kwargs):
        calls.append(int(ctx.size(0)))
        return torch.full(
            (length,),
            model.tokenizer.eos_id,
            dtype=torch.long,
            device=ctx.device,
        )

    monkeypatch.setattr(model, "_compiler_ltr_decode_one", decode_one)
    if fallback == "search":
        model.config.compiler_search_mode = "lattice"
    else:
        monkeypatch.setattr(model, "_speculative_ranker", lambda: object())
    ctx, ctx_pad = model._encode_context(["one", "two"])

    result = model._compiler_ltr_decode_batch(ctx, ctx_pad, 5, mode="restricted")

    assert result.shape == (2, 5)
    assert calls == [1, 1]
