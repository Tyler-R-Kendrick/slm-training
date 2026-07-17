"""Online diffusion adapter coverage for every required corruption policy."""

from __future__ import annotations

import random

import pytest

torch = pytest.importorskip("torch")

from slm_training.data.diffusion import (
    POLICIES,
    DiffusionConfig,
    align_token_edits,
    corrupt_batch,
    corrupt_tokens,
)
from slm_training.dsl.parser import validate
from slm_training.dsl.schema import ExampleRecord
from slm_training.models.dsl_tokenizer import DSLNativeTokenizer
from slm_training.models.tokenizer import OpenUITokenizer
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

PROGRAM = (
    'root = Stack([inline_card, panel, cta], "column")\n'
    "inline_card = Card([title])\n"
    'title = TextContent(":hero.title")\n'
    "panel = Card([body])\n"
    'body = TextContent(":hero.body")\n'
    'cta = Button(":cta.label")'
)
SHORT_PROGRAM = 'root = Stack([title])\ntitle = TextContent(":hero.title")'
LONG_PROGRAM = (
    "root = Stack([title, cta])\n"
    'title = TextContent(":hero.title")\n'
    'cta = Button(":cta.label")'
)


@pytest.fixture(params=["compositional", "lexer"])
def tokenizer(request: pytest.FixtureRequest):
    if request.param == "lexer":
        return DSLNativeTokenizer.build()
    return OpenUITokenizer.build([PROGRAM, SHORT_PROGRAM, LONG_PROGRAM])


@pytest.mark.parametrize("policy", POLICIES)
def test_every_policy_reconstructs_and_aligns(policy: str, tokenizer) -> None:
    validate(PROGRAM)
    ids = tokenizer.encode(PROGRAM)
    result = corrupt_tokens(
        ids,
        tokenizer,
        policy=policy,
        rng=random.Random(11),
        mask_rate=0.35,
        overallocate=4,
        max_length=len(ids) + 4,
    )

    assert result.reconstruct() == tuple(ids)
    assert result.policy == policy
    assert result.canvas_length == len(result.target_ids) == len(result.noisy_ids)
    assert any(result.predict_mask)
    assert all(
        len(labels) == result.canvas_length for labels in result.aux_labels.values()
    )


def test_disjoint_policy_has_separate_mask_islands() -> None:
    tokenizer = DSLNativeTokenizer.build()
    result = corrupt_tokens(
        tokenizer.encode(PROGRAM),
        tokenizer,
        policy="disjoint",
        rng=random.Random(3),
        mask_rate=0.25,
    )
    positions = [index for index, predict in enumerate(result.predict_mask) if predict]
    assert any(right - left > 1 for left, right in zip(positions, positions[1:]))


def test_macro_substitution_policy_masks_whole_blocks() -> None:
    """C3 (SLM-27): the macro corruption policy masks every macro token (one
    token = one bound block) and falls back to uniform when none exist."""
    from slm_training.data.macro_induction import induce_macros
    from slm_training.dsl.canonicalize import canonicalize

    tokenizer = DSLNativeTokenizer.build()
    result = induce_macros([PROGRAM], tokenizer)
    if not result.expansions:
        pytest.skip("fixture program mined no macros")
    tokenizer.set_macro_expansions(result.expansions)
    ids = tokenizer.encode(canonicalize(PROGRAM))
    macro_positions = [
        index for index, tid in enumerate(ids) if tokenizer.is_macro_id(tid)
    ]
    assert macro_positions, "expected macro tokens in the encoded program"
    corruption = corrupt_tokens(
        ids,
        tokenizer,
        policy="macro_substitution",
        rng=random.Random(0),
    )
    for index in macro_positions:
        assert corruption.predict_mask[index]
    assert list(corruption.reconstruct(list(corruption.target_ids))) == list(ids)
    # Fallback: no macros present -> still a valid corruption.
    plain = DSLNativeTokenizer.build()
    plain_ids = plain.encode(PROGRAM)
    fallback = corrupt_tokens(
        plain_ids,
        plain,
        policy="macro_substitution",
        rng=random.Random(0),
    )
    assert any(fallback.predict_mask)


def test_reference_policy_masks_definition_and_uses() -> None:
    tokenizer = DSLNativeTokenizer.build()
    ids = tokenizer.encode(PROGRAM)
    result = corrupt_tokens(
        ids,
        tokenizer,
        policy="reference",
        rng=random.Random(0),
    )
    masked_binders = {
        ids[index]
        for index, predict in enumerate(result.predict_mask)
        if predict and tokenizer.is_bind_id(ids[index])
    }
    assert masked_binders
    for binder in masked_binders:
        occurrences = [index for index, value in enumerate(ids) if value == binder]
        assert len(occurrences) > 1
        assert all(result.predict_mask[index] for index in occurrences)


def test_real_expansion_and_contraction_align_to_one_canvas() -> None:
    tokenizer = DSLNativeTokenizer.build()
    short = tokenizer.encode(SHORT_PROGRAM)
    long = tokenizer.encode(LONG_PROGRAM)

    expansion = align_token_edits(short, long, tokenizer, max_length=128)
    contraction = align_token_edits(long, short, tokenizer, max_length=128)

    assert expansion.policy == "expansion"
    assert any(expansion.insertion_mask)
    assert expansion.reconstruct() == tuple(long)
    assert contraction.policy == "contraction"
    assert any(contraction.deletion_mask)
    assert contraction.reconstruct() == tuple(short)
    assert expansion.source_length < expansion.target_length
    assert contraction.source_length > contraction.target_length


def test_online_batch_resamples_without_changing_clean_rows() -> None:
    tokenizer = DSLNativeTokenizer.build()
    rows = [tokenizer.encode(PROGRAM), tokenizer.encode(SHORT_PROGRAM)]
    original = [list(row) for row in rows]
    config = DiffusionConfig(policies=("uniform", "statement", "all_mask"))
    rng = random.Random(7)

    first = corrupt_batch(rows, tokenizer, config=config, rng=rng)
    second = corrupt_batch(rows, tokenizer, config=config, rng=rng)

    assert rows == original
    assert first.rows != second.rows
    assert all(row.clean_ids == tuple(clean) for row, clean in zip(first.rows, rows))


def test_twotower_diffusion_loss_trains_length_head_and_preserves_cache() -> None:
    records = [
        ExampleRecord(
            id="a",
            prompt="Build a hero with a CTA.",
            openui=LONG_PROGRAM,
            placeholders=[":hero.title", ":cta.label"],
        )
    ]
    config = TwoTowerConfig(
        output_tokenizer="lexer",
        mask_pattern="diffusion",
        diffusion_policies=("contraction",),
        diffusion_length_buckets=(16, 32, 64, 96),
        diffusion_overallocate=2,
        diffusion_length_loss_weight=0.2,
        d_model=32,
        n_heads=4,
        context_layers=1,
        denoiser_layers=1,
        max_prompt_len=32,
        max_target_len=96,
        ltr_loss_weight=0.0,
        seed=5,
    )
    model = TwoTowerModel.from_records(records, config=config, device="cpu")

    loss = model.training_loss(records)
    cached = list(model._target_ids_cache["a"])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.length_head is not None
    assert model.length_head.weight.grad is not None

    model.zero_grad(set_to_none=True)
    model.training_loss(records)
    assert model._target_ids_cache["a"] == cached

    with torch.no_grad():
        model.length_head.weight.zero_()
        model.length_head.bias.fill_(-1.0)
        model.length_head.bias[1] = 1.0
    context = torch.zeros(1, 2, config.d_model)
    assert model._predict_target_lengths(context, None) == [32]
