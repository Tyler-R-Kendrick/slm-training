"""`forced_token_fraction` must count every I2 bypass, not a subset of them.

The ratio is documented as "the share of the canvas that deterministic forcing
(I2 singleton bypass and forced spans) wrote without consulting the model".
Four of the six sites that commit a singleton bypass incremented
`forced_row_tokens_without_forward` and called `_record_exact_bypass` without
touching `forced_tokens`, so the ratio read **0.0** -- "the model chose every
token" -- on decodes where forcing wrote a good share of the canvas. Measured
in the S12 commit-authority profile: `forced_tokens` stayed 0 in every run
while `semantic_singleton_bypasses` was 18-27.

A telemetry channel that reads zero is worse than one that is absent: it
answers the question wrongly instead of declining to answer, and
`forced_token_fraction` is one of the endpoints a decode-authority campaign
reports.

The load-bearing test is the agreement one: whenever a decode records singleton
bypasses, `forced_tokens` must have counted them. It decodes a real model
rather than asserting on source text, so a future refactor that moves a bypass
site is caught by behaviour.
"""

from __future__ import annotations

import pytest

from slm_training.dsl.schema import ExampleRecord
from slm_training.models.decode_stats import DecodeStats
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel


def test_the_fraction_is_undefined_rather_than_zero_when_nothing_committed() -> None:
    """An undefined ratio is reported as undefined, never as a fabricated 0.0."""
    assert DecodeStats().forced_token_fraction is None


def test_the_fraction_is_forced_over_committed() -> None:
    stats = DecodeStats(forced_tokens=3, tokens_emitted=12)
    assert stats.forced_token_fraction == pytest.approx(0.25)


@pytest.fixture(scope="module")
def model() -> TwoTowerModel:
    """A minimal constrained model; small enough to stay far inside the run cap."""
    record = ExampleRecord(
        id="1",
        prompt="A separator",
        openui="root = Separator()",
        placeholders=[],
        split="train",
        meta={},
        accepted_outputs=[],
    )
    config = TwoTowerConfig(
        context_backend="scratch",
        output_tokenizer="lexer",
        compiler_decode_mode="tree",
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
    built = TwoTowerModel.from_records([record], config=config, device="cpu")
    built.eval()
    return built


def _decode_stats(model: TwoTowerModel) -> DecodeStats:
    _text, stats = model.generate_with_stats("A separator")
    return stats


def test_a_fully_determined_canvas_reports_a_forced_share_of_one(
    model: TwoTowerModel,
) -> None:
    """The regression, stated as the agreement it broke.

    `root = Separator()` is determined by the grammar from its prefix onward,
    so every committed token here is one the model did not choose and the
    share of the canvas written by forcing is exactly 1.0.

    Before the fix this decode reported 5 of 8 tokens forced (0.625): three
    commits went through bypass sites that incremented
    `forced_row_tokens_without_forward` and recorded the bypass but never
    counted the token. The ratio attributed three tokens to a model that never
    saw them. Measured both ways on this fixture.

    An assertion on the exact value rather than "> 0" is deliberate -- "> 0"
    passes on the broken code, which is how a telemetry defect survives a
    test.
    """
    stats = _decode_stats(model)

    # Guards, so a fixture that stopped forcing cannot pass this vacuously.
    assert stats.semantic_singleton_bypasses >= 1
    assert stats.committed_tokens > 0

    assert stats.forced_tokens == stats.committed_tokens, (
        f"{stats.committed_tokens - stats.forced_tokens} committed token(s) "
        "on a grammar-determined canvas are attributed to the model: a bypass "
        "site is not counting its token"
    )
    assert stats.forced_token_fraction == pytest.approx(1.0)


def test_forced_tokens_never_exceeds_what_was_committed(
    model: TwoTowerModel,
) -> None:
    """Double-counting one commit would push the fraction above 1."""
    stats = _decode_stats(model)

    assert stats.forced_tokens <= stats.committed_tokens
    if stats.forced_token_fraction is not None:
        assert 0.0 <= stats.forced_token_fraction <= 1.0


def test_forced_tokens_tracks_the_without_forward_counter(
    model: TwoTowerModel,
) -> None:
    """The two counters name the same events at every bypass site.

    They are separate fields because one is a token count and the other feeds
    the forwards-saved accounting, but a site that increments one and not the
    other is the defect this file exists for.
    """
    stats = _decode_stats(model)

    assert stats.forced_tokens >= stats.forced_row_tokens_without_forward
