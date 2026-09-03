"""`forced_token_fraction` must count every I2 bypass, not a subset of them.

The ratio is documented as "the share of the canvas that deterministic forcing
(I2 singleton bypass and forced spans) wrote without consulting the model".
Two of the sites that commit a singleton bypass incremented
`forced_row_tokens_without_forward` and called `_record_exact_bypass` without
touching `forced_tokens`, so on the lanes that reach them the ratio read
**0.0** -- "the model chose every token" -- while forcing wrote a good share of
the canvas. Measured in the S12 commit-authority profile: `forced_tokens`
stayed 0 in every run while `semantic_singleton_bypasses` was 18-27.

Measured here, before and after, on the fixture below:

| lane          | before | after |
|---------------|--------|-------|
| compiler off  | 0      | 4     |
| grammar LTR   | 0      | 4     |
| compiler tree | 6      | 6     |

The compiler-tree lane is unchanged on purpose: `commit` already counts a
forced span there, and it counts what it actually emitted (post-truncation,
post-eos substitution). An earlier draft of this fix added a second increment
on that lane and double-counted every span -- which a single-fixture ceiling
test did not catch, because that one fixture reached only that one lane. Hence
`test_no_lane_counts_a_committed_token_as_forced_twice` is parameterized: a
ratio is pinned per lane, and by its value, never by its sign.
"""

from __future__ import annotations

import pytest

from slm_training.dsl.schema import ExampleRecord
from slm_training.models.decode_stats import DecodeStats
from slm_training.models.twotower import TwoTowerConfig, TwoTowerModel

#: Lanes whose forced-token bookkeeping differs. ``compiler_off`` and
#: ``grammar_ltr`` reach the sites this file exists for; ``compiler_tree``
#: reaches ``commit`` instead and must stay exactly as it was.
_LANES: dict[str, dict[str, object]] = {
    "compiler_tree": {"compiler_decode_mode": "tree"},
    "compiler_off": {"compiler_decode_mode": "off"},
    "grammar_ltr": {"compiler_decode_mode": "off", "grammar_ltr_primary": True},
}

_PROMPTS = [
    ("A separator", "root = Separator()"),
    ("A card with a title", 'root = Stack([c1])\nc1 = TextContent(":slot_0")'),
    ("A button", 'root = Stack([b1])\nb1 = Button(":slot_0")'),
]


def _model(**overrides: object) -> TwoTowerModel:
    """A minimal constrained model; small enough to stay far inside the run cap."""
    records = [
        ExampleRecord(
            id=str(index),
            prompt=prompt,
            openui=openui,
            placeholders=[":slot_0"] if ":slot_0" in openui else [],
            split="train",
            meta={},
            accepted_outputs=[],
        )
        for index, (prompt, openui) in enumerate(_PROMPTS, start=1)
    ]
    settings: dict[str, object] = {
        "context_backend": "scratch",
        "output_tokenizer": "lexer",
        "d_model": 32,
        "n_heads": 2,
        "context_layers": 1,
        "denoiser_layers": 1,
        "max_prompt_len": 32,
        "max_target_len": 48,
        "grammar_ltr_max_tokens": 48,
        "gen_steps": 2,
        "seed": 0,
    }
    settings.update(overrides)
    built = TwoTowerModel.from_records(
        records, config=TwoTowerConfig(**settings), device="cpu"
    )
    built.eval()
    return built


# ---------------------------------------------------------------------------
# The ratio itself
# ---------------------------------------------------------------------------


def test_the_fraction_is_undefined_rather_than_zero_when_nothing_committed() -> None:
    """An undefined ratio is reported as undefined, never as a fabricated 0.0."""
    assert DecodeStats().forced_token_fraction is None


def test_the_fraction_is_forced_over_committed() -> None:
    stats = DecodeStats(forced_tokens=3, tokens_emitted=12)
    assert stats.forced_token_fraction == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# The regression, on the lanes that show it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("lane", ["compiler_off", "grammar_ltr"])
def test_a_lane_that_bypasses_singletons_attributes_them_to_forcing(
    lane: str,
) -> None:
    """These lanes commit four tokens without ever consulting the model.

    Before the fix ``forced_tokens`` read exactly 0 here while the bypass site
    fired four times, so the ratio said the model chose every token on a canvas
    it had not been asked about. The assertion is on the value: ``> 0`` would
    also pass on a lane where only some of the sites counted.
    """
    _text, stats = _model(**_LANES[lane]).generate_with_stats("A separator")

    assert stats.forced_tokens == 4, (
        f"{lane}: expected the four bypassed tokens to be attributed to "
        f"forcing, got {stats.forced_tokens}"
    )
    assert stats.forced_token_fraction == pytest.approx(4 / stats.tokens_emitted)


def test_the_compiler_tree_lane_is_untouched() -> None:
    """``commit`` already owns forcing there; a second increment double-counts.

    Pinned because an earlier draft added one, and every forced span on this
    lane was then counted twice, with pre-truncation lengths.
    """
    _text, stats = _model(**_LANES["compiler_tree"]).generate_with_stats("A separator")

    assert stats.forced_tokens == 6
    assert stats.tokens_emitted == 9


@pytest.mark.parametrize("lane", sorted(_LANES))
def test_no_lane_counts_a_committed_token_as_forced_twice(lane: str) -> None:
    """The ceiling, per lane -- one fixture on one lane is not evidence."""
    built = _model(**_LANES[lane])

    for prompt, _openui in _PROMPTS:
        _text, stats = built.generate_with_stats(prompt)
        assert stats.forced_tokens <= stats.tokens_emitted, (
            f"{lane}/{prompt}: {stats.forced_tokens} forced > "
            f"{stats.tokens_emitted} emitted -- a token is counted twice"
        )
        if stats.forced_token_fraction is not None:
            assert 0.0 <= stats.forced_token_fraction <= 1.0
