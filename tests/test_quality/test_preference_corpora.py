"""Gold-correction vs self-distilled preference corpus separation."""

from __future__ import annotations

import pytest

from slm_training.dsl import bridge_available
from slm_training.dsl.schema import ExampleRecord
from slm_training.preference import collect_pairs_with_generator

pytestmark = pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd tools/openui_bridge && npm ci",
)

GOLD = 'root = Stack([cta])\ncta = Button(":cta.label")'
GOOD_SAMPLE = 'root = Stack([cta])\ncta = TextContent(":cta.label")'
BAD_SAMPLE = 'root = Stack([cta])\ncta = TextContent(":wrong.label")'


def _record() -> ExampleRecord:
    return ExampleRecord(
        id="r1",
        prompt="Button only",
        openui=GOLD,
        placeholders=[":cta.label"],
        split="train",
    )


def test_include_gold_default_injects_and_tags() -> None:
    pairs = collect_pairs_with_generator(
        [_record()],
        lambda record: [BAD_SAMPLE],
        include_gold=True,
    )
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.chosen == GOLD
    assert pair.meta["pair_corpus"] == "gold_correction"
    assert pair.meta["gold_injected"] is True
    assert pair.meta["gold_used"] is True
    assert pair.meta["record_id"] == "r1"


def test_no_gold_never_emits_gold() -> None:
    pairs = collect_pairs_with_generator(
        [_record()],
        # Generator copied gold exactly — must be dropped in policy-only mode.
        lambda record: [record.openui, GOOD_SAMPLE, BAD_SAMPLE],
        include_gold=False,
        generator_checkpoint="ckpt-sha",
    )
    assert len(pairs) == 1
    pair = pairs[0]
    assert GOLD not in {pair.chosen, pair.rejected}
    assert pair.meta["pair_corpus"] == "self_distilled"
    assert pair.meta["gold_injected"] is False
    assert pair.meta["gold_used"] is False
    assert pair.meta["generator_checkpoint"] == "ckpt-sha"


def test_no_gold_with_indistinguishable_candidates_yields_no_pair() -> None:
    pairs = collect_pairs_with_generator(
        [_record()],
        lambda record: [record.openui, GOOD_SAMPLE],
        include_gold=False,
    )
    # Only one policy candidate remains — no strict ranking possible.
    assert pairs == []
