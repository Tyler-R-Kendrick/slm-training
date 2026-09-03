"""The playground demo corpus must stay trainable under the current contracts.

`scripts/bootstrap_playground.py` is the documented way to regenerate the
committed demo checkpoint (`docs/MODEL_CARD.md`), and it stopped running: its
records spelled semantic markers (`:hero.title`, `:cta.label`) that the
opaque-marker contract forbids, and they declared no `placeholders` list at
all. Nothing noticed, because a bootstrap script has no test and the committed
checkpoint it produces was already frozen.

That combination is what makes this worth pinning. The checkpoint predates
`symbol_only/v2`, so `TwoTowerModel.from_checkpoint` refuses it and every tool
defaulting to it is blocked; with the regeneration path also broken there was
no way back. These tests keep the corpus honest against the same four
contracts the trainer applies, so the next contract tightening fails here in a
tenth of a second instead of at the next attempted regeneration.

They deliberately do not train anything: 200 steps is well inside
MAX_RUN_MINUTES but pointless to repeat per test run, and every failure mode
seen so far was in the records, not the loop.
"""

from __future__ import annotations

import pytest

from scripts.bootstrap_playground import DEMO_RECORDS, _assert_trainable
from slm_training.dsl.parser import validate
from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord


def _records() -> list[ExampleRecord]:
    """The corpus exactly as ``main`` builds it, minus the DESIGN.md context."""
    built = []
    for index, (prompt, openui) in enumerate(DEMO_RECORDS, start=1):
        serialized = validate(openui).serialized or openui
        built.append(
            ExampleRecord(
                id=str(index),
                prompt=prompt,
                openui=serialized,
                placeholders=list(extract_placeholders(serialized)),
                split="train",
                meta={},
                accepted_outputs=[],
            )
        )
    return built


def test_the_demo_corpus_is_not_empty() -> None:
    """Without this the contract check below would pass vacuously."""
    assert len(DEMO_RECORDS) >= 5


def test_every_demo_record_satisfies_every_trainer_contract() -> None:
    """The check the script now runs before spending any training time."""
    _assert_trainable(_records())


def test_a_semantic_marker_is_refused_with_the_record_named() -> None:
    """A regressed record must fail loudly, and say which one.

    The original failure surfaced as a bare ValueError from deep inside
    `from_records` after the model had been built.
    """
    bad = _records()
    bad[0] = ExampleRecord(
        id="regressed",
        prompt=bad[0].prompt,
        openui='root = Stack([c1])\nc1 = TextContent(":hero.title")',
        placeholders=[":hero.title"],
        split="train",
        meta={},
        accepted_outputs=[],
    )

    with pytest.raises(ValueError) as excinfo:
        _assert_trainable(bad)

    assert "regressed" in str(excinfo.value)


def test_every_demo_program_parses_and_round_trips() -> None:
    for _prompt, openui in DEMO_RECORDS:
        program = validate(openui)
        assert program.serialized
        assert validate(program.serialized).serialized == program.serialized


def test_markers_are_contiguous_from_slot_zero_within_each_record() -> None:
    """The slot contract is positional, so a gap is a broken contract."""
    for record in _records():
        markers = list(record.placeholders)
        assert markers == [f":slot_{index}" for index in range(len(markers))], record.id


def test_the_declared_placeholders_match_the_program() -> None:
    """A declared inventory that disagrees with the target trains a lie."""
    for record in _records():
        assert set(record.placeholders) == set(extract_placeholders(record.openui))
