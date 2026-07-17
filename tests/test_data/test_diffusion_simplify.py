"""D1 simplification-consistent forward corruption tests."""

from __future__ import annotations

from slm_training.data.diffusion.simplify import (
    simplify_record,
    simplify_records,
    simplify_target,
)
from slm_training.dsl import validate
from slm_training.dsl.schema import ExampleRecord

HERO = 'root = Stack([hero], "column")\nhero = Card([t])\nt = TextContent(":x")'
HERO_RENAMED = (
    'root = Stack([box], "column")\nbox = Card([label])\nlabel = TextContent(":x")'
)


def test_simplify_target_is_canonical_and_valid() -> None:
    out = simplify_target(HERO)
    validate(out)
    assert simplify_target(out) == out  # idempotent


def test_alpha_variants_collapse_to_one_target() -> None:
    # The point of forward simplification: one canonical target per class.
    assert simplify_target(HERO) == simplify_target(HERO_RENAMED)


def test_simplify_target_passes_through_unparseable() -> None:
    assert simplify_target("not valid (((") == "not valid ((("


def test_simplify_record_preserves_placeholders() -> None:
    rec = ExampleRecord(id="a", prompt="Hero", openui=HERO_RENAMED, placeholders=[":x"])
    out = simplify_record(rec)
    assert out.placeholders == [":x"]
    assert out.openui == simplify_target(HERO_RENAMED)
    assert out.id == "a" and out.prompt == "Hero"


def test_simplify_records_reports_collapse_stats() -> None:
    records = [
        ExampleRecord(id="a", prompt="p", openui=HERO, placeholders=[":x"]),
        ExampleRecord(id="b", prompt="p", openui=HERO_RENAMED, placeholders=[":x"]),
    ]
    out, stats = simplify_records(records)
    assert stats["n"] == 2
    # Both are the same layout → they collapse to one distinct canonical target.
    assert stats["distinct_canonical_targets"] == 1
    assert stats["changed"] >= 1
    assert out[0].openui == out[1].openui
