"""Unit tests for quality experiment helpers."""

from __future__ import annotations

from slm_training.dsl.schema import ExampleRecord
from slm_training.models.twotower import format_context_text
from slm_training.preference import build_pairs_from_candidates, grammar_score
from slm_training.quality import (
    compact_schema_snippet,
    curriculum_schedule,
    soft_corrupt_openui,
    tag_curriculum_stage,
)
from slm_training.retrieval import build_skeleton_bank, nearest_skeletons


def test_soft_corrupt_keeps_placeholders() -> None:
    gold = (
        'root = Stack([hero, cta], "column", "m")\n'
        'hero = TextContent(":hero.title", "primary")\n'
        'cta = Button(":cta.label", "primary")\n'
    )
    bad = soft_corrupt_openui(gold)
    assert bad != gold
    assert ":hero.title" in bad or ":wrong." in bad or "TextContent(" in bad


def test_schema_and_retrieval_context_format() -> None:
    text = format_context_text(
        "Build a hero",
        "# Design\nbody",
        schema=compact_schema_snippet(),
        retrieved_skeleton='root = Stack([a], "column", "m")\na = TextContent(":a.t", "primary")',
    )
    assert "---SCHEMA---" in text
    assert "---RETRIEVED_SKELETON---" in text
    assert "---DESIGN.md---" in text


def test_curriculum_tags_and_schedule() -> None:
    rico = ExampleRecord(
        id="rico_train_1",
        prompt="screen",
        openui='root = Stack([a], "column", "m")\na = TextContent(":a.t", "primary")\n',
        source="rico",
    )
    assert tag_curriculum_stage(rico) == "A"
    assert curriculum_schedule(0, 100) == "A"
    assert curriculum_schedule(50, 100) == "B"
    assert curriculum_schedule(90, 100) == "C"


def test_retrieval_nearest() -> None:
    records = [
        ExampleRecord(
            id="1",
            prompt="hero card with title",
            openui='root = Stack([h], "column", "m")\nh = TextContent(":h.t", "primary")\n',
        ),
        ExampleRecord(
            id="2",
            prompt="settings form inputs",
            openui='root = Stack([f], "column", "m")\nf = Input(":f.v", "text")\n',
        ),
    ]
    bank = build_skeleton_bank(records)
    hits = nearest_skeletons(bank, "hero title card", k=1)
    assert hits and hits[0].record_id == "1"


def test_prefer_valid_rejects() -> None:
    gold = ExampleRecord(
        id="g",
        prompt="p",
        openui=(
            'root = Stack([a, b], "column", "m")\n'
            'a = TextContent(":a.t", "primary")\n'
            'b = Button(":b.l", "primary")\n'
        ),
    )
    worse = soft_corrupt_openui(gold.openui)
    broken = "root = Broken()"
    pair = build_pairs_from_candidates(
        gold.prompt,
        [gold.openui, worse, broken],
        gold=gold,
        prefer_valid_rejects=True,
    )
    assert pair is not None
    assert grammar_score(pair.rejected) > 0.0 or pair.rejected == worse
