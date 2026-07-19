"""Regression tests for synthetic-corpus diversity fingerprinting."""

from __future__ import annotations

from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.train_data.diversity import (
    fingerprint_record,
    summarize_fingerprints,
)


def _record(prompt: str, openui: str, **kwargs: object) -> ExampleRecord:
    return ExampleRecord(
        id="test-1",
        prompt=prompt,
        openui=openui,
        split="train",
        source="fixture",
        **kwargs,
    )


def test_fingerprint_record_returns_all_fields() -> None:
    source = (
        'root = Card([header, body])\n'
        'header = CardHeader(":card.title")\n'
        'body = TextContent(":card.body")\n'
    )
    record = _record("Build a card.", source)
    fp = fingerprint_record(record)
    assert fp.schema_version == "diversity_fingerprints/v1"
    assert fp.record_id == record.id
    assert fp.canonical_root_id
    assert fp.binding_aware_sketch
    assert fp.topology_sketch
    assert fp.type_action_multiset
    assert fp.prompt_intent_fingerprint
    assert fp.source_lineage_id
    assert fp.exact_structure_fingerprint


def test_alpha_equivalent_programs_share_canonical_root_id() -> None:
    a = _record(
        "Build a card.",
        'root = Card([h])\nh = CardHeader(":card.title")\n',
    )
    b = _record(
        "Build a card.",
        'root = Card([title])\ntitle = CardHeader(":card.title")\n',
    )
    assert fingerprint_record(a).canonical_root_id == fingerprint_record(b).canonical_root_id


def test_different_programs_have_different_canonical_root_ids() -> None:
    a = _record("Build a card.", 'root = Card([CardHeader(":card.title")])\n')
    b = _record("Build a button.", 'root = Button(":btn.action")\n')
    assert fingerprint_record(a).canonical_root_id != fingerprint_record(b).canonical_root_id


def test_same_prompt_intent_with_different_slots_matches() -> None:
    a = _record("Build a card with :hero.title and :hero.body.", 'root = Card([])\n')
    b = _record("Build a card with :item.title and :item.body.", 'root = Card([])\n')
    assert (
        fingerprint_record(a).prompt_intent_fingerprint
        == fingerprint_record(b).prompt_intent_fingerprint
    )


def test_different_prompt_intents_differ() -> None:
    a = _record("Build a card.", 'root = Card([])\n')
    b = _record("Build a button.", 'root = Card([])\n')
    assert (
        fingerprint_record(a).prompt_intent_fingerprint
        != fingerprint_record(b).prompt_intent_fingerprint
    )


def test_lineage_id_changes_with_parent() -> None:
    a = _record("Build a card.", 'root = Card([])\n', meta={"root_parent_id": "p1"})
    b = _record("Build a card.", 'root = Card([])\n', meta={"root_parent_id": "p2"})
    assert fingerprint_record(a).source_lineage_id != fingerprint_record(b).source_lineage_id


def test_summarize_counts_unique_values() -> None:
    records = [
        _record("Build a card.", 'root = Card([CardHeader(":card.title")])\n'),
        _record("Build a card.", 'root = Card([TextContent(":card.body")])\n'),
        _record("Build a card.", 'root = Card([CardHeader(":card.title")])\n'),
    ]
    fps = [fingerprint_record(r) for r in records]
    summary = summarize_fingerprints(fps)
    assert summary["n_records"] == 3
    assert summary["unique_counts"]["canonical_root_id"] == 2
    assert summary["unique_counts"]["topology_sketch"] >= 1


def test_to_dict_round_trip() -> None:
    source = 'root = Card([CardHeader(":card.title")])\n'
    record = _record("Build a card.", source)
    fp = fingerprint_record(record)
    d = fp.to_dict()
    assert d["record_id"] == record.id
    assert d["schema_version"] == "diversity_fingerprints/v1"
