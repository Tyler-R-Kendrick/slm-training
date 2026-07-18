"""Tests for CAP1-05 template-abstraction sufficiency audit."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.dsl.analysis.arity.template_sufficiency import (
    _structural_choice_stream,
    audit_template_sufficiency,
    extract_value_classes,
    generate_variants,
)
from slm_training.dsl.schema import ExampleRecord, load_jsonl


def _record(record_id: str, openui: str) -> ExampleRecord:
    return ExampleRecord(
        id=record_id,
        prompt="fixture prompt",
        openui=openui,
        split="train",
        source="fixture",
    )


def _load_seed_records() -> list[ExampleRecord]:
    path = Path("src/slm_training/resources/test_seeds.jsonl")
    return list(load_jsonl(path))


def test_extract_value_classes_finds_string_literals() -> None:
    records = [
        _record(
            "r1",
            'root = Stack([note], "column")\n'
            'note = Callout("info", ":foo.title", ":foo.body")',
        ),
    ]
    inventory = extract_value_classes(records)
    assert any(vc.value_kind == "string" for vc in inventory.value_classes)
    # The Callout severity literal should appear as a string value class.
    assert any("string_value" in vc.class_id for vc in inventory.value_classes)


def test_extract_value_classes_finds_number_literals() -> None:
    records = [
        _record(
            "r1",
            'root = Stack([volume], "column")\n'
            'volume = Slider("volume", "continuous", 0, 100, 1, [40], ":foo.label")',
        ),
    ]
    inventory = extract_value_classes(records)
    assert any(vc.value_kind == "number" for vc in inventory.value_classes)


def test_audit_runs_on_seed_records() -> None:
    records = _load_seed_records()
    report = audit_template_sufficiency(records, max_per_record=5)
    assert report.metrics["records_audited"] == len(records)
    assert report.metrics["value_classes"] >= 1
    assert report.metrics["variants_generated"] >= 1


def test_audit_records_violations_consistently() -> None:
    records = _load_seed_records()
    report = audit_template_sufficiency(records, max_per_record=5)
    for v in report.variants:
        assert v.original_choice_stream or v.variant_choice_stream
        structural_diff = (
            _structural_choice_stream(v.variant_openui)
            != _structural_choice_stream(v.original_openui)
        )
        assert v.is_violation == structural_diff


def test_structural_choice_stream_collapses_literal_payloads() -> None:
    s1 = 'root = Stack([note], "column")\nnote = Callout("info", ":foo.title", ":foo.body")'
    s2 = s1.replace('"info"', '"warning"')
    assert _structural_choice_stream(s1) == _structural_choice_stream(s2)
    # Changing the layout direction should change the structural stream.
    s3 = s1.replace('"column"', '"row"')
    assert _structural_choice_stream(s1) != _structural_choice_stream(s3)


def test_refinement_proposed_after_violation() -> None:
    records = _load_seed_records()
    report = audit_template_sufficiency(records, max_per_record=10)
    if report.violations:
        assert report.refinements
        for r in report.refinements:
            assert r.retained_attributes
            assert r.value_class_id in {v.value_class_id for v in report.violations}


def test_report_does_not_persist_raw_user_text() -> None:
    records = [
        _record(
            "r1",
            'root = Stack([note], "column")\n'
            'note = Callout("xyzzy_secret_user_value", ":foo.title", ":foo.body")',
        ),
    ]
    report = audit_template_sufficiency(records, max_per_record=3)
    raw = json.dumps(report.to_dict())
    assert "xyzzy_secret_user_value" not in raw
    assert "foo.title" not in raw


def test_inventory_examples_are_fingerprinted_not_raw() -> None:
    records = [
        _record(
            "r1",
            'root = Stack([note], "column")\n'
            'note = Callout("info", ":foo.title", ":foo.body")',
        ),
    ]
    inventory = extract_value_classes(records)
    assert inventory.value_classes
    vc = inventory.value_classes[0]
    d = vc.to_dict()
    assert "examples" not in d
    assert "example_fingerprints" in d
    assert all(isinstance(fp, str) and len(fp) == 16 for fp in d["example_fingerprints"])


def test_variant_to_dict_hashes_literals() -> None:
    records = [
        _record(
            "r1",
            'root = Stack([note], "column")\n'
            'note = Callout("info", ":foo.title", ":foo.body")',
        ),
    ]
    variants = generate_variants(
        records, extract_value_classes(records), max_per_record=1
    )
    assert variants
    d = variants[0].to_dict()
    assert "original_literal_fingerprint" in d
    assert "variant_literal_fingerprint" in d
    assert "original_openui" not in d
    assert "variant_openui" not in d
