"""Tests for the language-contract coverage corpus (P2 / SLM-6)."""

from __future__ import annotations

import pytest

from slm_training.data.language_contract import (
    LANGUAGE_CONTRACT_FAMILY,
    build_corpus,
    coverage_report,
    iter_negatives,
    iter_positives,
)
from slm_training.data.language_contract.corpus import NEGATIVE_GATES
from slm_training.data.contract import (
    GenerationRequest,
    assert_canonical_template_markers,
    canonicalize_example_template_markers,
)
from slm_training.data.quality import independent_judge
from slm_training.data.verify import Gate, GateStatus, evaluate_gate, verify_record
from slm_training.dsl.lang_core import bridge_available
from slm_training.dsl.parser import lexical_tokens, validate_output
from slm_training.dsl.schema import TASK_TOKENS, ExampleRecord
from slm_training.harnesses.model_build.eval_runner import _is_meaningful_program
from slm_training.harnesses.train_data.catalog import classify_source_family

# Positives are validated + serialized through the bridge, and the schema gate
# (G2) requires it, so skip the whole module when bridge deps are absent.
pytestmark = pytest.mark.skipif(
    not bridge_available(),
    reason="OpenUI bridge deps missing; run: cd src/apps/openui_bridge && npm ci",
)

_GATE_BY_ID = {gate.value: gate for gate in Gate}


def test_corpus_has_positives_and_negatives() -> None:
    positives = list(iter_positives())
    negatives = list(iter_negatives())
    assert len(positives) >= 60  # 9 productions + 54 components
    assert len(negatives) == len(list(_iter_negative_gates()))
    assert len(build_corpus()) == len(positives) + len(negatives)


def _iter_negative_gates():
    return [record.meta["expected_gate"] for record in iter_negatives()]


def test_every_positive_validates_meaningful_and_not_quarantined() -> None:
    for record in iter_positives():
        if record.target_kind != "document":
            validate_output(record.openui, record.target_kind, record.target_category)
            for target in record.accepted_outputs:
                validate_output(target.text, target.kind, target.category)
            continue
        meaningful, reason, _ = _is_meaningful_program(record.openui)
        assert meaningful, f"{record.id}: not meaningful ({reason})"
        report = verify_record(record)
        assert report.tier.value != "Quarantine", (
            f"{record.id}: quarantined at {report.failing_gate}"
        )


def test_boolean_prefers_one_symbol_but_accepts_full_documents() -> None:
    record = next(
        row for row in iter_positives() if row.id == "lc_boolean_literal"
    )
    assert record.openui == "true"
    assert record.target_kind == "lexical"
    assert lexical_tokens(record.openui) == ["true"]
    assert {target.kind for target in record.accepted_outputs} == {
        "lexical",
        "document",
    }
    assert any(target.text == "false" for target in record.accepted_outputs)


def test_all_contract_units_use_compact_gold_when_possible() -> None:
    records = list(iter_positives())
    assert {record.target_kind for record in records} == {
        "document",
        "statement",
        "expression",
        "lexical",
    }
    assert all(
        len(lexical_tokens(record.openui))
        <= min(
            [
                len(lexical_tokens(target.text))
                for target in record.accepted_outputs
            ]
            or [len(lexical_tokens(record.openui))]
        )
        for record in records
    )


def test_component_positives_match_generated_schema_roles() -> None:
    for record in iter_positives():
        schema_reasons = [
            reason
            for reason in independent_judge(record)["reasons"]
            if reason.startswith("schema_") or reason == "judge_schema_parse_failed"
        ]
        assert schema_reasons == [], f"{record.id}: {schema_reasons}"


def test_every_negative_fails_its_expected_gate() -> None:
    for record in iter_negatives():
        expected = record.meta["expected_gate"]
        gate = _GATE_BY_ID[expected]
        # The targeted gate must reject the program in isolation.
        assert evaluate_gate(gate, record).status is GateStatus.FAIL, (
            f"{record.id}: gate {expected} did not fail"
        )
        # For lexical/grammar/schema/reference corruptions the targeted gate is
        # also the *first* failure. v0.5 dataflow syntax (G4) is invalid at
        # several levels, so only the isolated dataflow check is asserted there.
        if expected != Gate.DATAFLOW.value:
            assert verify_record(record).failing_gate is gate, (
                f"{record.id}: first failing gate != {expected}"
            )


def test_all_components_are_covered() -> None:
    components = coverage_report()["components"]
    assert components["uncovered"] == []
    assert components["covered"] == components["total"] == 54


def test_all_required_prop_positions_covered() -> None:
    positions = coverage_report()["required_prop_positions"]
    assert positions["covered"] == positions["total"] > 0


def test_every_gate_has_a_negative() -> None:
    covered = set(coverage_report()["gates"]["covered"])
    assert covered == set(NEGATIVE_GATES)


def test_records_carry_contract_id_and_family() -> None:
    for record in build_corpus():
        assert record.source == "language_contract"
        assert record.meta.get("contract_id")
        assert classify_source_family(record) == LANGUAGE_CONTRACT_FAMILY


def test_task_tokens_are_valid() -> None:
    for record in iter_positives():
        assert record.meta["task"] == "generation"
    for record in iter_negatives():
        assert record.meta["task"] == "adversarial"
    assert {"generation", "adversarial"} <= TASK_TOKENS


def test_build_corpus_is_deterministic() -> None:
    first = [(r.id, r.openui) for r in build_corpus()]
    second = [(r.id, r.openui) for r in build_corpus()]
    assert first == second
    assert len({rid for rid, _ in first}) == len(first)  # unique ids


def test_output_targets_round_trip_and_reach_generation_request() -> None:
    record = next(
        row for row in iter_positives() if row.id == "lc_boolean_literal"
    )
    restored = ExampleRecord.from_dict(record.to_dict())
    assert restored.output_targets == record.output_targets
    request = GenerationRequest.from_record(restored)
    assert request.output_kind == "lexical"
    assert request.output_category == "boolean"
    assert GenerationRequest.from_dict(request.to_dict()) == request


def test_marker_canonicalization_updates_structural_metadata_atomically() -> None:
    record = ExampleRecord(
        id="semantic-marker",
        prompt="Use :hero.title and :cta.label.",
        openui=(
            'title = TextContent(":hero.title")\n'
            'cta = Button(":cta.label")\n'
            'root = Stack([title, cta], "column")'
        ),
        placeholders=[":hero.title", ":cta.label"],
        split="train",
        meta={
            "semantic_contract": {
                "version": 1,
                "placeholders": [":cta.label", ":hero.title"],
            },
            "nested": {":hero.title": ":cta.label"},
        },
    )

    canonical = canonicalize_example_template_markers(record)

    assert canonical.placeholders == [":slot_0", ":slot_1"]
    assert canonical.meta["semantic_contract"]["placeholders"] == [
        ":slot_0",
        ":slot_1",
    ]
    assert canonical.meta["nested"] == {":slot_0": ":slot_1"}
    assert_canonical_template_markers(canonical)


def test_metadata_only_markers_do_not_expand_completion_inventory() -> None:
    record = ExampleRecord(
        id="metadata-marker",
        prompt="Build the captured layout.",
        openui='root = TextContent(":title.copy")',
        placeholders=[":title.copy", ":offscreen.label"],
        meta={"normalized_ui_graph": [{"text": ":offscreen.label"}]},
    )

    canonical = canonicalize_example_template_markers(record)

    assert canonical.openui == 'root = TextContent(":slot_0")'
    assert canonical.placeholders == [":slot_0"]
    assert canonical.meta["normalized_ui_graph"][0]["text"] == ":slot_1"
    assert_canonical_template_markers(canonical)
