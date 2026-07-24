"""Tests for SLM-301 (LAR1-04) prompt-observability harness."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.dsl.placeholders import extract_placeholders
from slm_training.dsl.schema import ExampleRecord
from slm_training.harnesses.experiments.slm301_prompt_observability import (
    MIN_USEFUL_PAIRED_DELTA,
    OBSERVABLE_PROMPT,
    OBSERVABLE_REQUEST_ONLY,
    UNKNOWN,
    build_matched_requests,
    coverage_class,
    observability_report,
    paired_delta,
    paired_outcome_table,
)
from slm_training.models.template_fill import ensure_prompt_inventory

TEST_SEEDS = Path("src/slm_training/resources/test_seeds.jsonl")

_GOLD = (
    'root = Stack([title, body], "column")\n'
    'title = TextContent(":hero.title")\n'
    'body = TextContent(":hero.body")'
)

_GOLD_NO_SLOTS = 'root = Stack([rule], "column")\nrule = Separator()'


def _record(prompt: str, openui: str = _GOLD, placeholders: list[str] | None = None):
    return ExampleRecord(
        id="t1",
        prompt=prompt,
        openui=openui,
        placeholders=[":hero.title", ":hero.body"] if placeholders is None else placeholders,
        split="smoke",
    )


def test_coverage_class_observable_prompt():
    record = _record("Hero card.\nPlaceholders: :hero.title, :hero.body")
    req_a, _ = build_matched_requests(record)
    assert coverage_class(record, req_a) == OBSERVABLE_PROMPT


def test_coverage_class_request_only():
    # Prompt has no inventory section and names no components; coverage comes
    # only from the request slot contract.
    record = _record("Make the hero thing described.")
    req_a, _ = build_matched_requests(record)
    assert coverage_class(record, req_a) == OBSERVABLE_REQUEST_ONLY
    # Without the request the same record is unknown.
    assert coverage_class(record, None) == UNKNOWN


def test_coverage_class_unknown():
    record = _record("Make something.", openui=_GOLD_NO_SLOTS, placeholders=[])
    req_a, _ = build_matched_requests(record)
    assert coverage_class(record, req_a) == UNKNOWN


def test_matched_requests_differ_only_by_inventory_suffix():
    record = _record("Make the hero thing described.")
    req_a, req_b = build_matched_requests(record)
    assert req_b.prompt != req_a.prompt
    suffix = req_b.prompt[len(req_a.prompt.rstrip()):]
    assert suffix.startswith("\nPlaceholders: ")
    assert set(extract_placeholders(suffix)) == set(req_a.slot_contract)
    # Every other request field is identical.
    data_a, data_b = req_a.to_dict(), req_b.to_dict()
    data_a.pop("prompt")
    data_b.pop("prompt")
    assert data_a == data_b


def test_inventory_never_rewrites_inventoried_prompt():
    prompt = "Hero card.\nPlaceholders: :hero.title, :hero.body"
    assert ensure_prompt_inventory(prompt, [":other.slot"]) == prompt
    record = _record(prompt)
    req_a, req_b = build_matched_requests(record)
    assert req_b.prompt == req_a.prompt


def test_prompts_carry_no_placeholders_beyond_request_contract():
    # Audit: over the real smoke corpus, the +inventory arm prompt may only
    # ever expose placeholders from the arm-A request slot contract — never
    # extra gold placeholders.
    with TEST_SEEDS.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert rows, "test_seeds.jsonl missing"
    for row in rows:
        record = ExampleRecord(
            id=row["id"],
            prompt=row["prompt"],
            openui=row["openui"],
            placeholders=list(row.get("placeholders", [])),
            split=row.get("split", "smoke"),
        )
        req_a, req_b = build_matched_requests(record)
        contract = set(req_a.slot_contract)
        for prompt in (req_a.prompt, req_b.prompt):
            leaked = set(extract_placeholders(prompt)) - contract
            assert not leaked, f"{record.id}: prompt leaks non-contract slots {leaked}"


def test_paired_math_synthetic():
    pairs = [
        ("r1", False, True),
        ("r2", True, True),
        ("r3", False, False),
        ("r4", True, False),
        ("r5", None, True),  # unmeasured arm A: excluded from the delta
    ]
    table = paired_outcome_table(pairs)
    assert table == {
        "both_pass": 1,
        "a_pass_b_fail": 1,
        "a_fail_b_pass": 1,
        "both_fail": 1,
        "unpaired": 1,
    }
    result = paired_delta(pairs)
    assert result["n_measured"] == 4
    assert result["n_unpaired"] == 1
    assert result["delta"] == 0.0
    assert result["meets_min_useful_delta"] is False
    win = paired_delta([("r1", False, True), ("r2", False, True)])
    assert win["delta"] == 1.0
    assert win["meets_min_useful_delta"] is True
    empty = paired_delta([("r1", None, None)])
    assert empty["delta"] is None
    assert MIN_USEFUL_PAIRED_DELTA == 0.10


def test_phase0_report_deterministic():
    records = [
        _record("Hero card.\nPlaceholders: :hero.title, :hero.body"),
        ExampleRecord(
            id="t2",
            prompt="Make the hero thing described.",
            openui=_GOLD,
            placeholders=[":hero.title", ":hero.body"],
            split="smoke",
        ),
        ExampleRecord(
            id="t3",
            prompt="Make something.",
            openui=_GOLD_NO_SLOTS,
            placeholders=[],
            split="smoke",
        ),
    ]
    first = observability_report(records)
    second = observability_report(records)
    assert first == second
    assert first["n"] == 3
    assert first["class_histogram"] == {
        OBSERVABLE_PROMPT: 1,
        OBSERVABLE_REQUEST_ONLY: 1,
        UNKNOWN: 1,
    }
    assert [row["id"] for row in first["rows"]] == ["t1", "t2", "t3"]
