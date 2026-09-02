"""Tests for scripts/audit_regular_layer_gap.py (S8 literature-ledger N8 probe)."""

from __future__ import annotations

import json
from pathlib import Path

from scripts import audit_regular_layer_gap


def test_single_record_walk_writes_report(tmp_path: Path) -> None:
    out = tmp_path / "n8.json"
    rc = audit_regular_layer_gap.main(["--limit", "1", "--out", str(out)])
    assert rc == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["schema"] == "regular_layer_gap_probe/v1"
    assert report["records_walked"] == 1
    assert report["records_skipped_by_time_budget"] == []
    # One decision per prefix, including the EOS position after the gold.
    assert report["positions"] == report["gold_tokens"] + 1
    assert (
        report["singleton_or_empty"] + report["nonsingleton"] == report["positions"]
    )
    # The two non-singleton buckets partition at most the non-singleton count.
    assert (
        report["nonsingleton_automaton_strictly_larger"]
        + report["nonsingleton_sets_equal"]
        <= report["nonsingleton"]
    )
    assert report["singleton_forced_by_automaton"] <= report["singleton_or_empty"]
    (row,) = report["per_record"]
    assert row["id"] == "smoke_hero_01"
    assert len(row["gold_misses"]) == row["gold_outside_complete_domain"]


def test_time_budget_skips_records_without_walking(tmp_path: Path) -> None:
    out = tmp_path / "n8_budget.json"
    rc = audit_regular_layer_gap.main(
        ["--limit", "2", "--time-budget", "-1", "--out", str(out)]
    )
    assert rc == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["records_walked"] == 0
    assert report["records_skipped_by_time_budget"] == [
        "smoke_hero_01",
        "smoke_button_01",
    ]
    assert report["gap_fraction"] is None


def test_terminal_map_covers_end_marker() -> None:
    from slm_training.dsl.grammar.fastpath.engine import OpenUIIncrementalEngine
    from slm_training.models.dsl_tokenizer import DSLNativeTokenizer

    tokenizer = DSLNativeTokenizer.build()
    engine = OpenUIIncrementalEngine()
    engine.set_prefix("")
    term_to_ids = audit_regular_layer_gap._terminal_to_ids(tokenizer, engine)
    assert term_to_ids["$END"] == frozenset({tokenizer.eos_id})
    for term in ("COMPONENT", "NAME", "STRING", "NUMBER", "_NL", "LPAR"):
        assert term_to_ids[term], term
