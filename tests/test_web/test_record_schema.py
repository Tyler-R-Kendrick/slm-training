"""Normalizer contract for the committed experiment-record dialects."""

from __future__ import annotations

from slm_training.evals.record_schema import (
    canonical_envelope,
    normalize_experiment_record,
    normalize_suite_metrics,
)
import pytest


def test_short_keys_normalize_to_canonical() -> None:
    metrics = normalize_suite_metrics(
        {
            "n": 3,
            "parse": 1.0,
            "meaningful": 0.0,
            "fidelity": 0.33,
            "structure": 0.35,
            "component_recall": 0.0,
            "reward": 0.0,
        }
    )
    assert metrics["parse_rate"] == 1.0
    assert metrics["meaningful_program_rate"] == 0.0
    assert metrics["placeholder_fidelity"] == 0.33
    assert metrics["structural_similarity"] == 0.35
    assert metrics["component_type_recall"] == 0.0
    assert metrics["reward_score"] == 0.0
    # Explicit meaningful present → no legacy substitution, no tag.
    assert "meaningful_source" not in metrics


def test_legacy_parse_fallback_is_guarded_and_tagged() -> None:
    # Pre-split board: neither meaningful nor syntax_parse_rate → guarded
    # fallback applies and is tagged for consumers to badge.
    legacy = normalize_suite_metrics({"n": 5, "parse_rate": 0.8})
    assert legacy["meaningful_program_rate"] == 0.8
    assert legacy["meaningful_source"] == "parse_rate_legacy"
    # Post-split board: syntax_parse_rate present → parse_rate is
    # decoder-guaranteed syntax and must NOT feed the meaningful lever.
    modern = normalize_suite_metrics({"n": 5, "parse_rate": 1.0, "syntax_parse_rate": 1.0})
    assert modern.get("meaningful_program_rate") is None
    assert "meaningful_source" not in modern


def test_suites_dialects_normalize() -> None:
    honest = {
        "run_id": "e295-r1",
        "honest_evaluation": {
            "suites": {
                "smoke": {"n": 3, "parse": 1.0, "meaningful": 0.0, "fidelity": 0.3},
            }
        },
    }
    record, reason = normalize_experiment_record(honest, stem="e295")
    assert reason is None and record is not None
    assert record["run_id"] == "e295-r1"
    assert record["source_schema"] == "suites@honest_evaluation"
    assert record["suites"]["smoke"]["meaningful_program_rate"] == 0.0

    single = {
        "experiment": "E108",
        "eval": {"suite": "smoke", "n": 1, "parse_rate": 0.0, "structural_similarity": 0.0},
    }
    record, reason = normalize_experiment_record(single, stem="iter-e108-seed1")
    assert reason is None and record is not None
    # run_id synthesized from the filename stem when the payload has none.
    assert record["run_id"] == "iter-e108-seed1"
    assert list(record["suites"]) == ["smoke"]


def test_non_experiment_records_get_typed_reasons() -> None:
    record, reason = normalize_experiment_record(["not", "a", "dict"], stem="x")
    assert record is None and reason == "unreadable"
    record, reason = normalize_experiment_record({"title": "design note"}, stem="x")
    assert record is None and reason == "no_metric_blocks"


def test_canonical_envelope_validates() -> None:
    envelope = canonical_envelope(
        run_id="run-1",
        suites={"smoke": {"n": 3, "parse_rate": 1.0}},
        run_class="fixture_demo",
    )
    assert envelope["schema_version"] == 1
    assert envelope["run_class"] == "fixture_demo"
    with pytest.raises(ValueError):
        canonical_envelope(run_id="run-1", suites={}, run_class="production")
