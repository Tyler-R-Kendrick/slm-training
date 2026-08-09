"""EvidenceRecordV1 + fingerprint contract tests."""

from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from slm_training.evidence_store.records import (
    EVIDENCE_RECORD_SCHEMA,
    EvidenceRecordV1,
    compute_config_fingerprint,
)


def _record(**overrides: object) -> EvidenceRecordV1:
    payload: dict[str, object] = {
        "experiment_id": "exp-1",
        "config_fingerprint": "f" * 64,
        "endpoint_metric": "smoke.structural_similarity",
        "outcome": "screen_negative",
    }
    payload.update(overrides)
    return EvidenceRecordV1.model_validate(payload)


def test_fingerprint_is_key_order_insensitive() -> None:
    assert compute_config_fingerprint({"a": 1, "b": 2}) == compute_config_fingerprint(
        {"b": 2, "a": 1}
    )


def test_fingerprint_distinguishes_configs() -> None:
    assert compute_config_fingerprint({"a": 1}) != compute_config_fingerprint(
        {"a": 2}
    )


def test_fingerprint_matches_evidence_ledger_canonicalization() -> None:
    """Pin the exact canonical form used inline by evidence_ledger.build_ledger."""
    config = {"arm": "bounds", "seed": 3, "nested": {"z": 1, "a": [1, 2]}}
    expected = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    assert compute_config_fingerprint(config) == expected


def test_record_defaults_and_schema() -> None:
    record = _record()
    assert record.schema_version == EVIDENCE_RECORD_SCHEMA
    assert record.lever_keys == []
    assert record.effect_size is None
    assert record.p_value is None
    assert record.blocker is None
    assert record.version_stamp == {}


def test_record_rejects_unknown_outcome() -> None:
    with pytest.raises(ValidationError):
        _record(outcome="maybe_positive")


def test_record_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        _record(surprise_field=1)


def test_dedup_key_is_fingerprint_plus_metric() -> None:
    record = _record()
    assert record.dedup_key() == ("f" * 64, "smoke.structural_similarity")


def test_canonical_line_roundtrips_and_is_stable() -> None:
    record = _record(
        lever_keys=["bounds"],
        effect_size=0.125,
        version_stamp={"components": {"gates.ship": "v4"}},
        run_date="2026-08-09T00:00:00Z",
    )
    line = record.canonical_line()
    assert "\n" not in line
    assert EvidenceRecordV1.model_validate_json(line) == record
    # sort_keys canonicalization: identical record -> identical line.
    assert line == record.canonical_line()
    assert json.loads(line) == json.loads(
        json.dumps(json.loads(line), sort_keys=True)
    )
