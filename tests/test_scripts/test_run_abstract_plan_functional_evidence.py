"""Regression tests for SLM-313's bounded local intervention runner."""

from __future__ import annotations

import pytest

from scripts.run_abstract_plan_functional_evidence import (
    _load_complete_shards,
    _locked_records,
    _pair_map,
    _path_overrides,
    _record_complexity,
)
from slm_training.harnesses.experiments.abstract_plan_functional_evidence import load_locked_protocol


def test_shuffle_pairing_is_deterministic_and_has_no_fixed_points() -> None:
    ids = ("a", "b", "c", "d")
    first, digest = _pair_map(ids, "f" * 64)
    second, second_digest = _pair_map(ids, "f" * 64)

    assert first == second
    assert digest == second_digest
    assert set(first) == set(ids)
    assert set(first.values()) == set(ids)
    assert all(record_id != paired for record_id, paired in first.items())


def test_raw_path_is_constrained_shadow_not_unsafe_emission() -> None:
    raw = _path_overrides("raw")
    assert raw["grammar_constrained"] is True
    assert raw["grammar_ltr_repair"] is False
    assert raw["raw_shadow"] is True


def test_locked_rows_are_projected_to_opaque_request_local_slots() -> None:
    records = _locked_records()
    assert len(records) == 226
    assert all(
        marker.startswith(":slot_")
        for record in records.values()
        for marker in record.placeholders
    )
    binder_count, reference_diameter = _record_complexity(next(iter(records.values())))
    assert binder_count >= 0
    assert reference_diameter >= 0


def test_merge_refuses_partial_locked_coverage(tmp_path) -> None:
    with pytest.raises(ValueError, match="incomplete locked shard coverage"):
        _load_complete_shards(tmp_path, protocol=load_locked_protocol(), shard_count=226)
