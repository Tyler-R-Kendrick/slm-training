"""Regression tests for SLM-313's bounded local intervention runner."""

from __future__ import annotations

from scripts.run_abstract_plan_functional_evidence import _pair_map


def test_shuffle_pairing_is_deterministic_and_has_no_fixed_points() -> None:
    ids = ("a", "b", "c", "d")
    first, digest = _pair_map(ids, "f" * 64)
    second, second_digest = _pair_map(ids, "f" * 64)

    assert first == second
    assert digest == second_digest
    assert set(first) == set(ids)
    assert set(first.values()) == set(ids)
    assert all(record_id != paired for record_id, paired in first.items())
