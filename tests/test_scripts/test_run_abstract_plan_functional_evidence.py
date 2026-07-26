"""Regression tests for SLM-313's bounded local intervention runner."""

from __future__ import annotations

from scripts.run_abstract_plan_functional_evidence import _pair_map, _path_overrides


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
