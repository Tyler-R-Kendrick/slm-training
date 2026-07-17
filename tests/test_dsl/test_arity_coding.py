"""Regression tests for CAP0-03 coding-theory and precision reference functions."""
from __future__ import annotations

import pytest

from slm_training.dsl.analysis.arity import (
    ResidualScaleMode,
    assert_geometric_only,
    balanced_ternary_levels,
    build_mds_7_4_2_3,
    build_shortened_ternary_hamming_7_4_3,
    gilbert_greedy_guarantees,
    hamming_ball_volume,
    hamming_sphere_packing_holds,
    minimum_distance,
    minimum_margin_trit_planes,
    singleton_upper_bound,
    smallest_injective_arity,
    ternary_ecoc_width,
    verify_code,
)


def test_smallest_injective_arity():
    assert smallest_injective_arity(41, 4) == 3  # 2^4=16 < 41, 3^4=81 >= 41
    assert smallest_injective_arity(64, 6) == 2  # 2^6=64


def test_hamming_ball_volume():
    # q=2, n=4, radius=1 -> 1 + 4 = 5
    assert hamming_ball_volume(2, 4, 1) == 5
    # q=7, n=4, radius=1 -> 1 + 4*6 = 25
    assert hamming_ball_volume(7, 4, 1) == 25


def test_singleton_rejects_q6_for_toy():
    # M=41, n=4, d=3: q=6 bound is 36 < 41 -> infeasible.
    assert singleton_upper_bound(6, 4, 3) < 41
    # q=7 bound is 343 >= 41 -> feasible by Singleton.
    assert singleton_upper_bound(7, 4, 3) >= 41


def test_mds_7_4_2_3_construction():
    code = build_mds_7_4_2_3()
    result = verify_code(code, q=7, n=4, required_size=49, required_distance=3)
    assert result.ok
    assert result.size == 49
    assert result.minimum_distance == 3


def test_shortened_ternary_hamming_construction():
    code = build_shortened_ternary_hamming_7_4_3()
    result = verify_code(code, q=3, n=7, required_size=81, required_distance=3)
    assert result.ok
    assert result.size == 81
    assert result.minimum_distance == 3


def test_minimum_distance_detects_collision():
    with pytest.raises(ValueError, match="same length"):
        minimum_distance([(0, 0), (0,)], q=2)
    with pytest.raises(ValueError, match="not in F_2"):
        minimum_distance([(0, 0), (0, 2)], q=2)


def test_hamming_sphere_packing_and_gilbert():
    # 49 words in 7^4 space with t=1: 49*25=1225 <= 2401 -> bound holds.
    assert hamming_sphere_packing_holds(49, 7, 4, 1)
    # A small code is guaranteed by Gilbert-Varshamov.
    assert gilbert_greedy_guarantees(9, 7, 4, 3)


def test_strict_margin_regression():
    # error_radius=4, margin=1 -> R=3 (R=2 gives equality, not strict).
    assert minimum_margin_trit_planes(4.0, 1.0) == 3


def test_margin_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="margin"):
        minimum_margin_trit_planes(1.0, 0.0)
    with pytest.raises(ValueError, match="error_radius"):
        minimum_margin_trit_planes(-1.0, 1.0)


def test_ternary_ecoc_width():
    # 5 actions: ceil(log_3 5)=2, with detection -> 3.
    assert ternary_ecoc_width(5, detect_single_trit_error=False) == 2
    assert ternary_ecoc_width(5, detect_single_trit_error=True) == 3


def test_balanced_versus_learned_scale_guard():
    assert balanced_ternary_levels(3) == 27
    assert_geometric_only(ResidualScaleMode.GEOMETRIC_BALANCED)
    with pytest.raises(ValueError, match="invalid for scale mode"):
        assert_geometric_only(ResidualScaleMode.LEARNED_INDEPENDENT)
