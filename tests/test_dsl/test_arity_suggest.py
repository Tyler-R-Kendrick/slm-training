"""Regression tests for robust coding arm suggestions (CAP0-03)."""

from __future__ import annotations

import pytest

from slm_training.dsl.analysis.arity import (
    RobustArm,
    smallest_feasible_alphabet,
    suggest_robust_arms,
)


def test_smallest_feasible_alphabet_for_toy():
    # M=41, n=4: q=6 infeasible by Singleton, q=7 feasible.
    assert smallest_feasible_alphabet(41, 4) == 7


def test_suggest_robust_arms_includes_feasible_toy_arms():
    arms = suggest_robust_arms(41, dimensions=4, max_alphabet=8, max_distance=8)
    by_key = {(arm.q, arm.n, arm.d): arm for arm in arms}

    # Feasible arms that CAP0-03 keeps.
    assert (7, 4, 3) in by_key
    assert by_key[(7, 4, 3)].construction == "mds_7_4_2_3"

    # Verified ternary shortened Hamming arm has n=7, d=3.
    ternary_arms = [a for a in arms if a.q == 3]
    assert any(a.n == 7 and a.d == 3 for a in ternary_arms)


def test_suggest_robust_arms_excludes_removed_arms():
    arms = suggest_robust_arms(41, dimensions=4, max_alphabet=8, max_distance=8)
    by_key = {(arm.q, arm.n, arm.d): arm for arm in arms}

    # Removed arms: (K=6,d=4) -> (q=6,n=4,d=4) and (K=3,d=6) -> (q=3,n=6,d=6).
    assert (6, 4, 4) not in by_key
    assert (3, 6, 6) not in by_key


def test_suggest_robust_arms_rejects_invalid_input():
    with pytest.raises(ValueError, match="state_count"):
        suggest_robust_arms(0)
    with pytest.raises(ValueError, match="dimensions"):
        suggest_robust_arms(1, dimensions=0)


def test_robust_arm_immutable():
    arm = RobustArm(q=7, n=4, d=3, feasible=True, reason="test", construction=None)
    with pytest.raises(AttributeError):
        arm.q = 6
