"""Wilcoxon / sign-test paired screening decisions."""

from __future__ import annotations

from slm_training.autoresearch.paired_stats import (
    exact_sign_test,
    paired_screening_test,
    wilcoxon_signed_rank_p,
)


def test_wilcoxon_all_positive_matches_sign_flip_count() -> None:
    deltas = [1.0, 2.0, 3.0, 4.0, 5.0]
    w_plus, p, n = wilcoxon_signed_rank_p(deltas)
    assert n == 5
    assert w_plus == 15.0
    # One of 32 sign patterns is all-positive; two-sided = 2/32.
    assert abs(p - 2 / 32) < 1e-12


def test_sign_test_known_vector() -> None:
    pos, neg, p = exact_sign_test([1, 1, 1, 1, -1])
    assert pos == 4 and neg == 1
    # 2 * C(5,0..1) / 32 = 2 * 6 / 32 = 12/32.
    assert float(p) == 12 / 32


def test_ties_dropped_and_insufficient_nontied_is_mechanism_no_effect() -> None:
    result = paired_screening_test(
        [0.0, 0.0, 0.0, 0.1, -0.1], min_nontied_pairs=5
    )
    assert result.n_ties == 3
    assert result.n_nontied == 2
    assert result.verdict == "mechanism_no_effect"
    assert "nontied_pairs" in result.reason


def test_all_ties_is_mechanism_no_effect() -> None:
    result = paired_screening_test([0.0] * 6)
    assert result.verdict == "mechanism_no_effect"


def test_wilcoxon_win_on_consistent_positive_deltas() -> None:
    # n=12 > sign-test fallback; all-positive is extreme.
    deltas = [float(i) for i in range(1, 13)]
    result = paired_screening_test(deltas, min_nontied_pairs=5)
    assert result.kind == "wilcoxon_signed_rank"
    assert result.verdict == "win"
    assert result.p_value < 0.05


def test_sign_test_fallback_for_tiny_n() -> None:
    deltas = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    result = paired_screening_test(deltas, min_nontied_pairs=5)
    assert result.kind == "sign_test"
    assert result.verdict == "win"


def test_paired_record_deltas_pairs_by_id_and_signs_by_direction() -> None:
    from slm_training.autoresearch.paired_stats import paired_record_deltas

    control = {"r1": 2.0, "r2": 3.0, "r3": 1.0, "only_control": 9.0, "nan": float("nan")}
    candidate = {"r1": 1.5, "r2": 3.5, "r3": 1.0, "only_candidate": 0.1, "nan": 1.0}
    pairs = paired_record_deltas(control, candidate, direction="decrease")
    assert pairs.record_ids == ("r1", "r2", "r3")
    # decrease: control - candidate (positive = candidate better).
    assert pairs.deltas == (0.5, -0.5, 0.0)
    assert pairs.n_pairs == 3
    assert pairs.n_missing_control == 2  # only_control absent + NaN control
    assert pairs.n_missing_candidate == 1
    assert pairs.median_delta == 0.0
    assert pairs.sd is not None and pairs.sd > 0
    increase = paired_record_deltas(control, candidate, direction="increase")
    assert increase.deltas == (-0.5, 0.5, 0.0)


def test_paired_record_screening_win_requires_alpha_and_minimum_effect() -> None:
    from slm_training.autoresearch.paired_stats import paired_record_screening

    control = {f"r{i}": 3.0 + 0.01 * i for i in range(24)}
    candidate = {
        k: v - 0.2 - 0.001 * i for i, (k, v) in enumerate(control.items())
    }
    win = paired_record_screening(
        control, candidate, direction="decrease", minimum_effect=0.05
    )
    assert win["win"] is True
    assert win["n_pairs"] == 24
    assert win["p_value"] < 0.05
    assert win["median_delta"] > 0.05
    assert win["paired_sd"] is not None
    assert win["promotion_authority"] is False

    # Significant but below the policy minimum effect: not a win.
    small = {k: v - 0.02 for k, v in control.items()}
    sub = paired_record_screening(
        control, small, direction="decrease", minimum_effect=0.05
    )
    assert sub["win"] is False
    assert sub["p_value"] < 0.05

    # Three pairs can never reach alpha: mechanism_no_effect, never a win.
    tiny = paired_record_screening(
        dict(list(control.items())[:3]),
        dict(list(candidate.items())[:3]),
        direction="decrease",
        minimum_effect=0.05,
    )
    assert tiny["win"] is False
    assert tiny["verdict"] == "mechanism_no_effect"
    assert tiny["n_pairs"] == 3

    # A significant regression is a loss, not a win.
    worse = {k: v + 0.3 for k, v in control.items()}
    loss = paired_record_screening(
        control, worse, direction="decrease", minimum_effect=0.05
    )
    assert loss["win"] is False and loss["verdict"] == "loss"
