"""Exact-arithmetic tests for slm_training.autoresearch.power."""

from __future__ import annotations

import math
from fractions import Fraction

import pytest

from slm_training.autoresearch.evidence_ledger import (
    required_sign_test_n,
    sign_test_min_two_sided_p,
)
from slm_training.autoresearch.power import (
    DecidabilityReport,
    is_decidable,
    min_attainable_n,
    min_attainable_p,
    min_detectable_effect,
    required_n_for_effect,
)

Z_TOTAL = 1.9599639845400545 + 0.8416212335729143  # z(0.975) + z(0.80)


class TestMinAttainableP:
    def test_unpaired_3v3_matches_exact_combinatorics(self) -> None:
        # C(6, 3) = 20 labelings: one-sided floor 1/20, two-sided 2/20.
        assert min_attainable_p(3, 3, sided=1) == pytest.approx(0.05)
        assert min_attainable_p(3, 3) == pytest.approx(0.1)
        # RC1 framing: at n=3 vs n=3 the two-sided floor exceeds alpha=0.05.
        assert min_attainable_p(3, 3) > 0.05

    def test_unpaired_unbalanced(self) -> None:
        assert min_attainable_p(2, 4) == pytest.approx(2 / math.comb(6, 2))
        assert min_attainable_p(4, 2) == min_attainable_p(2, 4)

    def test_paired_matches_review_doc_and_ledger(self) -> None:
        # Review doc RC1: paired sign-test minimum two-sided p = 0.25 at n=3.
        assert min_attainable_p(3, 3, paired=True) == pytest.approx(0.25)
        # Reuses darkfactory phase 1 exact arithmetic.
        for n in range(1, 12):
            assert min_attainable_p(n, n, paired=True) == pytest.approx(
                float(sign_test_min_two_sided_p(n))
            )
        assert min_attainable_p(3, 3, paired=True, sided=1) == pytest.approx(0.125)

    def test_paired_uses_min_of_unequal_ns(self) -> None:
        assert min_attainable_p(3, 9, paired=True) == pytest.approx(0.25)

    def test_no_evidence_and_caps(self) -> None:
        assert min_attainable_p(0, 5) == 1.0
        assert min_attainable_p(5, 0) == 1.0
        assert min_attainable_p(1, 1) == 1.0  # 2/C(2,1) = 1
        assert min_attainable_p(1, 1, paired=True) == 1.0  # 2*(1/2)^1 = 1

    def test_monotone_decreasing_in_n(self) -> None:
        for paired in (False, True):
            values = [min_attainable_p(n, n, paired=paired) for n in range(1, 10)]
            assert values == sorted(values, reverse=True)

    def test_invalid_arguments_raise(self) -> None:
        with pytest.raises(ValueError):
            min_attainable_p(3, 3, test="t-test")
        with pytest.raises(ValueError):
            min_attainable_p(3, 3, sided=3)


class TestMinAttainableN:
    def test_paired_matches_darkfactory_required_sign_test_n(self) -> None:
        assert min_attainable_n(0.05, paired=True) == 6
        assert min_attainable_n(0.05, paired=True) == required_sign_test_n(
            Fraction(1, 20)
        )

    def test_unpaired_floor(self) -> None:
        # 2/C(8,4) = 2/70 <= 0.05 while 2/C(6,3) = 0.1 > 0.05.
        assert min_attainable_n(0.05) == 4
        assert min_attainable_p(4, 4) <= 0.05

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError):
            min_attainable_n(0.0)


class TestNormalApproximation:
    def test_min_detectable_effect_unpaired_formula(self) -> None:
        assert min_detectable_effect(8, 0.1) == pytest.approx(
            Z_TOTAL * 0.1 * math.sqrt(2 / 8)
        )

    def test_min_detectable_effect_paired_formula(self) -> None:
        assert min_detectable_effect(8, 0.1, paired=True) == pytest.approx(
            Z_TOTAL * 0.1 / math.sqrt(8)
        )

    def test_required_n_round_trips_with_mde(self) -> None:
        for paired in (False, True):
            for effect in (0.01, 0.05, 0.2):
                n = required_n_for_effect(effect, 0.1, paired=paired)
                assert (
                    min_detectable_effect(n, 0.1, paired=paired) <= effect + 1e-12
                )
                if n > 1:
                    assert min_detectable_effect(n - 1, 0.1, paired=paired) > effect

    def test_required_n_zero_sd(self) -> None:
        assert required_n_for_effect(0.01, 0.0) == 1

    def test_invalid_arguments_raise(self) -> None:
        with pytest.raises(ValueError):
            min_detectable_effect(0, 0.1)
        with pytest.raises(ValueError):
            min_detectable_effect(3, -0.1)
        with pytest.raises(ValueError):
            min_detectable_effect(3, 0.1, alpha=0.0)
        with pytest.raises(ValueError):
            min_detectable_effect(3, 0.1, power=1.0)
        with pytest.raises(ValueError):
            required_n_for_effect(0.0, 0.1)
        with pytest.raises(ValueError):
            required_n_for_effect(0.01, -1.0)


class TestIsDecidable:
    def test_rc1_screening_design_is_undecidable(self) -> None:
        # The loop's screening design: n=3 seeds, minimum_effect 0.01, paired.
        report = is_decidable(3, 0.0312, 0.01, paired=True)
        assert isinstance(report, DecidabilityReport)
        assert report.decidable is False
        assert report.min_attainable_p == pytest.approx(0.25)
        assert report.required_n_for_effect >= 6
        joined = " ".join(report.reasons)
        assert "min_attainable_p=0.25" in joined
        assert "required_n_for_effect" in joined

    def test_decidable_design_passes_both_axes(self) -> None:
        report = is_decidable(8, 0.0312, 0.05, paired=True)
        assert report.decidable is True
        assert report.min_attainable_p == pytest.approx(2 / 2**8)
        assert report.min_detectable_effect <= 0.05
        assert report.reasons  # explanatory even when decidable

    def test_required_n_includes_attainability_floor(self) -> None:
        # Tiny sd makes the analytic n small; the sign-test floor (6) binds.
        report = is_decidable(3, 1e-6, 0.01, paired=True)
        assert report.required_n_for_effect >= 6
        followup = is_decidable(
            report.required_n_for_effect, 1e-6, 0.01, paired=True
        )
        assert followup.decidable is True

    def test_degenerate_inputs_never_raise(self) -> None:
        for n, sd, effect, alpha in (
            (0, 0.1, 0.01, 0.05),
            (-3, 0.1, 0.01, 0.05),
            (3, -1.0, 0.01, 0.05),
            (3, 0.1, 0.0, 0.05),
            (3, 0.1, -0.5, 0.05),
            (3, 0.1, 0.01, 0.0),
            (3, 0.1, 0.01, 1.5),
        ):
            report = is_decidable(n, sd, effect, alpha=alpha)
            assert report.decidable is False
            assert report.reasons

    def test_unpaired_variant_uses_permutation_floor(self) -> None:
        report = is_decidable(3, 0.0312, 0.5, paired=False)
        assert report.min_attainable_p == pytest.approx(0.1)
