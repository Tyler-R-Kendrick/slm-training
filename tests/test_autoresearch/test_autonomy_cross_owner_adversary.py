"""Independently authored cross-owner guard tests, not copied implementations."""

from __future__ import annotations

import itertools
from fractions import Fraction
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from scripts.check_changed import select_tests
from slm_training.autoresearch.paired_stats import (
    PairedSelection,
    exact_sign_test,
    paired_record_screening,
    paired_screening_test,
)


def _covers(targets, node):
    return any(node == target or node.startswith(target + "/") for target in targets)


@given(
    st.lists(
        st.sampled_from(
            [
                "tests/test_autoresearch/test_paired_stats.py",
                "tests/test_scripts/test_check_changed.py",
                "new_unknown_runtime_config.toml",
                "package.json",
                "tests/conftest.py",
            ]
        ),
        unique=True,
    )
)
def test_required_coverage_monotone_over_actual_test_files(extra):
    root = Path(__file__).resolve().parents[2]
    source = ["src/slm_training/autoresearch/paired_stats.py"]
    before = select_tests(source)
    after = select_tests(source + extra)
    files = [
        path.relative_to(root).as_posix()
        for path in (root / "tests/test_autoresearch").glob("test_*.py")
    ]
    required = {path for path in files if _covers(before, path)}
    assert required, "coverage test cannot be vacuous"
    assert required <= {path for path in files if _covers(after, path)}


def test_unknown_config_alongside_known_test_requires_conservative_coverage():
    assert select_tests(
        [
            "unregistered/config.yaml",
            "tests/test_scripts/test_check_changed.py",
        ]
    ) == ["tests"]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf"), True])
def test_nonfinite_or_boolean_input_never_authorizes_pair(invalid):
    with pytest.raises(ValueError, match="finite"):
        paired_screening_test([1.0] * 6 + [invalid])
    result = paired_record_screening(
        {str(i): 2.0 for i in range(7)},
        {**{str(i): 1.0 for i in range(6)}, "6": invalid},
        selection=PairedSelection(record_ids=tuple(str(i) for i in range(7))),
    )
    assert not result["diagnostic_complete"]
    assert not result["win"]
    assert not result["promotion_authority"]


def test_equal_counts_disjoint_ids_never_form_pairs():
    result = paired_record_screening(
        {f"a{i}": 2.0 for i in range(6)},
        {f"b{i}": 1.0 for i in range(6)},
    )
    assert result["n_pairs"] == 0
    assert not result["diagnostic_complete"]
    assert not result["win"]


def test_related_roots_and_missing_hardest_case_preclude_win():
    control = {str(i): 2.0 for i in range(7)}
    candidate = {str(i): 1.0 for i in range(7)}
    roots = dict.fromkeys(control, "shared-root")
    related = paired_record_screening(
        control,
        candidate,
        selection=PairedSelection(record_ids=tuple(control), root_ids=roots),
    )
    assert related["diagnostic_complete"]
    assert related["root_family_n"] == 1
    assert not related["win"] and not related["independence_contract_met"]
    candidate.pop("6")
    missing = paired_record_screening(
        control, candidate, selection=PairedSelection(record_ids=tuple(control))
    )
    assert missing["n_pairs"] == 6
    assert not missing["diagnostic_complete"] and not missing["win"]


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4, 5, 6, 7])
def test_exact_sign_probability_matches_enumerated_independent_signs(count):
    # Enumerate the independent null distribution, not the implementation's formula.
    signs = list(itertools.product((-1, 1), repeat=count))
    for positives in range(count + 1):
        observed = [1] * positives + [-1] * (count - positives)
        deviation = abs(sum(observed))
        extreme = sum(abs(sum(sample)) >= deviation for sample in signs)
        _, _, probability = exact_sign_test(observed)
        assert probability == Fraction(extreme, len(signs))


def test_complete_ties_are_inconclusive_not_proof_of_no_useful_effect():
    result = paired_screening_test([0.0] * 10)
    assert result.verdict == "inconclusive"
    assert not result.promotion_authority


def test_finite_metric_guard_mutation_is_detected(monkeypatch):
    import slm_training.autoresearch.paired_stats as owner

    def rejection_oracle():
        with pytest.raises(ValueError):
            owner.exact_sign_test([float("nan")])

    rejection_oracle()
    with monkeypatch.context() as mutation:
        mutation.setattr(owner, "_finite_deltas", lambda values: list(values))
        with pytest.raises(pytest.fail.Exception, match="DID NOT RAISE"):
            rejection_oracle()


def test_pairing_missingness_guard_mutation_is_detected(monkeypatch):
    import slm_training.autoresearch.paired_stats as owner

    actual = owner.paired_record_deltas

    def rejection_oracle():
        result = owner.paired_record_screening(
            {str(i): 2.0 for i in range(7)},
            {str(i): 1.0 for i in range(6)},
            selection=PairedSelection(
                record_ids=tuple(str(i) for i in range(7)),
                root_ids={str(i): f"root-{i}" for i in range(7)},
            ),
        )
        assert not result["win"]

    rejection_oracle()

    def mutation(*args, **kwargs):
        pairs = actual(*args, **kwargs)
        return owner.PairedRecordDeltas(pairs.record_ids, pairs.deltas, 0, 0)

    with monkeypatch.context() as patch:
        patch.setattr(owner, "paired_record_deltas", mutation)
        with pytest.raises(AssertionError):
            rejection_oracle()


def test_keyed_pairing_guard_mutation_is_detected(monkeypatch):
    """Equal arm counts with disjoint IDs must not become positional pairs."""
    import slm_training.autoresearch.paired_stats as owner

    control = {f"a{i}": 2.0 for i in range(6)}
    candidate = {f"b{i}": 1.0 for i in range(6)}
    selection = PairedSelection(
        record_ids=tuple(control),
        root_ids={record_id: f"root-{i}" for i, record_id in enumerate(control)},
    )

    def rejection_oracle():
        result = owner.paired_record_screening(
            control, candidate, selection=selection
        )
        assert result["n_pairs"] == 0
        assert not result["diagnostic_complete"]
        assert not result["win"]

    rejection_oracle()
    actual = owner.paired_record_deltas

    def positional_mutation(*args, **kwargs):
        pairs = actual(*args, **kwargs)
        return owner.PairedRecordDeltas(
            tuple(control), tuple([1.0] * len(control)), 0, 0
        )

    with monkeypatch.context() as patch:
        patch.setattr(owner, "paired_record_deltas", positional_mutation)
        with pytest.raises(AssertionError):
            rejection_oracle()
