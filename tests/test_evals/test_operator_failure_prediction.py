"""Regression tests for the DSH3-31/SLM-406 replay-grounded operator-decode
failure-prediction taxonomy and feature-arm evaluation mechanism.

All fixtures here are small, hand-built ``dict[str, float]`` feature rows
(mirroring the fixture-construction style of
``tests/test_models/test_operator_hard_negative_training.py`` and
``tests/test_models/test_operator_ecoc.py``) -- no real ``DecodeStats``
production traces and no OpenUI bridge subprocess are involved.
"""

from __future__ import annotations

import pytest

from slm_training.evals.operator_failure_prediction import (
    ALL_COUNTER_FEATURES,
    PREREGISTERED_COMPACT_FEATURES,
    OperatorFailurePredictionReportV1,
    ReplayOutcomeLabel,
    assert_group_disjoint_split,
    auprc,
    auroc,
    classify_replay_outcome,
    evaluate_failure_prediction_arms,
    extract_features,
    score_from_features,
)


# --- classify_replay_outcome precedence -------------------------------------


def test_timeout_wins_over_every_other_signal() -> None:
    label = classify_replay_outcome(
        timed_out=True,
        illegal_selection=True,
        executor_rejected=True,
        replay_failed=True,
        dead_end=True,
        semantic_miss=True,
        credited_defer=True,
    )
    assert label is ReplayOutcomeLabel.TIMEOUT


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"illegal_selection": True}, ReplayOutcomeLabel.ILLEGAL_SELECTION),
        ({"executor_rejected": True}, ReplayOutcomeLabel.EXECUTOR_REJECTION),
        ({"replay_failed": True}, ReplayOutcomeLabel.REPLAY_FAILURE),
        ({"dead_end": True}, ReplayOutcomeLabel.DEAD_END),
        ({"semantic_miss": True}, ReplayOutcomeLabel.SEMANTIC_MISS),
        ({"timed_out": True}, ReplayOutcomeLabel.TIMEOUT),
        ({"credited_defer": True}, ReplayOutcomeLabel.SAFE_DEFER),
    ],
)
def test_each_label_is_reachable_in_isolation(kwargs: dict, expected: ReplayOutcomeLabel) -> None:
    assert classify_replay_outcome(**kwargs) is expected


def test_illegal_selection_outranks_executor_rejection_and_below() -> None:
    label = classify_replay_outcome(
        illegal_selection=True,
        executor_rejected=True,
        replay_failed=True,
        dead_end=True,
        semantic_miss=True,
        credited_defer=True,
    )
    assert label is ReplayOutcomeLabel.ILLEGAL_SELECTION


def test_no_signal_and_no_credited_defer_raises() -> None:
    with pytest.raises(ValueError):
        classify_replay_outcome()


# --- auroc / auprc sanity -----------------------------------------------


def test_auroc_perfect_separation_is_one() -> None:
    scored = [(0.9, True), (0.8, True), (0.2, False), (0.1, False)]
    assert auroc(scored) == 1.0


def test_auroc_identical_scores_across_classes_is_one_half() -> None:
    scored = [(0.5, True), (0.5, True), (0.5, False), (0.5, False)]
    assert auroc(scored) == 0.5


def test_auroc_all_one_class_is_none() -> None:
    assert auroc([(0.9, True), (0.8, True)]) is None
    assert auroc([(0.9, False), (0.8, False)]) is None
    assert auroc([]) is None


def test_auprc_no_positives_is_none() -> None:
    assert auprc([(0.9, False), (0.1, False)]) is None


def test_auprc_perfect_separation_is_one() -> None:
    scored = [(0.9, True), (0.8, True), (0.2, False), (0.1, False)]
    assert auprc(scored) == pytest.approx(1.0)


# --- assert_group_disjoint_split ---------------------------------------


def test_assert_group_disjoint_split_raises_on_overlap() -> None:
    with pytest.raises(ValueError):
        assert_group_disjoint_split(["req-1", "req-2"], ["req-2", "req-3"])


def test_assert_group_disjoint_split_passes_on_disjoint_sets() -> None:
    assert_group_disjoint_split(["req-1", "req-2"], ["req-3", "req-4"]) is None


# --- feature-name constants ----------------------------------------------


def test_preregistered_compact_features_matches_the_exact_named_set() -> None:
    assert PREREGISTERED_COMPACT_FEATURES == (
        "compiler_lattice_false_hard_eliminations",
        "compiler_lattice_selector_regret",
        "compiler_lattice_invalid_selected_over_valid",
        "constrained_dead_ends",
        "constrained_last_legal_candidates",
    )


def test_all_counter_features_is_a_superset_of_the_compact_subset() -> None:
    assert set(PREREGISTERED_COMPACT_FEATURES) <= set(ALL_COUNTER_FEATURES)
    assert len(ALL_COUNTER_FEATURES) > len(PREREGISTERED_COMPACT_FEATURES)


def test_extract_features_pulls_named_fields_off_a_decode_stats_instance() -> None:
    from slm_training.models.decode_stats import DecodeStats

    stats = DecodeStats(
        compiler_lattice_false_hard_eliminations=3,
        compiler_lattice_selector_regret=0.25,
    )
    features = extract_features(stats, feature_names=PREREGISTERED_COMPACT_FEATURES)
    assert features["compiler_lattice_false_hard_eliminations"] == 3.0
    assert features["compiler_lattice_selector_regret"] == 0.25
    assert set(features) == set(PREREGISTERED_COMPACT_FEATURES)


def test_score_from_features_is_an_equal_weighted_sum() -> None:
    features = {"a": 2.0, "b": 3.0, "c": 100.0}
    assert score_from_features(features, ("a", "b")) == 5.0


def test_score_from_features_empty_feature_names_is_always_zero() -> None:
    features = {"a": 2.0, "b": 3.0}
    assert score_from_features(features, ()) == 0.0


# --- the core DSH3-31 hypothesis-mechanism test -----------------------------


def _failure_row(
    *,
    label: ReplayOutcomeLabel,
    false_hard_eliminations: float,
    invalid_selected_over_valid: float,
    dead_ends: float,
    last_legal_candidates: float,
    selector_regret: float,
) -> tuple[dict[str, float], ReplayOutcomeLabel]:
    return (
        {
            "compiler_lattice_false_hard_eliminations": false_hard_eliminations,
            "compiler_lattice_invalid_selected_over_valid": invalid_selected_over_valid,
            "constrained_dead_ends": dead_ends,
            "constrained_last_legal_candidates": last_legal_candidates,
            "compiler_lattice_selector_regret": selector_regret,
        },
        label,
    )


def _build_hypothesis_fixture() -> list[tuple[dict[str, float], ReplayOutcomeLabel]]:
    """6 rows: 3 replay-grounded failures with large compact-subset counters
    but a ``selector_regret`` deliberately overlapping the SAFE_DEFER rows'
    regret range, and 3 SAFE_DEFER rows with near-zero compact-subset
    counters. By construction:

    * ``compact_subset`` (sum of all 5 ``PREREGISTERED_COMPACT_FEATURES``)
      perfectly ranks the 3 failures above the 3 SAFE_DEFER rows (the 4
      non-regret counters dominate the sum), giving ``auroc == 1.0``.
    * ``entropy_margin_baseline`` (``compiler_lattice_selector_regret`` alone)
      does not separate them: failure regrets are ``{0.45, 0.50, 0.55}`` and
      SAFE_DEFER regrets are ``{0.40, 0.50, 0.60}`` -- fully interleaved, so
      the rank-sum AUROC works out to exactly ``0.5`` (no better than
      chance).
    """
    return [
        _failure_row(
            label=ReplayOutcomeLabel.DEAD_END,
            false_hard_eliminations=5,
            invalid_selected_over_valid=4,
            dead_ends=3,
            last_legal_candidates=6,
            selector_regret=0.50,
        ),
        _failure_row(
            label=ReplayOutcomeLabel.REPLAY_FAILURE,
            false_hard_eliminations=6,
            invalid_selected_over_valid=5,
            dead_ends=4,
            last_legal_candidates=5,
            selector_regret=0.45,
        ),
        _failure_row(
            label=ReplayOutcomeLabel.EXECUTOR_REJECTION,
            false_hard_eliminations=7,
            invalid_selected_over_valid=3,
            dead_ends=5,
            last_legal_candidates=4,
            selector_regret=0.55,
        ),
        _failure_row(
            label=ReplayOutcomeLabel.SAFE_DEFER,
            false_hard_eliminations=0,
            invalid_selected_over_valid=0,
            dead_ends=0,
            last_legal_candidates=0,
            selector_regret=0.50,
        ),
        _failure_row(
            label=ReplayOutcomeLabel.SAFE_DEFER,
            false_hard_eliminations=0,
            invalid_selected_over_valid=0,
            dead_ends=0,
            last_legal_candidates=1,
            selector_regret=0.40,
        ),
        _failure_row(
            label=ReplayOutcomeLabel.SAFE_DEFER,
            false_hard_eliminations=1,
            invalid_selected_over_valid=0,
            dead_ends=0,
            last_legal_candidates=0,
            selector_regret=0.60,
        ),
    ]


def test_compact_subset_strictly_beats_entropy_margin_baseline_on_fixture() -> None:
    rows = _build_hypothesis_fixture()
    arms = {
        "compact_subset": PREREGISTERED_COMPACT_FEATURES,
        "entropy_margin_baseline": ("compiler_lattice_selector_regret",),
    }
    report = evaluate_failure_prediction_arms(rows, arms)
    by_arm = {arm.arm: arm for arm in report.arms}

    assert by_arm["compact_subset"].auroc == 1.0
    assert by_arm["entropy_margin_baseline"].auroc == 0.5
    assert by_arm["compact_subset"].auroc > by_arm["entropy_margin_baseline"].auroc
    assert report.best_arm_by_auroc == "compact_subset"


def test_report_to_dict_round_trips_expected_keys_and_best_arm() -> None:
    rows = _build_hypothesis_fixture()
    arms = {
        "compact_subset": PREREGISTERED_COMPACT_FEATURES,
        "entropy_margin_baseline": ("compiler_lattice_selector_regret",),
        "context_free_baseline": (),
    }
    report = evaluate_failure_prediction_arms(rows, arms)
    assert isinstance(report, OperatorFailurePredictionReportV1)
    payload = report.to_dict()
    assert payload["schema"] == "operator_failure_prediction_report/v1"
    assert {arm["arm"] for arm in payload["arms"]} == {
        "compact_subset",
        "entropy_margin_baseline",
        "context_free_baseline",
    }
    for arm_payload in payload["arms"]:
        assert set(arm_payload) == {
            "schema",
            "arm",
            "feature_names",
            "auroc",
            "auprc",
            "calibration_error",
            "n_examples",
            "n_positive",
        }
        assert arm_payload["n_examples"] == 6
        assert arm_payload["n_positive"] == 3
    assert payload["best_arm_by_auroc"] == "compact_subset"


def test_context_free_baseline_is_a_constant_zero_score_and_undiscriminative() -> None:
    rows = _build_hypothesis_fixture()
    report = evaluate_failure_prediction_arms(rows, {"context_free_baseline": ()})
    result = report.arms[0]
    # A constant score for every row means every score ties -- AUROC is the
    # tie-handling degenerate value (0.5), never better than chance.
    assert result.auroc == 0.5
