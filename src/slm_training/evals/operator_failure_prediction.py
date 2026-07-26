"""Replay-grounded operator-decode failure prediction from ``DecodeStats`` (DSH3-31).

DSH3-31's hypothesis is that a compact subset of already-existing
``DecodeStats`` compiler-lattice/constrained-decode counters
(``compiler_lattice_false_hard_eliminations``,
``compiler_lattice_invalid_selected_over_valid``, ``constrained_dead_ends``,
``constrained_last_legal_candidates``, plus ``compiler_lattice_selector_regret``
as the closest existing margin-like scalar) can predict an operator decode's
eventual replay-grounded failure outcome earlier and better than an
entropy/margin-only baseline (``compiler_lattice_selector_regret`` alone), and
strictly better than a context-free frequency baseline (a constant score).
``src/slm_training/evals/cap2_operator_policy_rebase.py``'s own
forward-reference inventory (see its ``item(...)`` entry for
``models.decode_stats.DecodeStats``) already named these exact counters as
DSH3-31 features and stated no replay-grounded outcome label taxonomy exists
yet on top of them -- this module closes that forward reference.

This module adds two things, and nothing else:

1. :class:`ReplayOutcomeLabel` -- the 7-label replay-grounded outcome
   taxonomy the issue names, plus :func:`classify_replay_outcome`, a strict-
   precedence classifier over boolean replay-evidence signals mirroring
   ``harnesses.model_build.decode_outcome.classify_decode_outcome``'s "strict
   precedence over one struct" shape.
2. A small deterministic feature-arm evaluation harness
   (:func:`extract_features`, :func:`score_from_features`, :func:`auroc`,
   :func:`auprc`, :func:`evaluate_failure_prediction_arms`) that scores a set
   of ``(features, label)`` rows under named feature arms and reports
   discrimination (AUROC/AUPRC) and calibration for each.

Honesty / scope, read before citing any number this module produces:

* **No real production ``DecodeStats`` traces are used anywhere in this
  module's tests or its companion script** -- every row is a small, hand-built
  synthetic fixture. Any AUROC/AUPRC/calibration number here is fixture-scale
  mechanism evidence that the compact subset *can* separate a
  by-construction-separable synthetic set better than the margin-only
  baseline; it is not a claim about real held-out generalization, real
  checkpoint cross-generalization, or production early-abort deployment
  readiness.
* **No time-indexed "earliest decode fraction reaching target precision"
  metric is computed.** That metric needs real time-indexed decode traces
  (a stream of per-step ``DecodeStats`` snapshots with a known eventual
  outcome), which do not exist yet in this repo. This is explicitly out of
  scope here -- it is not approximated or faked.
* **This module changes no production decode/abort behavior.** It is
  shadow-analysis only: a measurement/classification library over already-
  collected counters, never wired into any decode loop, compiler lattice, or
  constrained decode path.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from slm_training.evals.judge_independence import calibration_error

if TYPE_CHECKING:  # pragma: no cover - typing only
    from slm_training.models.decode_stats import DecodeStats

__all__ = [
    "ReplayOutcomeLabel",
    "classify_replay_outcome",
    "PREREGISTERED_COMPACT_FEATURES",
    "ALL_COUNTER_FEATURES",
    "extract_features",
    "score_from_features",
    "auroc",
    "auprc",
    "assert_group_disjoint_split",
    "OperatorFailurePredictionArmResultV1",
    "OperatorFailurePredictionReportV1",
    "evaluate_failure_prediction_arms",
]


class ReplayOutcomeLabel(str, Enum):
    """The 7 replay-grounded operator-decode outcome labels DSH3-31 names.

    No "success"/"model_valid" sentinel is included by design -- see
    :func:`classify_replay_outcome`'s docstring for why a genuinely
    successful decode (no failure/defer signal) is out of scope for this
    exact taxonomy.
    """

    ILLEGAL_SELECTION = "illegal_selection"
    EXECUTOR_REJECTION = "executor_rejection"
    REPLAY_FAILURE = "replay_failure"
    SEMANTIC_MISS = "semantic_miss"
    DEAD_END = "dead_end"
    TIMEOUT = "timeout"
    SAFE_DEFER = "safe_defer"


def classify_replay_outcome(
    *,
    timed_out: bool = False,
    illegal_selection: bool = False,
    executor_rejected: bool = False,
    replay_failed: bool = False,
    dead_end: bool = False,
    semantic_miss: bool = False,
    credited_defer: bool = False,
) -> ReplayOutcomeLabel:
    """Classify one decode's replay evidence into exactly one of the 7 labels.

    Precedence is strict and total:

        ``TIMEOUT > ILLEGAL_SELECTION > EXECUTOR_REJECTION > REPLAY_FAILURE
        > DEAD_END > SEMANTIC_MISS > SAFE_DEFER``

    Rationale (mirrors ``decode_outcome.classify_decode_outcome``'s "strict
    precedence over one struct" shape): a timeout is checked first because it
    can mask any other in-flight signal (a decode that timed out never got
    far enough to prove any of the others cleanly); an illegal selection is
    checked before an executor-level rejection or replay failure because it
    is detectable earlier in the pipeline (at selection time, before
    execution/replay ever runs); a dead end (no legal continuation existed)
    outranks a semantic miss (a legal, replayable decode that simply didn't
    match intent) because a dead end is a harder, structural failure; safe
    DEFER is the fallback only when nothing else fired and the defer was
    credited (mirrors ``DeferFallbackAttributionV1.credited_defer`` --
    ``PARTIAL``/``TIMEOUT``/``BUDGET_EXHAUSTED`` routes with a successful
    fallback).

    A genuinely successful decode -- no failure signal and no credited defer
    -- is not one of these 7 labels; this function requires at least one
    signal to be true and raises ``ValueError`` otherwise, rather than
    silently returning a label that would misrepresent a plain success as a
    taxonomy member.
    """
    if timed_out:
        return ReplayOutcomeLabel.TIMEOUT
    if illegal_selection:
        return ReplayOutcomeLabel.ILLEGAL_SELECTION
    if executor_rejected:
        return ReplayOutcomeLabel.EXECUTOR_REJECTION
    if replay_failed:
        return ReplayOutcomeLabel.REPLAY_FAILURE
    if dead_end:
        return ReplayOutcomeLabel.DEAD_END
    if semantic_miss:
        return ReplayOutcomeLabel.SEMANTIC_MISS
    if credited_defer:
        return ReplayOutcomeLabel.SAFE_DEFER
    raise ValueError(
        "classify_replay_outcome requires at least one signal to be true; a "
        "decode with no failure/defer signal is a plain success, which is "
        "outside this 7-label replay-grounded failure/defer taxonomy"
    )


#: The compact feature subset DSH3-31's hypothesis is about -- the exact
#: counters ``cap2_operator_policy_rebase.py``'s forward-reference inventory
#: named as reusable DSH3-31 features.
PREREGISTERED_COMPACT_FEATURES: tuple[str, ...] = (
    "compiler_lattice_false_hard_eliminations",
    "compiler_lattice_selector_regret",
    "compiler_lattice_invalid_selected_over_valid",
    "constrained_dead_ends",
    "constrained_last_legal_candidates",
)


def _all_counter_feature_names() -> tuple[str, ...]:
    """Every numeric (``int``/``float``) field on ``DecodeStats``, in
    declaration order. Uses ``dataclasses.fields`` introspection rather than
    a hand-maintained list so it never drifts from the real dataclass; string
    fields (``compiler_lattice_last_signature``, ``solver_terminal_status``,
    ``compiler_lattice_termination_reason``) and list-of-trace fields
    (``*_traces``) are excluded since they are not scalar counters."""
    from slm_training.models.decode_stats import DecodeStats as _DecodeStats

    return tuple(
        field.name
        for field in dataclasses.fields(_DecodeStats)
        if field.type in ("int", "float")
    )


#: The "all counters" arm's feature set: every numeric DecodeStats counter.
#: Computed once at import time via introspection (see
#: ``_all_counter_feature_names``).
ALL_COUNTER_FEATURES: tuple[str, ...] = _all_counter_feature_names()


def extract_features(stats: "DecodeStats", *, feature_names: Sequence[str]) -> dict[str, float]:
    """Pull ``feature_names`` off a ``DecodeStats`` instance into a flat float dict."""
    return {name: float(getattr(stats, name)) for name in feature_names}


def score_from_features(features: Mapping[str, float], feature_names: Sequence[str]) -> float:
    """Fixed, documented, equal-weighted sum of ``features[name]`` over ``feature_names``.

    This is a fixture-scale deterministic proxy, not a trained model (same
    honesty framing as ``operator_hard_negative_training.score_action_view``):
    every named feature is used as-is (no learned weights, no normalization)
    because every ``DecodeStats`` counter this module names is already a
    "bad is high" counter (a false hard elimination, an invalid-over-valid
    selection, a dead end, a large last-legal-candidate count, and a large
    selector regret are all evidence *for* failure, never against it), so a
    plain sum is a well-defined, arm-comparable risk score. An empty
    ``feature_names`` (the ``context_free_baseline`` arm) always scores
    ``0.0`` for every row -- a constant, context-free score.
    """
    return sum(float(features.get(name, 0.0)) for name in feature_names)


def _rank_transform(values: Sequence[float]) -> list[float]:
    """Ascending 1-indexed ranks with ties resolved to their average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        average_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = average_rank
        i = j + 1
    return ranks


def auroc(scored: Sequence[tuple[float, bool]]) -> float | None:
    """Rank-based AUROC over ``(score, is_positive)`` pairs.

    Higher score means more likely positive (failure). Computed via the
    standard Mann-Whitney U / average-rank formula: ranks are assigned
    ascending over all scores with ties resolved to the average rank of the
    tied group (so a set of identical scores across both classes yields
    ``0.5``, not an arbitrary tie-break-dependent value); ``U = rank_sum of
    positives - n_pos*(n_pos+1)/2`` and ``AUROC = U / (n_pos * n_neg)``.
    Returns ``None`` when all labels are the same class (AUROC is undefined
    without both a positive and a negative example).
    """
    n = len(scored)
    if n == 0:
        return None
    positive_flags = [is_positive for _, is_positive in scored]
    n_pos = sum(1 for flag in positive_flags if flag)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    values = [score for score, _ in scored]
    ranks = _rank_transform(values)
    rank_sum_pos = sum(rank for rank, flag in zip(ranks, positive_flags) if flag)
    u_statistic = rank_sum_pos - n_pos * (n_pos + 1) / 2
    return u_statistic / (n_pos * n_neg)


def auprc(scored: Sequence[tuple[float, bool]]) -> float | None:
    """Precision-recall AUC over ``(score, is_positive)`` pairs.

    Sweeps a descending-score threshold, recomputing precision/recall at
    every row boundary (ties broken by input order -- a stable sort on
    descending score), and integrates the resulting precision-recall curve
    with plain trapezoidal integration over recall (including the
    conventional ``(recall=0, precision=1)`` starting point). Returns
    ``None`` when there are no positive examples (precision/recall are
    undefined without at least one).
    """
    total_positive = sum(1 for _, is_positive in scored if is_positive)
    if total_positive == 0:
        return None
    ordered = sorted(scored, key=lambda pair: -pair[0])
    points: list[tuple[float, float]] = [(0.0, 1.0)]
    true_positive = 0
    false_positive = 0
    for _, is_positive in ordered:
        if is_positive:
            true_positive += 1
        else:
            false_positive += 1
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / total_positive
        points.append((recall, precision))
    area = 0.0
    prev_recall, prev_precision = points[0]
    for recall, precision in points[1:]:
        area += (recall - prev_recall) * (precision + prev_precision) / 2.0
        prev_recall, prev_precision = recall, precision
    return area


def assert_group_disjoint_split(
    train_group_ids: Sequence[str], held_out_group_ids: Sequence[str]
) -> None:
    """Raise ``ValueError`` if any group id appears in both splits.

    A small, directly testable enforcement of "group split by request/target
    cluster and checkpoint to prevent trace leakage" -- a set-intersection
    check, not a real splitting algorithm.
    """
    overlap = set(train_group_ids) & set(held_out_group_ids)
    if overlap:
        raise ValueError(
            f"group ids appear in both train and held-out splits: {sorted(overlap)}"
        )


@dataclass(frozen=True)
class OperatorFailurePredictionArmResultV1:
    """One feature-arm's discrimination/calibration result over a fixed row set."""

    arm: str
    feature_names: tuple[str, ...]
    auroc: float | None
    auprc: float | None
    calibration_error: float | None
    n_examples: int
    n_positive: int
    schema: str = "operator_failure_prediction_arm_result/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "arm": self.arm,
            "feature_names": list(self.feature_names),
            "auroc": self.auroc,
            "auprc": self.auprc,
            "calibration_error": self.calibration_error,
            "n_examples": self.n_examples,
            "n_positive": self.n_positive,
        }


@dataclass(frozen=True)
class OperatorFailurePredictionReportV1:
    """Per-arm operator-decode failure-prediction report over a fixed row set."""

    arms: tuple[OperatorFailurePredictionArmResultV1, ...]
    best_arm_by_auroc: str
    schema: str = "operator_failure_prediction_report/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "arms": [arm.to_dict() for arm in self.arms],
            "best_arm_by_auroc": self.best_arm_by_auroc,
        }


def _confidences_of_failure(scores: Sequence[float]) -> list[float]:
    """Min-max normalize raw scores into ``[0, 1]`` "confidence of failure"
    probabilities for :func:`calibration_error`. When every score in the
    fixture is identical (no spread to normalize), every row gets the
    degenerate midpoint confidence ``0.5`` -- a well-defined "no signal"
    value, not an arbitrary one."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi <= lo:
        return [0.5 for _ in scores]
    return [(value - lo) / (hi - lo) for value in scores]


def evaluate_failure_prediction_arms(
    rows: Sequence[tuple[Mapping[str, float], ReplayOutcomeLabel]],
    arms: Mapping[str, Sequence[str]],
) -> OperatorFailurePredictionReportV1:
    """Score every named feature arm over the same fixed ``(features, label)`` rows.

    "Positive" (failure) is every label except ``ReplayOutcomeLabel.SAFE_DEFER``
    -- a credited safe defer is the one label in this taxonomy that is not a
    replay-grounded failure, so it is the negative class for this
    discrimination task. Ties in ``best_arm_by_auroc`` (including an arm
    whose AUROC is ``None``, treated as the lowest possible rank) are broken
    by arm name ascending, for determinism.
    """
    results: list[OperatorFailurePredictionArmResultV1] = []
    for arm_name, feature_names in arms.items():
        feature_names = tuple(feature_names)
        scored: list[tuple[float, bool]] = []
        for features, label in rows:
            score = score_from_features(features, feature_names)
            is_positive = label != ReplayOutcomeLabel.SAFE_DEFER
            scored.append((score, is_positive))
        auroc_value = auroc(scored)
        auprc_value = auprc(scored)
        confidences = _confidences_of_failure([score for score, _ in scored])
        correct = [is_positive for _, is_positive in scored]
        calibration_value = (
            calibration_error(confidences, correct) if confidences else None
        )
        results.append(
            OperatorFailurePredictionArmResultV1(
                arm=arm_name,
                feature_names=feature_names,
                auroc=auroc_value,
                auprc=auprc_value,
                calibration_error=calibration_value,
                n_examples=len(rows),
                n_positive=sum(1 for _, is_positive in scored if is_positive),
            )
        )
    ranked = sorted(
        results,
        key=lambda result: (
            -(result.auroc if result.auroc is not None else -1.0),
            result.arm,
        ),
    )
    best_arm = ranked[0].arm if ranked else ""
    return OperatorFailurePredictionReportV1(arms=tuple(results), best_arm_by_auroc=best_arm)
