"""DSH4-02: one-teacher ceiling over exact operator legal sets (SLM-387).

SLM-386 (DSH4-01, ``operator_decision_state.py``) exports certified
``OperatorDecisionStateTraceV1`` records binding a compiler state, its
DSH3-06 exact/bounded legal action set, and acceptance evidence. This module
answers the DSH4-02 decision question over those traces *before any
distillation work happens*: does a teacher rank accepted operator actions
better than frequency, deterministic compiler order, and the current scorer,
robustly enough (order/name-stable, positively verifier-correlated) to
justify knowledge distillation?

Honesty scoping (read before using any part of this module)
-------------------------------------------------------------
There is no live external LLM teacher wired into this repository or this
environment. ``SyntheticDescriptorTeacherV1`` below is a **deterministic,
non-learned stand-in** for a real teacher: it scores candidates from
structural features already present on ``LegalOperatorActionV1``
(``operator_id``, argument arity, ``proof_checks``) plus a value-identity
term derived from ``semantic_id`` -- never from opaque ids, presentation
order, natural-language descriptions, or ``current_scores`` /
``accepted_application_ids`` (which would leak the evaluator's own labels
back into the "teacher"). It plugs into ``OperatorTeacherAdapter``, the same
narrow interface a real prompted LLM teacher would implement, so swapping in
a live teacher later requires no harness change.

**Every result this module produces is fixture/wiring evidence
(``claim_class: wiring``), never a production KD-readiness claim.** A
production claim requires: (1) a real teacher behind
``OperatorTeacherAdapter``, (2) re-running this exact harness against it, and
(3) an honest go/no-go read of the stop rule below over that real output.
Never edit this module to make the synthetic teacher "win" -- the stop rule
must report whatever the harness actually finds, including a legitimate
defer/no-go.

Reused, not duplicated
-----------------------
* ``OperatorDecisionStateTraceV1`` / ``CaptureStage`` (DSH4-01,
  ``slm_training.harnesses.distill.operator_decision_state``) -- the
  certified state/legal-set/acceptance export this harness consumes. This
  module never captures or replays traces itself.
* ``OperatorLegalSetV1`` / ``LegalOperatorActionV1`` / ``LegalSetCoverage``
  (DSH3-06, ``slm_training.dsl.operators.legal_set``) -- the exact/bounded
  legal action identity, compiler order, and coverage flag every comparator
  below ranks over.
* ``SupervisionSource`` (EFS3-01, ``slm_training.evals.solver_state_supervision``)
  -- gold-vs-on-policy state provenance used for the decision-family slices.

A scoping note on what is genuinely inference-visible here: DSH3-06's
``LegalOperatorActionV1`` does **not** carry the full ``ActionEffectV1`` for
unapplied candidates -- only opaque digests (``proof_fingerprint``,
``semantic_id``). A teacher with true ``ActionEffectV1``-level visibility
would need a live dry-run/replay per candidate, which is out of scope for
this fixture-only wiring slice; the synthetic teacher here is bounded by the
same digest-only visibility DSH3-06 already exports.

Stop rule (issue SLM-387): if the teacher is not better than the required
baselines with paired confidence evidence, or is order/name fragile beyond
the declared bound, or its scores do not positively correlate with
verifier-backed (accepted) outcomes -- do not distill it.
``evaluate_stop_rule`` implements this as a gate function, and
``compare_operator_teacher_ceiling`` reports its honest outcome (go or
no-go/defer) alongside every metric that fed it.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from slm_training.dsl.operators.legal_set import LegalOperatorActionV1, LegalSetCoverage
from slm_training.evals.solver_state_supervision import SupervisionSource
from slm_training.harnesses.distill.operator_decision_state import (
    OperatorDecisionStateTraceV1,
)

__all__ = [
    "DECLARED_PERTURBATION_BOUND",
    "DEFAULT_TOP_K",
    "PRIMARY_METRIC",
    "REQUIRED_BASELINE_IDS",
    "CompilerOrderBaselineV1",
    "CurrentScorerBaselineV1",
    "DecisionFamily",
    "DescriptorSimilarityBaselineV1",
    "FrequencyBaselineV1",
    "FrequencyTableV1",
    "OperatorTeacherAdapter",
    "OperatorTeacherCeilingReportV1",
    "OracleAcceptedSetComparatorV1",
    "PairedComparisonV1",
    "PerturbationKind",
    "PerturbationRobustnessReportV1",
    "RankingComparator",
    "RankingReliability",
    "RankingV1",
    "StopRuleVerdictV1",
    "SyntheticDescriptorTeacherV1",
    "TeacherQueryV1",
    "accepted_set_mass",
    "build_frequency_table",
    "classify_decision_family",
    "compare_operator_teacher_ceiling",
    "default_teacher_query",
    "descriptor_similarity",
    "evaluate_stop_rule",
    "expected_calibration_error",
    "ndcg_at_k",
    "pearson_correlation",
    "ranking_distance",
    "reciprocal_rank",
    "run_perturbation_robustness",
    "selective_risk_aurc",
    "spearman_correlation",
    "top_k_recall",
    "write_operator_teacher_ceiling_report",
]

_SCHEMA_VERSION = "dsh4-02.v1"

#: Rank-based metric used to gate the stop rule; it is defined for every
#: comparator (unlike ``accepted_set_mass``, which requires probabilities),
#: so it is the one preregistered confirmatory endpoint. Every other metric
#: is exploratory/reported, mirroring this repo's locked-confirmatory-vs-
#: exploratory campaign convention (AGENTS.md "Preregistered campaign law").
PRIMARY_METRIC = "mrr"

DEFAULT_TOP_K = 3

#: Baselines the teacher must significantly beat (paired, on ``PRIMARY_METRIC``)
#: before the stop rule can return ``go``.
REQUIRED_BASELINE_IDS: tuple[str, ...] = (
    "baseline:frequency",
    "baseline:compiler_order",
    "baseline:current_scorer",
    "baseline:descriptor_similarity",
)

#: Declared bound on ranking-distance drift under candidate-order, opaque-id,
#: description-length, and prompt-template perturbation (SLM-387
#: acceptance: "ID/order perturbation stays within the declared bound").
#: ``SyntheticDescriptorTeacherV1`` never reads any of the perturbed fields,
#: so a well-formed teacher run should measure exactly ``0.0``; the bound is
#: kept as an explicit, non-zero-by-necessity constant so a future teacher
#: with legitimate (bounded) sensitivity has somewhere to declare it.
DECLARED_PERTURBATION_BOUND = 0.0


# --------------------------------------------------------------------------- #
# Decision families (preregistered)
# --------------------------------------------------------------------------- #


class DecisionFamily(str, Enum):
    """Preregistered supervision-source x legal-set-size slices.

    Fixed in code (not derived post hoc from results) so every comparison
    below is "on preregistered decision families" per the SLM-387 acceptance
    bar, not a slice chosen after seeing which one looks best.
    """

    GOLD_EMPTY = "gold_empty"
    GOLD_SINGLETON = "gold_singleton"
    GOLD_SMALL = "gold_small"
    GOLD_LARGE = "gold_large"
    ON_POLICY_EMPTY = "on_policy_empty"
    ON_POLICY_SINGLETON = "on_policy_singleton"
    ON_POLICY_SMALL = "on_policy_small"
    ON_POLICY_LARGE = "on_policy_large"


def classify_decision_family(trace: OperatorDecisionStateTraceV1) -> DecisionFamily:
    """Classify one trace into its preregistered decision family."""
    n = len(trace.legal_set.operator_actions)
    prefix = "gold" if trace.supervision_source is SupervisionSource.GOLD else "on_policy"
    if n == 0:
        bucket = "empty"
    elif n == 1:
        bucket = "singleton"
    elif n <= 4:
        bucket = "small"
    else:
        bucket = "large"
    return DecisionFamily(f"{prefix}_{bucket}")


# --------------------------------------------------------------------------- #
# Ranking contract shared by the teacher and every baseline
# --------------------------------------------------------------------------- #


class RankingReliability(str, Enum):
    """Exact-vs-approximate provenance for one ranking (never silently dropped)."""

    EXACT = "exact"
    APPROXIMATE = "approximate"


def _trace_reliability(trace: OperatorDecisionStateTraceV1) -> tuple[RankingReliability, bool, str]:
    """Reliability bound shared by every comparator: a ranking can only be as
    exact as the DSH3-06 legal-set enumeration it ranks over."""
    if trace.approximate or trace.coverage is LegalSetCoverage.PARTIAL:
        return (
            RankingReliability.APPROXIMATE,
            True,
            "decision_state_trace_marked_approximate_or_partial_legal_set_coverage",
        )
    return RankingReliability.EXACT, False, "complete_legal_set_coverage"


@dataclass(frozen=True)
class RankingV1:
    """One comparator's ranking of ``trace``'s legal operator actions.

    ``application_order`` is best-first. ``probabilities`` (when present) is
    aligned to ``application_order`` and sums to ~1.0; comparators that
    cannot honestly produce a probability distribution (e.g. pure ordinal
    compiler order) leave it ``None`` rather than fabricate one -- metrics
    that need probabilities (``accepted_set_mass``, calibration, selective
    risk) then honestly report "not applicable" for that comparator instead
    of a fake number.
    """

    trace_id: str
    source_id: str
    application_order: tuple[str, ...]
    probabilities: tuple[float, ...] | None
    reliability: RankingReliability
    reliability_reason: str
    approximate: bool
    schema: str = "operator_teacher_ranking/v1"

    def __post_init__(self) -> None:
        if len(set(self.application_order)) != len(self.application_order):
            raise ValueError("ranking application_order must not repeat an id")
        if not self.application_order:
            if self.probabilities:
                raise ValueError("an empty ranking cannot carry probabilities")
            return
        if self.probabilities is None:
            return
        if len(self.probabilities) != len(self.application_order):
            raise ValueError("ranking probabilities must align with application_order")
        if any(p < 0.0 for p in self.probabilities):
            raise ValueError("ranking probabilities must be non-negative")
        total = sum(self.probabilities)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"ranking probabilities must sum to 1.0 (got {total!r})")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "trace_id": self.trace_id,
            "source_id": self.source_id,
            "application_order": list(self.application_order),
            "probabilities": (
                list(self.probabilities) if self.probabilities is not None else None
            ),
            "reliability": self.reliability.value,
            "reliability_reason": self.reliability_reason,
            "approximate": self.approximate,
        }


def _empty_ranking(
    trace: OperatorDecisionStateTraceV1, source_id: str, reason: str
) -> RankingV1:
    reliability, approximate, coverage_reason = _trace_reliability(trace)
    return RankingV1(
        trace_id=trace.trace_id,
        source_id=source_id,
        application_order=(),
        probabilities=None,
        reliability=reliability,
        reliability_reason=f"{reason}; {coverage_reason}",
        approximate=approximate,
    )


# --------------------------------------------------------------------------- #
# Teacher query envelope + perturbation harness
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TeacherQueryV1:
    """The exact payload handed to an ``OperatorTeacherAdapter``.

    A real prompted LLM teacher would render ``candidate_order``,
    ``opaque_id_labels``, ``description_lengths``, and
    ``prompt_template_hash`` into a prompt -- exactly the surface SLM-387
    requires perturbation-robustness tests over. None of these fields is
    scoring authority (AGENTS.md engineering norms): a compliant teacher's
    *ranking* must not depend on them, only its *presentation* may.
    """

    trace: OperatorDecisionStateTraceV1
    candidate_order: tuple[str, ...]
    opaque_id_labels: Mapping[str, str]
    description_lengths: Mapping[str, int]
    prompt_template_hash: str


def default_teacher_query(
    trace: OperatorDecisionStateTraceV1, *, prompt_template_hash: str = "canonical"
) -> TeacherQueryV1:
    """Build the canonical (unperturbed) query for ``trace``."""
    ids = tuple(action.application_id for action in trace.legal_set.operator_actions)
    return TeacherQueryV1(
        trace=trace,
        candidate_order=ids,
        opaque_id_labels={aid: f"candidate-{i}" for i, aid in enumerate(ids)},
        description_lengths={aid: 0 for aid in ids},
        prompt_template_hash=prompt_template_hash,
    )


class PerturbationKind(str, Enum):
    CANDIDATE_ORDER = "candidate_order"
    OPAQUE_ID_RELABEL = "opaque_id_relabel"
    DESCRIPTION_LENGTH = "description_length"
    PROMPT_TEMPLATE = "prompt_template"


def perturb_candidate_order(query: TeacherQueryV1, seed: int) -> TeacherQueryV1:
    ids = list(query.candidate_order)
    random.Random(seed).shuffle(ids)
    return replace(query, candidate_order=tuple(ids))


def perturb_opaque_id_labels(query: TeacherQueryV1, seed: int) -> TeacherQueryV1:
    rng = random.Random(seed)
    labels = {aid: f"relabeled-{rng.randrange(10**6):06d}" for aid in query.candidate_order}
    return replace(query, opaque_id_labels=labels)


def perturb_description_length(query: TeacherQueryV1, seed: int) -> TeacherQueryV1:
    rng = random.Random(seed)
    lengths = {aid: rng.randrange(0, 400) for aid in query.candidate_order}
    return replace(query, description_lengths=lengths)


def perturb_prompt_template(query: TeacherQueryV1, seed: int) -> TeacherQueryV1:
    return replace(
        query,
        prompt_template_hash=hashlib.sha256(f"perturbed-template-{seed}".encode()).hexdigest(),
    )


_PERTURBATION_FNS: dict[PerturbationKind, Callable[[TeacherQueryV1, int], TeacherQueryV1]] = {
    PerturbationKind.CANDIDATE_ORDER: perturb_candidate_order,
    PerturbationKind.OPAQUE_ID_RELABEL: perturb_opaque_id_labels,
    PerturbationKind.DESCRIPTION_LENGTH: perturb_description_length,
    PerturbationKind.PROMPT_TEMPLATE: perturb_prompt_template,
}


def ranking_distance(a: RankingV1, b: RankingV1) -> float:
    """Fraction of candidate pairs whose relative order disagrees between ``a`` and ``b``.

    0.0 means identical order; 1.0 means every pair is inverted. Both
    rankings must rank the identical candidate set (perturbation never
    changes *which* candidates are legal, only how they are presented).
    """
    if set(a.application_order) != set(b.application_order):
        raise ValueError("rankings must be over the identical candidate set")
    ids = list(a.application_order)
    n = len(ids)
    if n < 2:
        return 0.0
    pos_a = {aid: i for i, aid in enumerate(a.application_order)}
    pos_b = {aid: i for i, aid in enumerate(b.application_order)}
    total = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1
            if (pos_a[ids[i]] - pos_a[ids[j]]) * (pos_b[ids[i]] - pos_b[ids[j]]) < 0:
                discordant += 1
    return discordant / total if total else 0.0


@dataclass(frozen=True)
class PerturbationRobustnessReportV1:
    kind_distances: Mapping[str, float]
    bound: float
    within_bound: bool
    n_traces_compared: Mapping[str, int]
    schema: str = "operator_teacher_perturbation_report/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind_distances": dict(self.kind_distances),
            "bound": self.bound,
            "within_bound": self.within_bound,
            "n_traces_compared": dict(self.n_traces_compared),
        }


def run_perturbation_robustness(
    teacher: "OperatorTeacherAdapter",
    traces: Sequence[OperatorDecisionStateTraceV1],
    *,
    bound: float = DECLARED_PERTURBATION_BOUND,
    seed: int = 0,
) -> PerturbationRobustnessReportV1:
    """Run every declared perturbation kind against ``teacher`` over ``traces``.

    Traces with fewer than two legal actions are skipped (ranking distance is
    undefined/trivially zero for 0 or 1 candidates).
    """
    kind_distances: dict[str, float] = {}
    n_compared: dict[str, int] = {}
    for kind, fn in _PERTURBATION_FNS.items():
        max_distance = 0.0
        compared = 0
        for i, trace in enumerate(traces):
            if len(trace.legal_set.operator_actions) < 2:
                continue
            canonical_query = default_teacher_query(trace)
            canonical = teacher.rank(canonical_query)
            perturbed = teacher.rank(fn(canonical_query, seed + i))
            if canonical is None or perturbed is None:
                continue
            max_distance = max(max_distance, ranking_distance(canonical, perturbed))
            compared += 1
        kind_distances[kind.value] = max_distance
        n_compared[kind.value] = compared
    overall = max(kind_distances.values()) if kind_distances else 0.0
    return PerturbationRobustnessReportV1(
        kind_distances=kind_distances,
        bound=bound,
        within_bound=overall <= bound,
        n_traces_compared=n_compared,
    )


# --------------------------------------------------------------------------- #
# Teacher adapter interface + the synthetic stand-in teacher
# --------------------------------------------------------------------------- #


class OperatorTeacherAdapter(Protocol):
    """Interface any teacher -- synthetic today, a real LLM later -- implements."""

    source_id: str

    def rank(self, query: TeacherQueryV1) -> RankingV1 | None: ...


def _descriptor_feature_vector(action: LegalOperatorActionV1) -> tuple[float, ...]:
    """Structural features genuinely available on an unapplied legal action.

    Deliberately excludes ``serialized`` / opaque ids / arguments' opaque
    refs (realization data, never scoring authority per AGENTS.md
    engineering norms) and excludes ``current_scores`` / accepted-set
    membership (evaluator-only labels the teacher must stay blind to).
    """
    operator_hash = int(hashlib.sha256(action.operator_id.encode()).hexdigest()[:8], 16)
    return (
        float(len(action.proof_checks)),
        float(len(action.arguments)),
        operator_hash / 0xFFFFFFFF,
    )


def descriptor_similarity(
    action: LegalOperatorActionV1, all_actions: Sequence[LegalOperatorActionV1]
) -> float:
    """Simple non-learned "typicality" heuristic: closeness to the legal
    set's own feature centroid. In (0, 1]; 1.0 means dead-on centroid."""
    vectors = [_descriptor_feature_vector(a) for a in all_actions]
    dims = len(vectors[0])
    centroid = tuple(sum(v[d] for v in vectors) / len(vectors) for d in range(dims))
    v = _descriptor_feature_vector(action)
    distance = math.sqrt(sum((v[d] - centroid[d]) ** 2 for d in range(dims)))
    return 1.0 / (1.0 + distance)


def _deterministic_noise(action: LegalOperatorActionV1, salt: str) -> float:
    """Deterministic pseudo-random term in [0, 1) seeded by ``semantic_id``.

    ``semantic_id`` is a content-identity digest (operator fingerprint +
    each bound value's descriptor semantic fingerprint) -- it does not
    depend on opaque ids, request ids, or presentation order, so this term
    is architecturally invariant under the opaque-id/candidate-order
    perturbations SLM-387 requires testing.
    """
    digest = hashlib.sha256(f"{salt}:{action.semantic_id}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


@dataclass(frozen=True)
class SyntheticDescriptorTeacherV1:
    """Deterministic stand-in teacher (SLM-387 synthetic wiring evidence only).

    **This is not a real teacher.** It scores each candidate as a weighted
    mix of ``descriptor_similarity`` (structural typicality) and a
    deterministic ``semantic_id``-seeded noise term, then returns exact
    softmax logits when the legal set was completely enumerated (DSH3-06
    ``LegalSetCoverage.COMPLETE``) or an explicitly reliability-marked
    approximate preference otherwise (budget-truncated ``PARTIAL``
    coverage). It never reads ``query.candidate_order``,
    ``query.opaque_id_labels``, ``query.description_lengths``,
    ``query.prompt_template_hash``, ``trace.current_scores``, or
    ``trace.accepted_application_ids`` -- by construction it is blind to
    presentation metadata, the current scorer, and gold/evaluator-only
    labels. A production KD-readiness claim requires replacing this class
    with a real LLM teacher behind the same ``OperatorTeacherAdapter``
    interface and re-running the full harness.
    """

    source_id: str = "teacher:synthetic_descriptor_v1"
    noise_weight: float = 0.15
    salt: str = "dsh4-02-synthetic-teacher-v1"

    def rank(self, query: TeacherQueryV1) -> RankingV1 | None:
        trace = query.trace
        actions = trace.legal_set.operator_actions
        if not actions:
            return _empty_ranking(trace, self.source_id, "empty_legal_set")

        raw_scores = []
        for action in actions:
            similarity = descriptor_similarity(action, actions)
            noise = _deterministic_noise(action, self.salt)
            raw_scores.append((1.0 - self.noise_weight) * similarity + self.noise_weight * noise)

        peak = max(raw_scores)
        exps = [math.exp(score - peak) for score in raw_scores]
        total = sum(exps)
        probs = [value / total for value in exps]

        order = sorted(
            range(len(actions)),
            key=lambda i: (-probs[i], actions[i].application_id),
        )
        ids = tuple(actions[i].application_id for i in order)
        ordered_probs = tuple(probs[i] for i in order)

        exact = trace.coverage is LegalSetCoverage.COMPLETE
        reliability, trace_approximate, coverage_reason = _trace_reliability(trace)
        reason = (
            "exact_finite_set_logits_over_complete_legal_set"
            if exact
            else "provenance_marked_approximate_preference_over_partial_legal_set"
        )
        return RankingV1(
            trace_id=trace.trace_id,
            source_id=self.source_id,
            application_order=ids,
            probabilities=ordered_probs,
            reliability=reliability,
            reliability_reason=f"{reason}; {coverage_reason}",
            approximate=trace_approximate,
        )


# --------------------------------------------------------------------------- #
# Baseline comparators
# --------------------------------------------------------------------------- #


class RankingComparator(Protocol):
    source_id: str

    def rank(self, trace: OperatorDecisionStateTraceV1) -> RankingV1 | None: ...


@dataclass(frozen=True)
class CompilerOrderBaselineV1:
    """Deterministic DSH3-06 enumeration order (no learned or scored signal)."""

    source_id: str = "baseline:compiler_order"

    def rank(self, trace: OperatorDecisionStateTraceV1) -> RankingV1 | None:
        actions = trace.legal_set.operator_actions
        if not actions:
            return _empty_ranking(trace, self.source_id, "empty_legal_set")
        reliability, approximate, coverage_reason = _trace_reliability(trace)
        return RankingV1(
            trace_id=trace.trace_id,
            source_id=self.source_id,
            application_order=tuple(a.application_id for a in actions),
            probabilities=None,
            reliability=reliability,
            reliability_reason=f"deterministic_compiler_enumeration_order; {coverage_reason}",
            approximate=approximate,
        )


@dataclass(frozen=True)
class CurrentScorerBaselineV1:
    """Ranks by ``trace.current_scores`` -- the model/scorer already in production."""

    source_id: str = "baseline:current_scorer"

    def rank(self, trace: OperatorDecisionStateTraceV1) -> RankingV1 | None:
        actions = trace.legal_set.operator_actions
        if not actions:
            return _empty_ranking(trace, self.source_id, "empty_legal_set")
        scores = trace.current_scores
        if not scores:
            return None
        known = [scores[a.application_id] for a in actions if a.application_id in scores]
        if not known:
            return None
        floor = min(known) - 1.0
        raw = [scores.get(a.application_id, floor) for a in actions]
        peak = max(raw)
        exps = [math.exp(v - peak) for v in raw]
        total = sum(exps)
        probs = [v / total for v in exps]
        order = sorted(
            range(len(actions)), key=lambda i: (-probs[i], actions[i].application_id)
        )
        reliability, approximate, coverage_reason = _trace_reliability(trace)
        partial_scores = len(known) < len(actions)
        reason = (
            "current_scores_partial_missing_candidates_ranked_last"
            if partial_scores
            else "current_scores_softmax"
        )
        return RankingV1(
            trace_id=trace.trace_id,
            source_id=self.source_id,
            application_order=tuple(actions[i].application_id for i in order),
            probabilities=tuple(probs[i] for i in order),
            reliability=(RankingReliability.APPROXIMATE if partial_scores else reliability),
            reliability_reason=f"{reason}; {coverage_reason}",
            approximate=approximate or partial_scores,
        )


@dataclass(frozen=True)
class DescriptorSimilarityBaselineV1:
    """Pure structural-typicality ranker: same features as the synthetic
    teacher, without its noise term. A simple, non-learned baseline."""

    source_id: str = "baseline:descriptor_similarity"

    def rank(self, trace: OperatorDecisionStateTraceV1) -> RankingV1 | None:
        actions = trace.legal_set.operator_actions
        if not actions:
            return _empty_ranking(trace, self.source_id, "empty_legal_set")
        sims = [descriptor_similarity(a, actions) for a in actions]
        total = sum(sims)
        probs = [s / total for s in sims]
        order = sorted(
            range(len(actions)), key=lambda i: (-probs[i], actions[i].application_id)
        )
        reliability, approximate, coverage_reason = _trace_reliability(trace)
        return RankingV1(
            trace_id=trace.trace_id,
            source_id=self.source_id,
            application_order=tuple(actions[i].application_id for i in order),
            probabilities=tuple(probs[i] for i in order),
            reliability=reliability,
            reliability_reason=f"structural_descriptor_centroid_similarity; {coverage_reason}",
            approximate=approximate,
        )


@dataclass(frozen=True)
class OracleAcceptedSetComparatorV1:
    """Upper-bound reference: perfect ranking of the accepted set. Never a
    baseline the teacher must "beat" -- it exists to normalize/contextualize
    the achievable ceiling, not as a fair comparator."""

    source_id: str = "oracle:accepted_set"

    def rank(self, trace: OperatorDecisionStateTraceV1) -> RankingV1 | None:
        actions = trace.legal_set.operator_actions
        if not actions:
            return _empty_ranking(trace, self.source_id, "empty_legal_set")
        accepted = set(trace.accepted_application_ids)
        order = sorted(
            actions,
            key=lambda a: (0 if a.application_id in accepted else 1, a.application_id),
        )
        reliability, approximate, coverage_reason = _trace_reliability(trace)
        if not accepted:
            return RankingV1(
                trace_id=trace.trace_id,
                source_id=self.source_id,
                application_order=tuple(a.application_id for a in order),
                probabilities=None,
                reliability=RankingReliability.APPROXIMATE,
                reliability_reason=f"no_accepted_actions_to_oracle; {coverage_reason}",
                approximate=True,
            )
        probs = [
            (1.0 / len(accepted)) if a.application_id in accepted else 0.0 for a in order
        ]
        return RankingV1(
            trace_id=trace.trace_id,
            source_id=self.source_id,
            application_order=tuple(a.application_id for a in order),
            probabilities=tuple(probs),
            reliability=reliability,
            reliability_reason=f"oracle_perfect_ranking_of_accepted_set; {coverage_reason}",
            approximate=approximate,
        )


def _frequency_key(action: LegalOperatorActionV1) -> str:
    """Content-identity key for the frequency prior.

    ``semantic_id`` already folds ``operator_fingerprint`` with each bound
    argument's descriptor semantic fingerprint (DSH3-06
    ``_semantic_action_id``) -- it is stable across states/opaque-ref
    relabeling whenever the same conceptual candidate recurs, which is
    exactly the identity a "how often was this chosen historically" prior
    needs.
    """
    return action.semantic_id


@dataclass(frozen=True)
class FrequencyTableV1:
    counts: Mapping[str, int]
    total: int
    schema: str = "operator_frequency_table/v1"

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "counts": dict(self.counts), "total": self.total}


def build_frequency_table(
    training_traces: Sequence[OperatorDecisionStateTraceV1],
) -> FrequencyTableV1:
    """Count accepted-action frequency by content identity over a *training*
    trace pool. Callers must keep this pool disjoint from the evaluation
    pool (``compare_operator_teacher_ceiling`` enforces this)."""
    counts: dict[str, int] = {}
    for trace in training_traces:
        accepted = set(trace.accepted_application_ids)
        if not accepted:
            continue
        by_id = {a.application_id: a for a in trace.legal_set.operator_actions}
        for application_id in accepted:
            action = by_id.get(application_id)
            if action is None:
                continue
            key = _frequency_key(action)
            counts[key] = counts.get(key, 0) + 1
    return FrequencyTableV1(counts=counts, total=sum(counts.values()))


@dataclass(frozen=True)
class FrequencyBaselineV1:
    table: FrequencyTableV1
    source_id: str = "baseline:frequency"

    def rank(self, trace: OperatorDecisionStateTraceV1) -> RankingV1 | None:
        actions = trace.legal_set.operator_actions
        if not actions:
            return _empty_ranking(trace, self.source_id, "empty_legal_set")
        # Laplace(+1) smoothing so an unseen candidate is possible, never zero.
        smoothed = [(a, self.table.counts.get(_frequency_key(a), 0) + 1) for a in actions]
        smoothed.sort(key=lambda pair: (-pair[1], pair[0].application_id))
        total = sum(count for _, count in smoothed)
        reliability, approximate, coverage_reason = _trace_reliability(trace)
        return RankingV1(
            trace_id=trace.trace_id,
            source_id=self.source_id,
            application_order=tuple(a.application_id for a, _ in smoothed),
            probabilities=tuple(count / total for _, count in smoothed),
            reliability=reliability,
            reliability_reason=(
                f"laplace_smoothed_historical_acceptance_frequency"
                f"(n_training_examples={self.table.total}); {coverage_reason}"
            ),
            approximate=approximate,
        )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def reciprocal_rank(ranking: RankingV1, accepted_ids: frozenset[str]) -> float | None:
    if not accepted_ids:
        return None
    for i, application_id in enumerate(ranking.application_order, start=1):
        if application_id in accepted_ids:
            return 1.0 / i
    return 0.0


def top_k_recall(ranking: RankingV1, accepted_ids: frozenset[str], k: int) -> float | None:
    if not accepted_ids:
        return None
    top = set(ranking.application_order[:k])
    return len(top & accepted_ids) / len(accepted_ids)


def ndcg_at_k(ranking: RankingV1, accepted_ids: frozenset[str], k: int) -> float | None:
    if not accepted_ids:
        return None
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, application_id in enumerate(ranking.application_order[:k], start=1)
        if application_id in accepted_ids
    )
    ideal_hits = min(len(accepted_ids), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else None


def accepted_set_mass(ranking: RankingV1, accepted_ids: frozenset[str]) -> float | None:
    if ranking.probabilities is None:
        return None
    return sum(
        p
        for application_id, p in zip(ranking.application_order, ranking.probabilities)
        if application_id in accepted_ids
    )


def expected_calibration_error(
    rows: Sequence[tuple[float, bool]], *, n_bins: int = 10
) -> float | None:
    """Standard binned ECE over (predicted probability, was-accepted) pairs."""
    if not rows:
        return None
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for probability, correct in rows:
        idx = min(int(probability * n_bins), n_bins - 1)
        bins[idx].append((probability, correct))
    total = len(rows)
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        confidence = sum(p for p, _ in bucket) / len(bucket)
        accuracy = sum(1 for _, c in bucket if c) / len(bucket)
        ece += (len(bucket) / total) * abs(confidence - accuracy)
    return ece


def selective_risk_aurc(rows: Sequence[tuple[float, bool]]) -> float | None:
    """Area under the risk-coverage curve for top-1 selections, ordered by
    descending confidence. Lower is better (a well-calibrated, discriminative
    ranker keeps risk low even at high coverage)."""
    if not rows:
        return None
    ordered = sorted(rows, key=lambda row: -row[0])
    n = len(ordered)
    wrong = 0
    risks = []
    for i, (_, correct) in enumerate(ordered, start=1):
        if not correct:
            wrong += 1
        risks.append(wrong / i)
    return sum(risks) / n


def _rank_transform(values: Sequence[float]) -> list[float]:
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


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0.0 or var_y == 0.0:
        return None
    return covariance / math.sqrt(var_x * var_y)


def spearman_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson_correlation(_rank_transform(xs), _rank_transform(ys))


def _pooled_probability_rows(
    per_trace: Mapping[str, RankingV1 | None],
    accepted_by_trace: Mapping[str, frozenset[str]],
    *,
    top1_only: bool,
) -> list[tuple[float, bool]]:
    rows: list[tuple[float, bool]] = []
    for trace_id, ranking in per_trace.items():
        if ranking is None or ranking.probabilities is None or not ranking.application_order:
            continue
        accepted = accepted_by_trace[trace_id]
        if top1_only:
            rows.append((ranking.probabilities[0], ranking.application_order[0] in accepted))
        else:
            for application_id, probability in zip(
                ranking.application_order, ranking.probabilities
            ):
                rows.append((probability, application_id in accepted))
    return rows


def _metric_specs(
    top_k: int,
) -> dict[str, Callable[[RankingV1, frozenset[str]], float | None]]:
    return {
        "mrr": reciprocal_rank,
        "accepted_set_mass": accepted_set_mass,
        f"top_{top_k}_recall": lambda r, a: top_k_recall(r, a, top_k),
        f"ndcg_at_{top_k}": lambda r, a: ndcg_at_k(r, a, top_k),
    }


# --------------------------------------------------------------------------- #
# Paired comparison + stop rule
# --------------------------------------------------------------------------- #


def _bootstrap_ci(
    diffs: Sequence[float], *, n_boot: int, seed: int, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Paired bootstrap over per-trace metric differences. Returns
    ``(mean_diff, ci_low, ci_high)``; an empty ``diffs`` returns all zeros
    (the caller reports ``n_paired == 0`` alongside so this is never
    mistaken for "no difference measured with confidence")."""
    if not diffs:
        return 0.0, 0.0, 0.0
    n = len(diffs)
    if n == 1:
        return diffs[0], diffs[0], diffs[0]
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return sum(diffs) / n, lo, hi


@dataclass(frozen=True)
class PairedComparisonV1:
    """Paired teacher-vs-baseline comparison on one metric (overall or one
    preregistered decision family)."""

    metric_name: str
    family: DecisionFamily | None
    baseline_source_id: str
    teacher_source_id: str
    n_paired: int
    mean_diff: float
    ci_low: float
    ci_high: float
    win_rate: float
    schema: str = "operator_teacher_paired_comparison/v1"

    @property
    def teacher_significantly_better(self) -> bool:
        return self.n_paired > 0 and self.ci_low > 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "metric_name": self.metric_name,
            "family": self.family.value if self.family is not None else None,
            "baseline_source_id": self.baseline_source_id,
            "teacher_source_id": self.teacher_source_id,
            "n_paired": self.n_paired,
            "mean_diff": self.mean_diff,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "win_rate": self.win_rate,
            "teacher_significantly_better": self.teacher_significantly_better,
        }


def _paired_comparison(
    *,
    metric_name: str,
    family: DecisionFamily | None,
    baseline_source_id: str,
    teacher_source_id: str,
    diffs: Sequence[float],
    n_boot: int,
    seed: int,
) -> PairedComparisonV1:
    mean_diff, lo, hi = _bootstrap_ci(diffs, n_boot=n_boot, seed=seed)
    win_rate = (sum(1 for d in diffs if d >= 0.0) / len(diffs)) if diffs else 0.0
    return PairedComparisonV1(
        metric_name=metric_name,
        family=family,
        baseline_source_id=baseline_source_id,
        teacher_source_id=teacher_source_id,
        n_paired=len(diffs),
        mean_diff=mean_diff,
        ci_low=lo,
        ci_high=hi,
        win_rate=win_rate,
    )


@dataclass(frozen=True)
class StopRuleVerdictV1:
    """Honest go/no-go read of the SLM-387 stop rule.

    ``go=True`` means: the teacher significantly beat every required
    baseline (paired, ``primary_metric``, overall), perturbation robustness
    stayed within the declared bound, and teacher scores positively
    correlate with verifier-backed (accepted) outcomes. Any single failure
    forces ``go=False`` with the reason recorded -- this function never
    partially passes.
    """

    go: bool
    reasons: tuple[str, ...]
    beats_required_baselines: bool
    failing_baselines: tuple[str, ...]
    insufficient_data_baselines: tuple[str, ...]
    perturbation_within_bound: bool
    verifier_regret_correlation: float | None
    correlation_positive: bool
    primary_metric: str
    schema: str = "operator_teacher_stop_rule_verdict/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "go": self.go,
            "recommendation": "go" if self.go else "no_go_defer",
            "reasons": list(self.reasons),
            "beats_required_baselines": self.beats_required_baselines,
            "failing_baselines": list(self.failing_baselines),
            "insufficient_data_baselines": list(self.insufficient_data_baselines),
            "perturbation_within_bound": self.perturbation_within_bound,
            "verifier_regret_correlation": self.verifier_regret_correlation,
            "correlation_positive": self.correlation_positive,
            "primary_metric": self.primary_metric,
        }


def evaluate_stop_rule(
    *,
    paired_comparisons: Sequence[PairedComparisonV1],
    required_baseline_ids: Sequence[str] = REQUIRED_BASELINE_IDS,
    perturbation_report: PerturbationRobustnessReportV1,
    verifier_regret_correlation: float | None,
    min_correlation: float = 0.0,
    primary_metric: str = PRIMARY_METRIC,
) -> StopRuleVerdictV1:
    """Evaluate the SLM-387 stop rule from already-computed evidence.

    Pure function over typed evidence objects (no trace/teacher machinery
    required) so gate logic is directly unit-testable with hand-built
    ``PairedComparisonV1`` / ``PerturbationRobustnessReportV1`` fixtures.
    """
    reasons: list[str] = []
    failing: list[str] = []
    insufficient: list[str] = []

    for baseline_id in required_baseline_ids:
        matches = [
            c
            for c in paired_comparisons
            if c.baseline_source_id == baseline_id
            and c.family is None
            and c.metric_name == primary_metric
        ]
        if not matches or all(c.n_paired == 0 for c in matches):
            insufficient.append(baseline_id)
            failing.append(baseline_id)
            continue
        if not any(c.teacher_significantly_better for c in matches):
            failing.append(baseline_id)

    beats = not failing
    if failing:
        reasons.append(
            f"teacher does not significantly beat required baselines on "
            f"{primary_metric!r} (paired CI lower bound <= 0): {sorted(failing)}"
        )

    if not perturbation_report.within_bound:
        reasons.append(
            "perturbation robustness "
            f"{dict(perturbation_report.kind_distances)} exceeds the declared "
            f"bound ({perturbation_report.bound})"
        )

    correlation_positive = (
        verifier_regret_correlation is not None
        and verifier_regret_correlation > min_correlation
    )
    if not correlation_positive:
        reasons.append(
            "teacher scores do not positively correlate with verifier-backed "
            f"(accepted) outcomes (correlation={verifier_regret_correlation!r})"
        )

    go = beats and perturbation_report.within_bound and correlation_positive
    return StopRuleVerdictV1(
        go=go,
        reasons=tuple(reasons),
        beats_required_baselines=beats,
        failing_baselines=tuple(sorted(failing)),
        insufficient_data_baselines=tuple(sorted(insufficient)),
        perturbation_within_bound=perturbation_report.within_bound,
        verifier_regret_correlation=verifier_regret_correlation,
        correlation_positive=correlation_positive,
        primary_metric=primary_metric,
    )


# --------------------------------------------------------------------------- #
# Top-level orchestration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OperatorTeacherCeilingReportV1:
    schema: str
    teacher_source_id: str
    n_eval_traces: int
    n_frequency_training_traces: int
    top_k: int
    metrics_overall: Mapping[str, Mapping[str, Mapping[str, float | int | None]]]
    metrics_by_family: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float | int | None]]]]
    paired_comparisons: tuple[PairedComparisonV1, ...]
    perturbation: PerturbationRobustnessReportV1
    verifier_regret_correlation: Mapping[str, float | None]
    stop_rule: StopRuleVerdictV1
    caveats: tuple[str, ...]
    version_stamp: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "teacher_source_id": self.teacher_source_id,
            "n_eval_traces": self.n_eval_traces,
            "n_frequency_training_traces": self.n_frequency_training_traces,
            "top_k": self.top_k,
            "metrics_overall": {
                comparator: dict(metrics) for comparator, metrics in self.metrics_overall.items()
            },
            "metrics_by_family": {
                family: {comparator: dict(metrics) for comparator, metrics in per_comparator.items()}
                for family, per_comparator in self.metrics_by_family.items()
            },
            "paired_comparisons": [c.to_dict() for c in self.paired_comparisons],
            "perturbation": self.perturbation.to_dict(),
            "verifier_regret_correlation": dict(self.verifier_regret_correlation),
            "stop_rule": self.stop_rule.to_dict(),
            "caveats": list(self.caveats),
            "version_stamp": dict(self.version_stamp) if self.version_stamp is not None else None,
        }


def _aggregate_metrics(
    rankings: Mapping[str, Mapping[str, RankingV1 | None]],
    accepted_by_trace: Mapping[str, frozenset[str]],
    metric_specs: Mapping[str, Callable[[RankingV1, frozenset[str]], float | None]],
    *,
    trace_ids: frozenset[str] | None,
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    out: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for comparator_id, per_trace in rankings.items():
        filtered = {
            trace_id: ranking
            for trace_id, ranking in per_trace.items()
            if trace_ids is None or trace_id in trace_ids
        }
        out[comparator_id] = {}
        for metric_name, fn in metric_specs.items():
            values: list[float] = []
            for trace_id, ranking in filtered.items():
                if ranking is None:
                    continue
                value = fn(ranking, accepted_by_trace[trace_id])
                if value is not None:
                    values.append(value)
            out[comparator_id][metric_name] = {
                "mean": (sum(values) / len(values)) if values else None,
                "n": len(values),
            }
        calibration_rows = _pooled_probability_rows(filtered, accepted_by_trace, top1_only=False)
        selective_rows = _pooled_probability_rows(filtered, accepted_by_trace, top1_only=True)
        out[comparator_id]["calibration_ece"] = {
            "mean": expected_calibration_error(calibration_rows),
            "n": len(calibration_rows),
        }
        out[comparator_id]["selective_risk_aurc"] = {
            "mean": selective_risk_aurc(selective_rows),
            "n": len(selective_rows),
        }
    return out


def compare_operator_teacher_ceiling(
    *,
    eval_traces: Sequence[OperatorDecisionStateTraceV1],
    frequency_training_traces: Sequence[OperatorDecisionStateTraceV1],
    teacher: OperatorTeacherAdapter,
    top_k: int = DEFAULT_TOP_K,
    perturbation_bound: float = DECLARED_PERTURBATION_BOUND,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 0,
    perturbation_seed: int = 0,
    min_correlation: float = 0.0,
) -> OperatorTeacherCeilingReportV1:
    """Run the full SLM-387 one-teacher-ceiling comparison.

    ``frequency_training_traces`` must be disjoint from ``eval_traces`` --
    this is enforced, not merely documented, because a frequency prior
    trained on the traces it is then scored against would leak the
    evaluation's own gold labels into a "baseline."
    """
    if not eval_traces:
        raise ValueError("compare_operator_teacher_ceiling requires at least one eval trace")
    eval_ids = {t.trace_id for t in eval_traces}
    if len(eval_ids) != len(eval_traces):
        raise ValueError("eval_traces must have unique trace_id values")
    train_ids = {t.trace_id for t in frequency_training_traces}
    overlap = eval_ids & train_ids
    if overlap:
        raise ValueError(
            f"frequency_training_traces leak into eval_traces: {sorted(overlap)}"
        )

    frequency_table = build_frequency_table(frequency_training_traces)
    comparators: dict[str, RankingComparator] = {
        "baseline:frequency": FrequencyBaselineV1(frequency_table),
        "baseline:compiler_order": CompilerOrderBaselineV1(),
        "baseline:current_scorer": CurrentScorerBaselineV1(),
        "baseline:descriptor_similarity": DescriptorSimilarityBaselineV1(),
        "oracle:accepted_set": OracleAcceptedSetComparatorV1(),
    }

    rankings: dict[str, dict[str, RankingV1 | None]] = {
        comparator_id: {trace.trace_id: comparator.rank(trace) for trace in eval_traces}
        for comparator_id, comparator in comparators.items()
    }
    rankings[teacher.source_id] = {
        trace.trace_id: teacher.rank(default_teacher_query(trace)) for trace in eval_traces
    }

    metric_specs = _metric_specs(top_k)
    accepted_by_trace = {
        trace.trace_id: frozenset(trace.accepted_application_ids) for trace in eval_traces
    }
    family_by_trace = {trace.trace_id: classify_decision_family(trace) for trace in eval_traces}

    metrics_overall = _aggregate_metrics(
        rankings, accepted_by_trace, metric_specs, trace_ids=None
    )
    families = sorted({family.value for family in family_by_trace.values()})
    metrics_by_family = {
        family: _aggregate_metrics(
            rankings,
            accepted_by_trace,
            metric_specs,
            trace_ids=frozenset(
                tid for tid, f in family_by_trace.items() if f.value == family
            ),
        )
        for family in families
    }

    paired: list[PairedComparisonV1] = []
    teacher_rankings = rankings[teacher.source_id]
    for baseline_id in REQUIRED_BASELINE_IDS:
        baseline_rankings = rankings[baseline_id]
        for metric_name, fn in metric_specs.items():
            diffs = []
            for trace in eval_traces:
                teacher_ranking = teacher_rankings.get(trace.trace_id)
                baseline_ranking = baseline_rankings.get(trace.trace_id)
                if teacher_ranking is None or baseline_ranking is None:
                    continue
                accepted = accepted_by_trace[trace.trace_id]
                teacher_value = fn(teacher_ranking, accepted)
                baseline_value = fn(baseline_ranking, accepted)
                if teacher_value is None or baseline_value is None:
                    continue
                diffs.append(teacher_value - baseline_value)
            paired.append(
                _paired_comparison(
                    metric_name=metric_name,
                    family=None,
                    baseline_source_id=baseline_id,
                    teacher_source_id=teacher.source_id,
                    diffs=diffs,
                    n_boot=n_bootstrap,
                    seed=bootstrap_seed,
                )
            )
        for family in families:
            family_trace_ids = {
                tid for tid, f in family_by_trace.items() if f.value == family
            }
            diffs = []
            for trace_id in family_trace_ids:
                teacher_ranking = teacher_rankings.get(trace_id)
                baseline_ranking = baseline_rankings.get(trace_id)
                if teacher_ranking is None or baseline_ranking is None:
                    continue
                accepted = accepted_by_trace[trace_id]
                teacher_value = reciprocal_rank(teacher_ranking, accepted)
                baseline_value = reciprocal_rank(baseline_ranking, accepted)
                if teacher_value is None or baseline_value is None:
                    continue
                diffs.append(teacher_value - baseline_value)
            paired.append(
                _paired_comparison(
                    metric_name=PRIMARY_METRIC,
                    family=DecisionFamily(family),
                    baseline_source_id=baseline_id,
                    teacher_source_id=teacher.source_id,
                    diffs=diffs,
                    n_boot=n_bootstrap,
                    seed=bootstrap_seed,
                )
            )

    perturbation = run_perturbation_robustness(
        teacher, eval_traces, bound=perturbation_bound, seed=perturbation_seed
    )

    regret_correlation: dict[str, float | None] = {}
    for comparator_id, per_trace in rankings.items():
        rows = _pooled_probability_rows(per_trace, accepted_by_trace, top1_only=False)
        regret_correlation[comparator_id] = (
            spearman_correlation([r[0] for r in rows], [1.0 if r[1] else 0.0 for r in rows])
            if rows
            else None
        )

    stop_rule = evaluate_stop_rule(
        paired_comparisons=tuple(
            c for c in paired if c.family is None and c.metric_name == PRIMARY_METRIC
        ),
        perturbation_report=perturbation,
        verifier_regret_correlation=regret_correlation.get(teacher.source_id),
        min_correlation=min_correlation,
    )

    caveats = (
        "SLM-387 (DSH4-02): synthetic stand-in teacher only -- no real external "
        "LLM teacher is called or scored. A production KD-readiness claim "
        "requires swapping a real teacher behind OperatorTeacherAdapter and "
        "re-running this exact harness.",
        "claim_class=wiring; this report is fixture/wiring evidence, never a "
        "ship or KD-readiness claim.",
        "LegalOperatorActionV1 does not carry full ActionEffectV1 for unapplied "
        "candidates (only proof/effect digests); descriptor features are "
        "bounded by that same DSH3-06 visibility.",
    )

    return OperatorTeacherCeilingReportV1(
        schema=_SCHEMA_VERSION,
        teacher_source_id=teacher.source_id,
        n_eval_traces=len(eval_traces),
        n_frequency_training_traces=len(frequency_training_traces),
        top_k=top_k,
        metrics_overall=metrics_overall,
        metrics_by_family=metrics_by_family,
        paired_comparisons=tuple(paired),
        perturbation=perturbation,
        verifier_regret_correlation=regret_correlation,
        stop_rule=stop_rule,
        caveats=caveats,
    )


def write_operator_teacher_ceiling_report(
    path: str | Path, report: OperatorTeacherCeilingReportV1
) -> None:
    Path(path).write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
