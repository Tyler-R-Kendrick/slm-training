"""DSH4-03: accepted-set and verifier-backed operator distillation with DEFER (SLM-388).

SLM-387 (DSH4-02, ``operator_teacher_ceiling.py``) asked *should we distill
this teacher at all* and answered, honestly, ``no_go_defer`` on synthetic
data: ``SyntheticDescriptorTeacherV1`` did not significantly beat any
required baseline on paired MRR and its scores negatively correlated with
verifier-backed (accepted) outcomes. This module answers the DSH4-03
follow-on question the issue poses regardless of that outcome: *given the
teacher we actually have, does adding accepted-set mass, verifier-backed
ranking, calibration awareness, and a fail-closed DEFER gate produce a
better matched-arm distillation target than hard accepted-set labels or the
current (nondistilled) certified controller* -- and does ``DEFER`` protect
the exact/ordinary path when the evidence is not good enough to trust?

Honesty scoping (read before using any part of this module)
-------------------------------------------------------------
**This module does not train or evaluate a neural student.** Exactly like
DSH4-02's teacher-ceiling harness, every "arm" below produces a *target
ranking* -- the distribution a student trained under that arm's loss
combination would be pushed toward on one decision -- computed directly from
typed evidence, never learned by gradient descent. A production
KD-readiness claim requires actually training a student under each arm's
loss and evaluating its live rollouts; that is out of scope in this
CPU-only, no-GPU sandbox (AGENTS.md hard run cap / no paid GPU without
approval). Whatever this harness reports -- including a defer -- is its
honest finding on synthetic data, never a production claim.

**No real external LLM teacher is called or scored.** This module reuses
DSH4-02's ``SyntheticDescriptorTeacherV1`` unchanged. It is not re-run to
flip DSH4-02's own verdict; DSH4-03 asks whether verifier-anchored KD and
DEFER can still add value *given* that teacher, or whether the honest answer
remains "retain the teacher as an evaluation/data tool only" (this issue's
own stop rule).

**The held-out "CAP2-proxy" metric is not the frozen ``dsh3-13`` CAP2
operator suite.** Running that suite requires full compiler+controller
wiring beyond this fixture's scope. This module reports a fixture-scale
substitute -- paired accepted-set-mass bootstrap confidence intervals on a
held-out split, across seeds -- for the same acceptance question ("does the
arm improve certified-controller ranking quality on held-out states"), and
labels it as a substitution rather than silently claiming the frozen suite
ran.

Reused, not duplicated
-----------------------
* ``OperatorDecisionStateTraceV1`` (DSH4-01,
  ``slm_training.harnesses.distill.operator_decision_state``) -- the
  certified state/legal-set/acceptance export every arm ranks over.
* ``RankingV1`` / ``RankingReliability`` / ``OperatorTeacherAdapter`` /
  ``DecisionFamily`` / ``classify_decision_family`` / ``default_teacher_query``
  / ``CompilerOrderBaselineV1`` / ``CurrentScorerBaselineV1`` /
  ``FrequencyBaselineV1`` / ``FrequencyTableV1`` / ``build_frequency_table`` /
  ``OracleAcceptedSetComparatorV1`` / ``accepted_set_mass`` /
  ``expected_calibration_error`` / ``spearman_correlation`` (DSH4-02,
  ``slm_training.harnesses.distill.operator_teacher_ceiling``) -- the ranking
  contract, decision-family slices, deterministic compiler-order fallback,
  current-scorer baseline used here as the "nondistilled certified
  controller" reference, and the disjoint-pool frequency prior reused below
  as the *verifier ranking* signal (see "Verifier ranking, not leakage").
* ``wilson_interval`` / ``bootstrap_paired_ci`` (SLM-183,
  ``slm_training.evals.power_protocol``) -- the canonical proportion and
  paired-bootstrap significance machinery, reused instead of a third local
  reimplementation of either.

Verifier ranking, not leakage
------------------------------
A naive "verifier-backed ranking" signal would read a decision's own
``accepted_application_ids`` when building that same decision's KD target --
which would be circular (the arm would trivially "predict" the label it was
handed). Every arm below avoids this: ``FrequencyBaselineV1``, trained on a
pool of traces *disjoint* from every trace any arm is scored against
(exactly DSH4-02's own enforced disjointness contract), stands in for
"verifier ranking" -- a generalizable prior over historically
replay-verified acceptances, never the specific decision's own answer. The
``hard_accepted_set`` arm below is that same disjoint-trained prior collapsed
to a hard one-hot label (the standard "train a classifier with one-hot
cross-entropy on the accepted label" baseline this issue's Decision section
asks to distill *beyond*) -- not an oracle with per-decision label access.

Matched arms (issue "Experiment")
-----------------------------------
* ``hard_accepted_set`` -- disjoint-trained accepted-label frequency prior,
  hard-argmaxed. The standard SFT-with-hard-labels control.
* ``teacher_argmax`` -- the (weak, per DSH4-02) synthetic teacher's own
  top-1 pick, hard-argmaxed. Pure single-teacher-vote imitation.
* ``offline_conditional_kd`` -- the teacher's full legal-set-conditioned
  distribution, entropy/reliability-mode shaped (forward-KL/JS mass-covering
  in high-entropy or approximate-reliability states; reverse-KL-style
  sharpening only in low-entropy, ``EXACT``-reliability states -- see
  ``classify_kl_mode``). No verifier signal.
* ``verifier_ranking_kd`` -- ``offline_conditional_kd``'s shaped target,
  additionally anchored toward the disjoint-trained verifier (frequency)
  prior, weighted more heavily when trace reliability is ``APPROXIMATE``
  (the KD/teacher signal is least trustworthy exactly where legal-set
  coverage itself is not exact).
* ``verifier_ranking_kd_defer`` -- ``verifier_ranking_kd``'s target, but
  fails closed to ``CompilerOrderBaselineV1`` (the deterministic, exact,
  ordinary path) whenever the state is a forced singleton/empty decision,
  the legal set is not exactly enumerated, or the KD target's own top-1/
  top-2 margin is below a declared threshold -- i.e. DEFER never lets a
  learned target override a decision the compiler has not exactly proven,
  operationalizing "never allow the controller to create legality or
  destructively prune without exact proof."

Stop rule (issue SLM-388): if distillation is inert (predictions identical
to the nondistilled controller), fragile (regresses forced-bypass/legality
invariants), or does not improve held-out CAP2-proxy ranking quality beyond
the nondistilled controller across every seed -- retain the teacher as an
evaluation/data tool only. ``compare_operator_verifier_kd_defer`` reports
this honestly per arm; it never partially passes an arm and never edits an
arm's construction to force a particular verdict.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from slm_training.evals.power_protocol import bootstrap_paired_ci, wilson_interval
from slm_training.harnesses.distill.operator_decision_state import (
    OperatorDecisionStateTraceV1,
)
from slm_training.harnesses.distill.operator_teacher_ceiling import (
    CompilerOrderBaselineV1,
    CurrentScorerBaselineV1,
    DecisionFamily,
    FrequencyBaselineV1,
    OperatorTeacherAdapter,
    RankingComparator,
    RankingReliability,
    RankingV1,
    accepted_set_mass,
    build_frequency_table,
    classify_decision_family,
    default_teacher_query,
    expected_calibration_error,
    spearman_correlation,
)

__all__ = [
    "DEFER_MARGIN_THRESHOLD",
    "FORWARD_KL_MASS_COVER_WEIGHT",
    "INELIGIBLE_DECISION_FAMILIES",
    "REVERSE_KL_ENTROPY_RATIO_THRESHOLD",
    "REVERSE_KL_SHARPEN_TEMPERATURE",
    "VERIFIER_ANCHOR_WEIGHT_APPROXIMATE",
    "VERIFIER_ANCHOR_WEIGHT_EXACT",
    "ARM_ORDER",
    "ArmDecisionChangeReportV1",
    "ArmReportV1",
    "Cap2ProxySeedResultV1",
    "DistillArm",
    "Dsh403StopRuleVerdictV1",
    "KLMode",
    "OperatorVerifierKDDeferReportV1",
    "StructuralRegressionReportV1",
    "apply_kd_shaping",
    "build_arm_target",
    "classify_kl_mode",
    "compare_operator_verifier_kd_defer",
    "evaluate_decision_changes",
    "evaluate_structural_regressions",
    "normalized_entropy",
    "write_operator_verifier_kd_defer_report",
]

_SCHEMA_VERSION = "dsh4-03.v1"

#: Decision families with at most one legal action. A controller never has a
#: choice to change on these -- invariant #5 (forced bypass on singletons) --
#: so they are excluded from every "eligible choice changed" measurement and
#: instead used as the structural forced-bypass regression check.
INELIGIBLE_DECISION_FAMILIES: frozenset[DecisionFamily] = frozenset(
    {
        DecisionFamily.GOLD_EMPTY,
        DecisionFamily.GOLD_SINGLETON,
        DecisionFamily.ON_POLICY_EMPTY,
        DecisionFamily.ON_POLICY_SINGLETON,
    }
)

#: Preregistered entropy-ratio bound below which reverse-KL-style
#: (mode-seeking) pressure is allowed -- only in ``EXACT``-reliability
#: states (issue: "reverse-KL-style pressure only in low-entropy,
#: high-reliability strata").
REVERSE_KL_ENTROPY_RATIO_THRESHOLD = 0.35
#: Sharpening "temperature" (< 1 sharpens) used as the reverse-KL-pressure
#: proxy: reverse KL is a well-documented mode-seeking (under-dispersed)
#: divergence relative to forward KL, so a peaked reweighting is the
#: intended-direction proxy at this abstraction level (no gradient is
#: literally computed here -- see module honesty scoping).
REVERSE_KL_SHARPEN_TEMPERATURE = 0.5
#: Blend-toward-uniform weight used as the forward-KL/JS mass-covering
#: proxy (issue: "forward KL/JS for high-entropy or multimodal reliable
#: states") -- forward KL / JS divergence is mass-covering (does not
#: zero-force any mode with real support), so a modest blend toward the
#: uniform distribution over the legal set is the intended-direction proxy.
FORWARD_KL_MASS_COVER_WEIGHT = 0.15

#: Verifier (disjoint-trained frequency prior) anchor weight in the KD
#: target blend. Higher when trace reliability is ``APPROXIMATE`` -- the
#: teacher/KD signal is least trustworthy exactly where the legal set was
#: not exactly enumerated, so the verifier prior is trusted more there.
VERIFIER_ANCHOR_WEIGHT_EXACT = 0.3
VERIFIER_ANCHOR_WEIGHT_APPROXIMATE = 0.6

#: Declared minimum top1/top2 probability margin below which the DEFER arm
#: falls back to the deterministic compiler order rather than trust the KD
#: target (issue: "never allow the controller to ... destructively prune
#: without exact proof").
DEFER_MARGIN_THRESHOLD = 0.2


# --------------------------------------------------------------------------- #
# Matched arms
# --------------------------------------------------------------------------- #


class DistillArm(str, Enum):
    HARD_ACCEPTED_SET = "hard_accepted_set"
    TEACHER_ARGMAX = "teacher_argmax"
    OFFLINE_CONDITIONAL_KD = "offline_conditional_kd"
    VERIFIER_RANKING_KD = "verifier_ranking_kd"
    DEFER = "verifier_ranking_kd_defer"


ARM_ORDER: tuple[DistillArm, ...] = (
    DistillArm.HARD_ACCEPTED_SET,
    DistillArm.TEACHER_ARGMAX,
    DistillArm.OFFLINE_CONDITIONAL_KD,
    DistillArm.VERIFIER_RANKING_KD,
    DistillArm.DEFER,
)

#: Arms that are actually gated for "go." ``hard_accepted_set`` is the
#: reference control the Decision section asks other arms to beat, never a
#: candidate itself.
_GATED_ARMS: tuple[DistillArm, ...] = ARM_ORDER[1:]


# --------------------------------------------------------------------------- #
# Small shared helpers (deliberately not imported from operator_teacher_ceiling
# private names -- these read only public trace/ranking fields)
# --------------------------------------------------------------------------- #


def _reliability_of(trace: OperatorDecisionStateTraceV1) -> RankingReliability:
    from slm_training.dsl.operators.legal_set import LegalSetCoverage

    if trace.approximate or trace.coverage is LegalSetCoverage.PARTIAL:
        return RankingReliability.APPROXIMATE
    return RankingReliability.EXACT


def _empty_ranking(
    trace: OperatorDecisionStateTraceV1, source_id: str, reason: str
) -> RankingV1:
    reliability = _reliability_of(trace)
    return RankingV1(
        trace_id=trace.trace_id,
        source_id=source_id,
        application_order=(),
        probabilities=None,
        reliability=reliability,
        reliability_reason=reason,
        approximate=(reliability is RankingReliability.APPROXIMATE),
    )


def _top1_id(ranking: RankingV1) -> str | None:
    if not ranking.application_order:
        return None
    if ranking.probabilities:
        idx = max(range(len(ranking.probabilities)), key=lambda i: ranking.probabilities[i])
        return ranking.application_order[idx]
    return ranking.application_order[0]


def _argmax_hard_label(ranking: RankingV1, source_id: str) -> RankingV1:
    """Collapse ``ranking`` to a hard one-hot label on its own top-1 pick."""
    if not ranking.application_order:
        return replace(ranking, source_id=source_id)
    if ranking.probabilities is not None:
        top_idx = max(range(len(ranking.probabilities)), key=lambda i: ranking.probabilities[i])
    else:
        top_idx = 0
    n = len(ranking.application_order)
    order = (ranking.application_order[top_idx],) + tuple(
        aid for i, aid in enumerate(ranking.application_order) if i != top_idx
    )
    probabilities = (1.0,) + (0.0,) * (n - 1)
    return RankingV1(
        trace_id=ranking.trace_id,
        source_id=source_id,
        application_order=order,
        probabilities=probabilities,
        reliability=ranking.reliability,
        reliability_reason=f"hard_argmax_label; {ranking.reliability_reason}",
        approximate=ranking.approximate,
    )


# --------------------------------------------------------------------------- #
# Entropy/reliability-aware KL-mode shaping
# --------------------------------------------------------------------------- #


class KLMode(str, Enum):
    FORWARD_KL_JS = "forward_kl_js"
    REVERSE_KL_PRESSURE = "reverse_kl_pressure"


def normalized_entropy(probabilities: tuple[float, ...] | None) -> float | None:
    """Shannon entropy of ``probabilities``, normalized to ``[0, 1]`` by
    ``log2(n)``. ``None`` when there is nothing to measure (no distribution
    or a single candidate)."""
    if not probabilities:
        return None
    n = len(probabilities)
    if n <= 1:
        return 0.0
    bits = -sum(p * math.log2(p) for p in probabilities if p > 0.0)
    max_bits = math.log2(n)
    return bits / max_bits if max_bits > 0 else 0.0


def classify_kl_mode(ranking: RankingV1, trace: OperatorDecisionStateTraceV1) -> KLMode:
    """Choose forward-KL/JS (mass-covering, default) vs reverse-KL-style
    (mode-seeking) pressure. Reverse-KL-style pressure is allowed only when
    the trace's legal-set enumeration is ``EXACT`` *and* ``ranking`` is
    already low-entropy -- the issue's "low-entropy, high-reliability
    strata" bound."""
    entropy_ratio = normalized_entropy(ranking.probabilities)
    if (
        _reliability_of(trace) is RankingReliability.EXACT
        and entropy_ratio is not None
        and entropy_ratio <= REVERSE_KL_ENTROPY_RATIO_THRESHOLD
    ):
        return KLMode.REVERSE_KL_PRESSURE
    return KLMode.FORWARD_KL_JS


def apply_kd_shaping(ranking: RankingV1, mode: KLMode, *, arm_source_id: str) -> RankingV1:
    """Reshape ``ranking``'s probabilities per ``mode``. A monotonic
    transform of each probability (power sharpening, or a uniform blend with
    a fixed weight) never changes the relative order of candidates, so
    ``application_order`` is preserved as-is."""
    if ranking.probabilities is None or not ranking.application_order:
        return replace(ranking, source_id=arm_source_id)
    probs = ranking.probabilities
    if mode is KLMode.REVERSE_KL_PRESSURE:
        exponent = 1.0 / REVERSE_KL_SHARPEN_TEMPERATURE
        raw = [p**exponent for p in probs]
        reason_tag = f"reverse_kl_pressure_sharpened(temperature={REVERSE_KL_SHARPEN_TEMPERATURE})"
    else:
        n = len(probs)
        uniform = 1.0 / n
        raw = [
            (1.0 - FORWARD_KL_MASS_COVER_WEIGHT) * p + FORWARD_KL_MASS_COVER_WEIGHT * uniform
            for p in probs
        ]
        reason_tag = f"forward_kl_js_mass_covering(weight={FORWARD_KL_MASS_COVER_WEIGHT})"
    total = sum(raw)
    shaped_probs = tuple(v / total for v in raw)
    return RankingV1(
        trace_id=ranking.trace_id,
        source_id=arm_source_id,
        application_order=ranking.application_order,
        probabilities=shaped_probs,
        reliability=ranking.reliability,
        reliability_reason=f"{reason_tag}; {ranking.reliability_reason}",
        approximate=ranking.approximate,
    )


def _verifier_anchor_blend(
    shaped: RankingV1,
    trace: OperatorDecisionStateTraceV1,
    verifier: RankingComparator,
    *,
    source_id: str,
) -> RankingV1:
    """Blend ``shaped`` toward the disjoint-trained verifier (frequency)
    prior. Never reads ``trace.accepted_application_ids`` -- see module
    docstring "Verifier ranking, not leakage"."""
    if shaped.probabilities is None or not shaped.application_order:
        return replace(shaped, source_id=source_id)
    verifier_ranking = verifier.rank(trace)
    if verifier_ranking is None or verifier_ranking.probabilities is None:
        return replace(shaped, source_id=source_id)
    weight = (
        VERIFIER_ANCHOR_WEIGHT_EXACT
        if _reliability_of(trace) is RankingReliability.EXACT
        else VERIFIER_ANCHOR_WEIGHT_APPROXIMATE
    )
    verifier_by_id = dict(zip(verifier_ranking.application_order, verifier_ranking.probabilities))
    blended_raw = [
        (1.0 - weight) * p + weight * verifier_by_id.get(aid, 0.0)
        for aid, p in zip(shaped.application_order, shaped.probabilities)
    ]
    total = sum(blended_raw)
    if total <= 0.0:
        return replace(shaped, source_id=source_id)
    pairs = sorted(
        zip(shaped.application_order, (v / total for v in blended_raw)),
        key=lambda pair: (-pair[1], pair[0]),
    )
    ids, probabilities = zip(*pairs)
    return RankingV1(
        trace_id=shaped.trace_id,
        source_id=source_id,
        application_order=tuple(ids),
        probabilities=tuple(probabilities),
        reliability=shaped.reliability,
        reliability_reason=f"verifier_frequency_anchor(weight={weight}); {shaped.reliability_reason}",
        approximate=shaped.approximate,
    )


def _should_defer(
    trace: OperatorDecisionStateTraceV1, ranking: RankingV1
) -> tuple[bool, str]:
    n = len(trace.legal_set.operator_actions)
    if n <= 1:
        return True, "singleton_or_empty_legal_set_forced_bypass"
    if _reliability_of(trace) is RankingReliability.APPROXIMATE:
        return True, "approximate_or_partial_legal_set_coverage_no_exact_proof"
    if not ranking.probabilities or len(ranking.probabilities) < 2:
        return True, "kd_target_missing_probability_margin"
    top1, top2 = sorted(ranking.probabilities, reverse=True)[:2]
    margin = top1 - top2
    if margin < DEFER_MARGIN_THRESHOLD:
        return (
            True,
            f"kd_target_margin_below_threshold(margin={margin:.4f},threshold={DEFER_MARGIN_THRESHOLD})",
        )
    return False, "kd_target_confident_and_exact"


# --------------------------------------------------------------------------- #
# Arm target construction
# --------------------------------------------------------------------------- #


def build_arm_target(
    trace: OperatorDecisionStateTraceV1,
    *,
    teacher: OperatorTeacherAdapter,
    verifier: RankingComparator,
    arm: DistillArm,
) -> RankingV1:
    """Build ``arm``'s KD target ranking for one decision-state trace.

    ``verifier`` must be a comparator (canonically ``FrequencyBaselineV1``)
    trained on a pool disjoint from ``trace`` -- callers are responsible for
    that disjointness; ``compare_operator_verifier_kd_defer`` enforces it.
    """
    actions = trace.legal_set.operator_actions
    source_id = f"arm:{arm.value}"
    if not actions:
        return _empty_ranking(trace, source_id, "empty_legal_set")

    if arm is DistillArm.HARD_ACCEPTED_SET:
        verifier_ranking = verifier.rank(trace)
        if verifier_ranking is None:
            return _empty_ranking(trace, source_id, "verifier_returned_none")
        return _argmax_hard_label(verifier_ranking, source_id)

    teacher_ranking = teacher.rank(default_teacher_query(trace))
    if teacher_ranking is None:
        teacher_ranking = _empty_ranking(trace, source_id, "teacher_returned_none")

    if arm is DistillArm.TEACHER_ARGMAX:
        return _argmax_hard_label(teacher_ranking, source_id)

    kl_mode = classify_kl_mode(teacher_ranking, trace)
    shaped = apply_kd_shaping(teacher_ranking, kl_mode, arm_source_id=source_id)

    if arm is DistillArm.OFFLINE_CONDITIONAL_KD:
        return shaped

    verifier_anchored = _verifier_anchor_blend(shaped, trace, verifier, source_id=source_id)

    if arm is DistillArm.VERIFIER_RANKING_KD:
        return verifier_anchored

    if arm is DistillArm.DEFER:
        should_defer, reason = _should_defer(trace, verifier_anchored)
        if should_defer:
            fallback = CompilerOrderBaselineV1().rank(trace)
            if fallback is None:
                return _empty_ranking(trace, source_id, f"DEFER:{reason}")
            return replace(
                fallback,
                source_id=source_id,
                reliability_reason=f"DEFER:{reason}; {fallback.reliability_reason}",
            )
        return replace(verifier_anchored, source_id=source_id)

    raise ValueError(f"unknown arm {arm!r}")


def _defer_reason(
    trace: OperatorDecisionStateTraceV1,
    *,
    teacher: OperatorTeacherAdapter,
    verifier: RankingComparator,
) -> str | None:
    """Recompute whether the DEFER arm would defer on ``trace``, and why."""
    actions = trace.legal_set.operator_actions
    if not actions:
        return None
    teacher_ranking = teacher.rank(default_teacher_query(trace))
    if teacher_ranking is None:
        teacher_ranking = _empty_ranking(trace, "internal:defer_probe", "teacher_returned_none")
    kl_mode = classify_kl_mode(teacher_ranking, trace)
    shaped = apply_kd_shaping(teacher_ranking, kl_mode, arm_source_id="internal:defer_probe")
    anchored = _verifier_anchor_blend(shaped, trace, verifier, source_id="internal:defer_probe")
    should_defer, reason = _should_defer(trace, anchored)
    return reason if should_defer else None


# --------------------------------------------------------------------------- #
# Decision-change evidence (acceptance: >=5% eligible choices change, correct > wrong)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArmDecisionChangeReportV1:
    arm: str
    baseline_source_id: str
    n_eligible: int
    n_changed: int
    n_correct_changes: int
    n_wrong_changes: int
    pct_changed: Mapping[str, Any]
    correct_vs_wrong: Mapping[str, Any]
    identical_prediction_fraction: float
    prediction_identical_rejected: bool
    meets_change_threshold: bool
    correct_exceeds_wrong: bool
    schema: str = "operator_verifier_kd_defer_change_report/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "arm": self.arm,
            "baseline_source_id": self.baseline_source_id,
            "n_eligible": self.n_eligible,
            "n_changed": self.n_changed,
            "n_correct_changes": self.n_correct_changes,
            "n_wrong_changes": self.n_wrong_changes,
            "pct_changed": dict(self.pct_changed),
            "correct_vs_wrong": dict(self.correct_vs_wrong),
            "identical_prediction_fraction": self.identical_prediction_fraction,
            "prediction_identical_rejected": self.prediction_identical_rejected,
            "meets_change_threshold": self.meets_change_threshold,
            "correct_exceeds_wrong": self.correct_exceeds_wrong,
        }


def evaluate_decision_changes(
    arm_rankings: Mapping[str, RankingV1 | None],
    reference_rankings: Mapping[str, RankingV1 | None],
    accepted_by_trace: Mapping[str, frozenset[str]],
    family_by_trace: Mapping[str, DecisionFamily],
    *,
    arm: DistillArm,
    baseline_source_id: str,
    change_threshold: float = 0.05,
    identical_reject_threshold: float = 0.99,
) -> ArmDecisionChangeReportV1:
    """Compare ``arm_rankings``' top-1 pick against ``reference_rankings``'
    over every *eligible* (non-forced) decision.

    A changed decision is "correct" iff the arm's new top-1 pick is in the
    trace's replay-verified accepted set, "wrong" otherwise. If the arm's
    top-1 pick is identical to the reference on
    ``identical_reject_threshold`` or more of eligible decisions, the whole
    comparison is rejected (issue acceptance: "prediction-identical
    auxiliary-loss results are rejected") rather than scored.
    """
    eligible_ids = [
        tid for tid, fam in family_by_trace.items() if fam not in INELIGIBLE_DECISION_FAMILIES
    ]
    n_eligible = 0
    n_changed = 0
    n_correct = 0
    n_wrong = 0
    n_identical = 0
    for tid in eligible_ids:
        arm_ranking = arm_rankings.get(tid)
        reference_ranking = reference_rankings.get(tid)
        if (
            arm_ranking is None
            or reference_ranking is None
            or not arm_ranking.application_order
            or not reference_ranking.application_order
        ):
            continue
        n_eligible += 1
        arm_top1 = _top1_id(arm_ranking)
        reference_top1 = _top1_id(reference_ranking)
        if arm_top1 == reference_top1:
            n_identical += 1
            continue
        n_changed += 1
        accepted = accepted_by_trace[tid]
        if arm_top1 in accepted:
            n_correct += 1
        else:
            n_wrong += 1

    pct_changed = wilson_interval(n_changed, n_eligible)
    correct_total = n_correct + n_wrong
    correct_vs_wrong = wilson_interval(n_correct, correct_total)
    identical_fraction = (n_identical / n_eligible) if n_eligible else 1.0
    prediction_identical_rejected = identical_fraction >= identical_reject_threshold
    meets_change_threshold = (
        not prediction_identical_rejected
        and n_eligible > 0
        and (n_changed / n_eligible) >= change_threshold
    )
    correct_exceeds_wrong = not prediction_identical_rejected and n_correct > n_wrong
    return ArmDecisionChangeReportV1(
        arm=arm.value,
        baseline_source_id=baseline_source_id,
        n_eligible=n_eligible,
        n_changed=n_changed,
        n_correct_changes=n_correct,
        n_wrong_changes=n_wrong,
        pct_changed=pct_changed,
        correct_vs_wrong=correct_vs_wrong,
        identical_prediction_fraction=identical_fraction,
        prediction_identical_rejected=prediction_identical_rejected,
        meets_change_threshold=meets_change_threshold,
        correct_exceeds_wrong=correct_exceeds_wrong,
    )


# --------------------------------------------------------------------------- #
# Structural regression checks (acceptance: legal-set exactness / CAP0-CAP1 /
# fallback do not regress)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StructuralRegressionReportV1:
    arm: str
    n_singleton_or_empty_checked: int
    singleton_forced_bypass_preserved: bool
    n_illegal_action_checked: int
    no_illegal_action_introduced: bool
    defer_count: int
    defer_rate: float | None
    defer_reasons: Mapping[str, int]
    schema: str = "operator_verifier_kd_defer_structural_report/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "arm": self.arm,
            "n_singleton_or_empty_checked": self.n_singleton_or_empty_checked,
            "singleton_forced_bypass_preserved": self.singleton_forced_bypass_preserved,
            "n_illegal_action_checked": self.n_illegal_action_checked,
            "no_illegal_action_introduced": self.no_illegal_action_introduced,
            "defer_count": self.defer_count,
            "defer_rate": self.defer_rate,
            "defer_reasons": dict(self.defer_reasons),
        }


def evaluate_structural_regressions(
    arm_rankings: Mapping[str, RankingV1 | None],
    traces_by_id: Mapping[str, OperatorDecisionStateTraceV1],
    family_by_trace: Mapping[str, DecisionFamily],
    *,
    arm: DistillArm,
    defer_reason_by_trace: Mapping[str, str] | None = None,
) -> StructuralRegressionReportV1:
    """Fixture-scale proxy for "legal-set exactness, CAP0/CAP1 retention,
    and fallback behavior do not regress": every arm must (a) never rank an
    action outside the trace's own certified legal set, and (b) never touch
    a forced singleton/empty decision (invariant #5, forced bypass)."""
    n_singleton = 0
    singleton_ok = True
    n_illegal_checked = 0
    illegal_ok = True
    for tid, ranking in arm_rankings.items():
        trace = traces_by_id[tid]
        legal_ids = {a.application_id for a in trace.legal_set.operator_actions}
        if ranking is not None:
            n_illegal_checked += 1
            if not set(ranking.application_order) <= legal_ids:
                illegal_ok = False
        family = family_by_trace[tid]
        if family in INELIGIBLE_DECISION_FAMILIES:
            n_singleton += 1
            got = tuple(sorted(ranking.application_order)) if ranking is not None else ()
            expected = tuple(sorted(legal_ids))
            if got != expected:
                singleton_ok = False

    defer_reasons_counter: dict[str, int] = {}
    defer_count = 0
    if defer_reason_by_trace:
        for reason in defer_reason_by_trace.values():
            defer_count += 1
            defer_reasons_counter[reason] = defer_reasons_counter.get(reason, 0) + 1
    n_total = len(arm_rankings)
    defer_rate = (defer_count / n_total) if (arm is DistillArm.DEFER and n_total) else None

    return StructuralRegressionReportV1(
        arm=arm.value,
        n_singleton_or_empty_checked=n_singleton,
        singleton_forced_bypass_preserved=singleton_ok,
        n_illegal_action_checked=n_illegal_checked,
        no_illegal_action_introduced=illegal_ok,
        defer_count=defer_count,
        defer_rate=defer_rate,
        defer_reasons=defer_reasons_counter,
    )


# --------------------------------------------------------------------------- #
# Held-out CAP2-proxy (acceptance: held-out CAP2 improves across seeds)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Cap2ProxySeedResultV1:
    seed: int
    n_paired: int
    mean_diff: float
    ci_low: float
    ci_high: float
    improves: bool
    schema: str = "operator_verifier_kd_defer_cap2_proxy_seed/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "seed": self.seed,
            "n_paired": self.n_paired,
            "mean_diff": self.mean_diff,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "improves": self.improves,
        }


def _cap2_proxy_seed_result(
    *,
    seed: int,
    pool: Sequence[OperatorDecisionStateTraceV1],
    arm: DistillArm,
    teacher: OperatorTeacherAdapter,
    verifier: RankingComparator,
    baseline: RankingComparator,
    n_bootstrap: int,
    bootstrap_seed: int,
) -> Cap2ProxySeedResultV1:
    arm_vals: list[float] = []
    base_vals: list[float] = []
    for trace in pool:
        accepted = frozenset(trace.accepted_application_ids)
        arm_ranking = build_arm_target(trace, teacher=teacher, verifier=verifier, arm=arm)
        base_ranking = baseline.rank(trace)
        if arm_ranking is None or base_ranking is None:
            continue
        arm_mass = accepted_set_mass(arm_ranking, accepted)
        base_mass = accepted_set_mass(base_ranking, accepted)
        if arm_mass is None or base_mass is None:
            continue
        arm_vals.append(arm_mass)
        base_vals.append(base_mass)
    if not arm_vals:
        return Cap2ProxySeedResultV1(
            seed=seed, n_paired=0, mean_diff=0.0, ci_low=0.0, ci_high=0.0, improves=False
        )
    ci = bootstrap_paired_ci(
        arm_vals,
        base_vals,
        metric=lambda left, right: float(sum(left) / len(left) - sum(right) / len(right)),
        seed=bootstrap_seed + seed,
        resamples=n_bootstrap,
    )
    return Cap2ProxySeedResultV1(
        seed=seed,
        n_paired=len(arm_vals),
        mean_diff=ci["estimate"],
        ci_low=ci["low"],
        ci_high=ci["high"],
        improves=(ci["low"] > 0.0),
    )


# --------------------------------------------------------------------------- #
# Per-arm report + stop rule
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ArmReportV1:
    arm: str
    decision_change_vs_baseline: ArmDecisionChangeReportV1
    decision_change_vs_hard_accepted_set: ArmDecisionChangeReportV1 | None
    cap2_proxy_vs_baseline: tuple[Cap2ProxySeedResultV1, ...]
    cap2_proxy_improves_all_seeds: bool
    structural: StructuralRegressionReportV1
    verifier_regret_correlation: float | None
    calibration_ece: float | None
    arm_go: bool
    reasons: tuple[str, ...]
    schema: str = "operator_verifier_kd_defer_arm_report/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "arm": self.arm,
            "decision_change_vs_baseline": self.decision_change_vs_baseline.to_dict(),
            "decision_change_vs_hard_accepted_set": (
                self.decision_change_vs_hard_accepted_set.to_dict()
                if self.decision_change_vs_hard_accepted_set is not None
                else None
            ),
            "cap2_proxy_vs_baseline": [r.to_dict() for r in self.cap2_proxy_vs_baseline],
            "cap2_proxy_improves_all_seeds": self.cap2_proxy_improves_all_seeds,
            "structural": self.structural.to_dict(),
            "verifier_regret_correlation": self.verifier_regret_correlation,
            "calibration_ece": self.calibration_ece,
            "arm_go": self.arm_go,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class Dsh403StopRuleVerdictV1:
    go: bool
    passing_arms: tuple[str, ...]
    reasons: tuple[str, ...]
    schema: str = "operator_verifier_kd_defer_stop_rule/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "go": self.go,
            "recommendation": "go" if self.go else "no_go_defer",
            "passing_arms": list(self.passing_arms),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class OperatorVerifierKDDeferReportV1:
    schema: str
    teacher_source_id: str
    nondistilled_baseline_source_id: str
    n_eval_traces: int
    n_verifier_training_traces: int
    held_out_seeds: tuple[int, ...]
    n_held_out_traces_per_seed: Mapping[int, int]
    arms: Mapping[str, ArmReportV1]
    stop_rule: Dsh403StopRuleVerdictV1
    caveats: tuple[str, ...]
    version_stamp: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "teacher_source_id": self.teacher_source_id,
            "nondistilled_baseline_source_id": self.nondistilled_baseline_source_id,
            "n_eval_traces": self.n_eval_traces,
            "n_verifier_training_traces": self.n_verifier_training_traces,
            "held_out_seeds": list(self.held_out_seeds),
            "n_held_out_traces_per_seed": dict(self.n_held_out_traces_per_seed),
            "arms": {arm_id: report.to_dict() for arm_id, report in self.arms.items()},
            "stop_rule": self.stop_rule.to_dict(),
            "caveats": list(self.caveats),
            "version_stamp": dict(self.version_stamp) if self.version_stamp is not None else None,
        }


def compare_operator_verifier_kd_defer(
    *,
    eval_traces: Sequence[OperatorDecisionStateTraceV1],
    verifier_training_traces: Sequence[OperatorDecisionStateTraceV1],
    held_out_traces_by_seed: Mapping[int, Sequence[OperatorDecisionStateTraceV1]],
    teacher: OperatorTeacherAdapter,
    nondistilled_baseline: RankingComparator | None = None,
    change_threshold: float = 0.05,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 0,
) -> OperatorVerifierKDDeferReportV1:
    """Run the full SLM-388 matched-arm distillation comparison.

    ``verifier_training_traces`` must be disjoint from ``eval_traces`` and
    every pool in ``held_out_traces_by_seed`` (enforced, not merely
    documented -- exactly DSH4-02's own frequency-training/eval disjointness
    contract, extended to every held-out seed pool as well).
    """
    if not eval_traces:
        raise ValueError("compare_operator_verifier_kd_defer requires at least one eval trace")
    if not held_out_traces_by_seed:
        raise ValueError("compare_operator_verifier_kd_defer requires at least one held-out seed pool")

    baseline = nondistilled_baseline or CurrentScorerBaselineV1()

    eval_ids = {t.trace_id for t in eval_traces}
    if len(eval_ids) != len(eval_traces):
        raise ValueError("eval_traces must have unique trace_id values")
    train_ids = {t.trace_id for t in verifier_training_traces}
    overlap = eval_ids & train_ids
    if overlap:
        raise ValueError(f"verifier_training_traces leak into eval_traces: {sorted(overlap)}")

    all_held_ids: set[str] = set()
    for seed, pool in held_out_traces_by_seed.items():
        ids = {t.trace_id for t in pool}
        if len(ids) != len(pool):
            raise ValueError(f"held_out_traces_by_seed[{seed}] must have unique trace_id values")
        if ids & eval_ids:
            raise ValueError(f"held_out_traces_by_seed[{seed}] leaks into eval_traces")
        if ids & train_ids:
            raise ValueError(f"held_out_traces_by_seed[{seed}] leaks into verifier_training_traces")
        if ids & all_held_ids:
            raise ValueError(f"held_out_traces_by_seed[{seed}] overlaps another seed's held-out pool")
        all_held_ids |= ids

    verifier = FrequencyBaselineV1(build_frequency_table(verifier_training_traces))

    traces_by_id = {t.trace_id: t for t in eval_traces}
    family_by_trace = {t.trace_id: classify_decision_family(t) for t in eval_traces}
    accepted_by_trace = {
        t.trace_id: frozenset(t.accepted_application_ids) for t in eval_traces
    }
    baseline_rankings = {t.trace_id: baseline.rank(t) for t in eval_traces}

    arm_rankings: dict[DistillArm, dict[str, RankingV1 | None]] = {}
    arm_defer_reasons: dict[DistillArm, dict[str, str]] = {}
    for arm in ARM_ORDER:
        rankings: dict[str, RankingV1 | None] = {}
        reasons: dict[str, str] = {}
        for trace in eval_traces:
            rankings[trace.trace_id] = build_arm_target(
                trace, teacher=teacher, verifier=verifier, arm=arm
            )
            if arm is DistillArm.DEFER:
                reason = _defer_reason(trace, teacher=teacher, verifier=verifier)
                if reason is not None:
                    reasons[trace.trace_id] = reason
        arm_rankings[arm] = rankings
        arm_defer_reasons[arm] = reasons

    hard_accepted_rankings = arm_rankings[DistillArm.HARD_ACCEPTED_SET]

    arm_reports: dict[str, ArmReportV1] = {}
    for arm in ARM_ORDER:
        rankings = arm_rankings[arm]

        change_vs_baseline = evaluate_decision_changes(
            rankings,
            baseline_rankings,
            accepted_by_trace,
            family_by_trace,
            arm=arm,
            baseline_source_id=baseline.source_id,
            change_threshold=change_threshold,
        )
        change_vs_hard: ArmDecisionChangeReportV1 | None = None
        if arm is not DistillArm.HARD_ACCEPTED_SET:
            change_vs_hard = evaluate_decision_changes(
                rankings,
                hard_accepted_rankings,
                accepted_by_trace,
                family_by_trace,
                arm=arm,
                baseline_source_id="arm:hard_accepted_set",
                change_threshold=change_threshold,
            )

        structural = evaluate_structural_regressions(
            rankings,
            traces_by_id,
            family_by_trace,
            arm=arm,
            defer_reason_by_trace=arm_defer_reasons.get(arm),
        )

        rows: list[tuple[float, bool]] = []
        for trace_id, ranking in rankings.items():
            if ranking is None or ranking.probabilities is None:
                continue
            accepted = accepted_by_trace[trace_id]
            for application_id, probability in zip(
                ranking.application_order, ranking.probabilities
            ):
                rows.append((probability, application_id in accepted))
        vr_correlation = (
            spearman_correlation([r[0] for r in rows], [1.0 if r[1] else 0.0 for r in rows])
            if rows
            else None
        )
        calibration = expected_calibration_error(rows) if rows else None

        seed_results = tuple(
            _cap2_proxy_seed_result(
                seed=seed,
                pool=pool,
                arm=arm,
                teacher=teacher,
                verifier=verifier,
                baseline=baseline,
                n_bootstrap=n_bootstrap,
                bootstrap_seed=bootstrap_seed,
            )
            for seed, pool in sorted(held_out_traces_by_seed.items())
        )
        cap2_all_seeds = bool(seed_results) and all(r.improves for r in seed_results)

        reasons_list: list[str] = []
        if arm is DistillArm.HARD_ACCEPTED_SET:
            arm_go = False
            reasons_list.append(
                "hard_accepted_set is the reference control arm the Decision asks other "
                "arms to beat; it is never itself gated"
            )
        else:
            if change_vs_baseline.prediction_identical_rejected:
                reasons_list.append(
                    "rejected: top-1 predictions identical to the nondistilled baseline "
                    "on >=99% of eligible decisions (no causal decision change)"
                )
            if not change_vs_baseline.meets_change_threshold:
                reasons_list.append(
                    f"change rate {change_vs_baseline.pct_changed.get('estimate')!r} "
                    f"below the {change_threshold} threshold"
                )
            if not change_vs_baseline.correct_exceeds_wrong:
                reasons_list.append(
                    f"correct changes ({change_vs_baseline.n_correct_changes}) do not "
                    f"exceed wrong changes ({change_vs_baseline.n_wrong_changes})"
                )
            if not cap2_all_seeds:
                reasons_list.append(
                    "held-out CAP2-proxy accepted-set-mass improvement over the "
                    "nondistilled baseline was not significant on every seed"
                )
            if not structural.singleton_forced_bypass_preserved:
                reasons_list.append(
                    "REGRESSION: a forced singleton/empty decision was altered"
                )
            if not structural.no_illegal_action_introduced:
                reasons_list.append(
                    "REGRESSION: an arm ranked an action outside the certified legal set"
                )
            arm_go = (
                not change_vs_baseline.prediction_identical_rejected
                and change_vs_baseline.meets_change_threshold
                and change_vs_baseline.correct_exceeds_wrong
                and cap2_all_seeds
                and structural.singleton_forced_bypass_preserved
                and structural.no_illegal_action_introduced
            )

        arm_reports[arm.value] = ArmReportV1(
            arm=arm.value,
            decision_change_vs_baseline=change_vs_baseline,
            decision_change_vs_hard_accepted_set=change_vs_hard,
            cap2_proxy_vs_baseline=seed_results,
            cap2_proxy_improves_all_seeds=cap2_all_seeds,
            structural=structural,
            verifier_regret_correlation=vr_correlation,
            calibration_ece=calibration,
            arm_go=arm_go,
            reasons=tuple(reasons_list),
        )

    passing = tuple(
        arm.value for arm in (DistillArm.VERIFIER_RANKING_KD, DistillArm.DEFER)
        if arm_reports[arm.value].arm_go
    )
    overall_go = bool(passing)
    stop_reasons: tuple[str, ...] = ()
    if not overall_go:
        stop_reasons = (
            "neither verifier_ranking_kd nor verifier_ranking_kd_defer cleared every "
            "acceptance gate; per the SLM-388 stop rule, retain the teacher as an "
            "evaluation/data tool only",
        )
    stop_rule = Dsh403StopRuleVerdictV1(go=overall_go, passing_arms=passing, reasons=stop_reasons)

    caveats = (
        "SLM-388 (DSH4-03): reuses SLM-387's SyntheticDescriptorTeacherV1 unchanged -- "
        "no real external LLM teacher is called or scored. SLM-387 already found this "
        "synthetic teacher does not beat simple baselines and negatively correlates "
        "with verifier-backed outcomes; this harness asks whether verifier-anchored KD "
        "and DEFER still add value on top of that weak signal, not a re-run of DSH4-02.",
        "This harness does not train or evaluate a neural student. Each arm's ranking "
        "is the *target distribution* a student trained under that arm's loss "
        "combination would be pushed toward on that decision, computed directly (not "
        "learned by gradient descent). A production KD-readiness claim requires "
        "actually training a student under each arm's loss and evaluating its live "
        "rollouts -- out of scope in this CPU-only, no-GPU environment.",
        "The held-out 'CAP2-proxy' metric (paired accepted-set-mass bootstrap CI "
        "across seeds) is NOT the frozen dsh3-13 CAP2 operator suite; it is a "
        "fixture-scale substitute for the same acceptance question, explicitly "
        "labeled as a substitution rather than a claim the frozen suite ran.",
        "claim_class=wiring; this report is fixture/wiring evidence, never a ship or "
        "KD-readiness claim.",
    )

    return OperatorVerifierKDDeferReportV1(
        schema=_SCHEMA_VERSION,
        teacher_source_id=teacher.source_id,
        nondistilled_baseline_source_id=baseline.source_id,
        n_eval_traces=len(eval_traces),
        n_verifier_training_traces=len(verifier_training_traces),
        held_out_seeds=tuple(sorted(held_out_traces_by_seed)),
        n_held_out_traces_per_seed={
            seed: len(pool) for seed, pool in held_out_traces_by_seed.items()
        },
        arms=arm_reports,
        stop_rule=stop_rule,
        caveats=caveats,
    )


def write_operator_verifier_kd_defer_report(
    path: str | Path, report: OperatorVerifierKDDeferReportV1
) -> None:
    Path(path).write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
