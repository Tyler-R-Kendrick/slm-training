"""Tests for DSH4-03 accepted-set/verifier-backed operator distillation with
DEFER (SLM-388).

Exercises the acceptance bar from the Linear issue directly:

* every matched arm (hard accepted-set, teacher argmax, offline conditional
  KD, verifier-ranking+KD, +DEFER) builds a valid, legal-set-bounded target
  ranking, including empty/singleton legal sets;
* entropy/reliability-aware KL-mode selection picks reverse-KL-style
  pressure only in low-entropy, ``EXACT``-reliability strata and forward-
  KL/JS mass-covering everywhere else;
* the verifier (frequency) anchor never reads a trace's own accepted set --
  it is built strictly from a disjoint training pool;
* DEFER falls back to the deterministic compiler order on forced
  singleton/empty decisions, on approximate/partial legal-set coverage, and
  whenever its own KD target's top1/top2 margin is too small, and never on
  anything else;
* the decision-change evaluator correctly measures "eligible choices
  changed" / "correct exceeds wrong" and rejects a prediction-identical
  comparison rather than scoring it;
* the structural regression checks catch an illegal-action or forced-bypass
  violation;
* the end-to-end orchestrator enforces eval/verifier-training/held-out
  disjointness and produces a JSON-round-trippable report whose stop rule
  never partially passes.
"""

from __future__ import annotations

import json

import pytest

from slm_training.evals.solver_state_supervision import SupervisionSource
from slm_training.harnesses.distill import operator_verifier_kd_defer as kd
from slm_training.harnesses.distill.operator_teacher_ceiling import (
    CompilerOrderBaselineV1,
    CurrentScorerBaselineV1,
    DecisionFamily,
    FrequencyBaselineV1,
    RankingReliability,
    RankingV1,
    SyntheticDescriptorTeacherV1,
    build_frequency_table,
)
from tests.test_harnesses.distill.test_operator_teacher_ceiling import _build


def _ranking(
    trace_id: str,
    ids: tuple[str, ...],
    probs: tuple[float, ...] | None,
    *,
    reliability: RankingReliability = RankingReliability.EXACT,
) -> RankingV1:
    return RankingV1(
        trace_id=trace_id,
        source_id="test:ranking",
        application_order=ids,
        probabilities=probs,
        reliability=reliability,
        reliability_reason="test",
        approximate=(reliability is RankingReliability.APPROXIMATE),
    )


# --------------------------------------------------------------------------- #
# normalized_entropy / classify_kl_mode / apply_kd_shaping
# --------------------------------------------------------------------------- #


def test_normalized_entropy_none_for_empty_or_missing():
    assert kd.normalized_entropy(None) is None
    assert kd.normalized_entropy(()) is None


def test_normalized_entropy_zero_for_single_candidate():
    assert kd.normalized_entropy((1.0,)) == 0.0


def test_normalized_entropy_one_for_uniform():
    assert kd.normalized_entropy((0.25, 0.25, 0.25, 0.25)) == pytest.approx(1.0)


def test_normalized_entropy_near_zero_for_peaked():
    entropy = kd.normalized_entropy((0.97, 0.01, 0.01, 0.01))
    assert 0.0 <= entropy < 0.35


def test_classify_kl_mode_reverse_only_when_exact_and_low_entropy():
    fx = _build(trace_id="t-kl-1", n_candidates=5, accepted_indices=(0,))
    peaked = _ranking(
        fx.trace.trace_id,
        tuple(a.application_id for a in fx.trace.legal_set.operator_actions),
        (0.97, 0.01, 0.01, 0.005, 0.005),
    )
    assert kd.classify_kl_mode(peaked, fx.trace) is kd.KLMode.REVERSE_KL_PRESSURE


def test_classify_kl_mode_forward_when_high_entropy():
    fx = _build(trace_id="t-kl-2", n_candidates=5, accepted_indices=(0,))
    flat = _ranking(
        fx.trace.trace_id,
        tuple(a.application_id for a in fx.trace.legal_set.operator_actions),
        (0.2, 0.2, 0.2, 0.2, 0.2),
    )
    assert kd.classify_kl_mode(flat, fx.trace) is kd.KLMode.FORWARD_KL_JS


def test_classify_kl_mode_forward_when_approximate_even_if_peaked():
    fx = _build(
        trace_id="t-kl-3", n_candidates=5, accepted_indices=(0,), approximate=True
    )
    peaked = _ranking(
        fx.trace.trace_id,
        tuple(a.application_id for a in fx.trace.legal_set.operator_actions),
        (0.97, 0.01, 0.01, 0.005, 0.005),
    )
    assert kd.classify_kl_mode(peaked, fx.trace) is kd.KLMode.FORWARD_KL_JS


def test_apply_kd_shaping_reverse_sharpens_without_reordering():
    ranking = _ranking("t", ("a", "b", "c"), (0.5, 0.3, 0.2))
    shaped = kd.apply_kd_shaping(ranking, kd.KLMode.REVERSE_KL_PRESSURE, arm_source_id="arm:x")
    assert shaped.application_order == ranking.application_order
    assert shaped.probabilities[0] > ranking.probabilities[0]
    assert sum(shaped.probabilities) == pytest.approx(1.0)
    assert shaped.source_id == "arm:x"


def test_apply_kd_shaping_forward_flattens_toward_uniform():
    ranking = _ranking("t", ("a", "b", "c"), (0.8, 0.1, 0.1))
    shaped = kd.apply_kd_shaping(ranking, kd.KLMode.FORWARD_KL_JS, arm_source_id="arm:x")
    assert shaped.application_order == ranking.application_order
    assert shaped.probabilities[0] < ranking.probabilities[0]
    assert sum(shaped.probabilities) == pytest.approx(1.0)


def test_apply_kd_shaping_empty_ranking_is_a_no_op():
    ranking = _ranking("t", (), None)
    shaped = kd.apply_kd_shaping(ranking, kd.KLMode.FORWARD_KL_JS, arm_source_id="arm:x")
    assert shaped.application_order == ()
    assert shaped.probabilities is None
    assert shaped.source_id == "arm:x"


# --------------------------------------------------------------------------- #
# Verifier anchor blend (must never read the trace's own accepted set)
# --------------------------------------------------------------------------- #


def test_verifier_anchor_blend_uses_only_the_disjoint_verifier_table():
    fx = _build(trace_id="t-anchor-1", n_candidates=3, accepted_indices=(1,))
    ids = tuple(a.application_id for a in fx.trace.legal_set.operator_actions)
    shaped = _ranking(fx.trace.trace_id, ids, (0.34, 0.33, 0.33))

    # A verifier trained on an *empty* pool has an all-Laplace-uniform prior;
    # blending with it must not somehow recover this trace's own accepted id.
    empty_verifier = FrequencyBaselineV1(build_frequency_table([]))
    blended = kd._verifier_anchor_blend(shaped, fx.trace, empty_verifier, source_id="arm:x")
    assert set(blended.application_order) == set(ids)
    assert sum(blended.probabilities) == pytest.approx(1.0)


def test_verifier_anchor_blend_weights_approximate_traces_more_heavily():
    fx_exact = _build(trace_id="t-anchor-exact", n_candidates=3, accepted_indices=(0,))
    fx_approx = _build(
        trace_id="t-anchor-approx", n_candidates=3, accepted_indices=(0,), approximate=True
    )
    other = _build(trace_id="t-anchor-train", n_candidates=3, accepted_indices=(2,))
    verifier = FrequencyBaselineV1(build_frequency_table([other.trace]))

    ids_exact = tuple(a.application_id for a in fx_exact.trace.legal_set.operator_actions)
    ids_approx = tuple(a.application_id for a in fx_approx.trace.legal_set.operator_actions)
    shaped_exact = _ranking(fx_exact.trace.trace_id, ids_exact, (0.5, 0.3, 0.2))
    shaped_approx = _ranking(
        fx_approx.trace.trace_id,
        ids_approx,
        (0.5, 0.3, 0.2),
        reliability=RankingReliability.APPROXIMATE,
    )

    blended_exact = kd._verifier_anchor_blend(
        shaped_exact, fx_exact.trace, verifier, source_id="arm:x"
    )
    blended_approx = kd._verifier_anchor_blend(
        shaped_approx, fx_approx.trace, verifier, source_id="arm:x"
    )
    assert "weight=0.3" in blended_exact.reliability_reason
    assert "weight=0.6" in blended_approx.reliability_reason


def test_verifier_anchor_blend_no_op_when_shaped_has_no_probabilities():
    ranking = _ranking("t", ("a", "b"), None)
    fx = _build(trace_id="t-anchor-2", n_candidates=2, accepted_indices=(0,))
    verifier = FrequencyBaselineV1(build_frequency_table([]))
    out = kd._verifier_anchor_blend(ranking, fx.trace, verifier, source_id="arm:x")
    assert out.probabilities is None
    assert out.source_id == "arm:x"


# --------------------------------------------------------------------------- #
# DEFER gate
# --------------------------------------------------------------------------- #


def test_should_defer_on_singleton():
    fx = _build(trace_id="t-defer-singleton", n_candidates=1, accepted_indices=(0,))
    ranking = _ranking(
        fx.trace.trace_id,
        (fx.trace.legal_set.operator_actions[0].application_id,),
        (1.0,),
    )
    defer, reason = kd._should_defer(fx.trace, ranking)
    assert defer is True
    assert "singleton" in reason


def test_should_defer_on_approximate_coverage():
    fx = _build(
        trace_id="t-defer-approx", n_candidates=3, accepted_indices=(0,), approximate=True
    )
    ids = tuple(a.application_id for a in fx.trace.legal_set.operator_actions)
    ranking = _ranking(fx.trace.trace_id, ids, (0.9, 0.05, 0.05))
    defer, reason = kd._should_defer(fx.trace, ranking)
    assert defer is True
    assert "approximate" in reason


def test_should_defer_on_thin_margin():
    fx = _build(trace_id="t-defer-thin", n_candidates=3, accepted_indices=(0,))
    ids = tuple(a.application_id for a in fx.trace.legal_set.operator_actions)
    ranking = _ranking(fx.trace.trace_id, ids, (0.34, 0.33, 0.33))
    defer, reason = kd._should_defer(fx.trace, ranking)
    assert defer is True
    assert "margin" in reason


def test_should_not_defer_when_confident_and_exact():
    fx = _build(trace_id="t-defer-confident", n_candidates=3, accepted_indices=(0,))
    ids = tuple(a.application_id for a in fx.trace.legal_set.operator_actions)
    ranking = _ranking(fx.trace.trace_id, ids, (0.9, 0.06, 0.04))
    defer, reason = kd._should_defer(fx.trace, ranking)
    assert defer is False
    assert reason == "kd_target_confident_and_exact"


# --------------------------------------------------------------------------- #
# build_arm_target
# --------------------------------------------------------------------------- #


def test_build_arm_target_empty_legal_set_every_arm():
    fx = _build(trace_id="t-empty", n_candidates=0, accepted_indices=())
    teacher = SyntheticDescriptorTeacherV1()
    verifier = FrequencyBaselineV1(build_frequency_table([]))
    for arm in kd.ARM_ORDER:
        ranking = kd.build_arm_target(fx.trace, teacher=teacher, verifier=verifier, arm=arm)
        assert ranking.application_order == ()
        assert ranking.probabilities is None


def test_build_arm_target_hard_accepted_set_is_hard_argmax_of_verifier():
    train = _build(trace_id="t-hard-train", n_candidates=3, accepted_indices=(1,))
    verifier = FrequencyBaselineV1(build_frequency_table([train.trace]))
    fx = _build(trace_id="t-hard-eval", n_candidates=3, accepted_indices=(1,))
    teacher = SyntheticDescriptorTeacherV1()
    ranking = kd.build_arm_target(
        fx.trace, teacher=teacher, verifier=verifier, arm=kd.DistillArm.HARD_ACCEPTED_SET
    )
    assert ranking.probabilities is not None
    assert sorted(ranking.probabilities, reverse=True) == list(ranking.probabilities)
    assert ranking.probabilities[0] == 1.0
    assert set(ranking.probabilities[1:]) == {0.0}


def test_build_arm_target_teacher_argmax_is_hard_argmax_of_teacher():
    fx = _build(trace_id="t-targmax", n_candidates=5, accepted_indices=(2,))
    teacher = SyntheticDescriptorTeacherV1()
    verifier = FrequencyBaselineV1(build_frequency_table([]))
    ranking = kd.build_arm_target(
        fx.trace, teacher=teacher, verifier=verifier, arm=kd.DistillArm.TEACHER_ARGMAX
    )
    teacher_ranking = teacher.rank(kd.default_teacher_query(fx.trace))
    assert ranking.application_order[0] == teacher_ranking.application_order[0]
    assert ranking.probabilities[0] == 1.0


def test_build_arm_target_offline_conditional_kd_never_touches_accepted_set():
    """A teacher-only arm ranking must be identical whether or not the trace
    carries accepted labels -- offline_conditional_kd has no verifier signal
    at all, so it cannot leak accepted_application_ids."""
    fx_a = _build(trace_id="t-kd-a", n_candidates=4, accepted_indices=(0,))
    fx_b = _build(trace_id="t-kd-a", n_candidates=4, accepted_indices=(3,))
    teacher = SyntheticDescriptorTeacherV1()
    verifier = FrequencyBaselineV1(build_frequency_table([]))
    ranking_a = kd.build_arm_target(
        fx_a.trace, teacher=teacher, verifier=verifier, arm=kd.DistillArm.OFFLINE_CONDITIONAL_KD
    )
    ranking_b = kd.build_arm_target(
        fx_b.trace, teacher=teacher, verifier=verifier, arm=kd.DistillArm.OFFLINE_CONDITIONAL_KD
    )
    assert ranking_a.application_order == ranking_b.application_order
    assert ranking_a.probabilities == ranking_b.probabilities


def test_build_arm_target_verifier_ranking_kd_stays_within_legal_set():
    train = _build(trace_id="t-vkd-train", n_candidates=4, accepted_indices=(2,))
    verifier = FrequencyBaselineV1(build_frequency_table([train.trace]))
    fx = _build(trace_id="t-vkd-eval", n_candidates=4, accepted_indices=(2,))
    teacher = SyntheticDescriptorTeacherV1()
    ranking = kd.build_arm_target(
        fx.trace, teacher=teacher, verifier=verifier, arm=kd.DistillArm.VERIFIER_RANKING_KD
    )
    legal_ids = {a.application_id for a in fx.trace.legal_set.operator_actions}
    assert set(ranking.application_order) <= legal_ids
    assert sum(ranking.probabilities) == pytest.approx(1.0)


def test_build_arm_target_defer_falls_back_to_compiler_order_on_singleton():
    fx = _build(trace_id="t-defer-arm-singleton", n_candidates=1, accepted_indices=(0,))
    teacher = SyntheticDescriptorTeacherV1()
    verifier = FrequencyBaselineV1(build_frequency_table([]))
    ranking = kd.build_arm_target(
        fx.trace, teacher=teacher, verifier=verifier, arm=kd.DistillArm.DEFER
    )
    fallback = CompilerOrderBaselineV1().rank(fx.trace)
    assert ranking.application_order == fallback.application_order
    assert "DEFER" in ranking.reliability_reason
    assert set(ranking.application_order) <= {
        a.application_id for a in fx.trace.legal_set.operator_actions
    }


def test_build_arm_target_unknown_arm_raises():
    fx = _build(trace_id="t-unknown-arm", n_candidates=2, accepted_indices=(0,))
    teacher = SyntheticDescriptorTeacherV1()
    verifier = FrequencyBaselineV1(build_frequency_table([]))

    class _FakeArm:
        value = "not-a-real-arm"

    with pytest.raises(ValueError, match="unknown arm"):
        kd.build_arm_target(fx.trace, teacher=teacher, verifier=verifier, arm=_FakeArm())


# --------------------------------------------------------------------------- #
# evaluate_decision_changes
# --------------------------------------------------------------------------- #


def _family_map(trace_ids: tuple[str, ...], family: DecisionFamily) -> dict:
    return {tid: family for tid in trace_ids}


def test_evaluate_decision_changes_counts_correct_and_wrong():
    arm_rankings = {
        "t1": _ranking("t1", ("a", "b"), (0.9, 0.1)),  # changed, correct (a accepted)
        "t2": _ranking("t2", ("b", "a"), (0.9, 0.1)),  # changed, wrong (b not accepted)
        "t3": _ranking("t3", ("a", "b"), (0.9, 0.1)),  # identical to baseline
    }
    baseline_rankings = {
        "t1": _ranking("t1", ("b", "a"), (0.9, 0.1)),
        "t2": _ranking("t2", ("a", "b"), (0.9, 0.1)),
        "t3": _ranking("t3", ("a", "b"), (0.9, 0.1)),
    }
    accepted_by_trace = {"t1": frozenset({"a"}), "t2": frozenset({"a"}), "t3": frozenset({"a"})}
    family_by_trace = _family_map(("t1", "t2", "t3"), DecisionFamily.GOLD_SMALL)

    report = kd.evaluate_decision_changes(
        arm_rankings,
        baseline_rankings,
        accepted_by_trace,
        family_by_trace,
        arm=kd.DistillArm.VERIFIER_RANKING_KD,
        baseline_source_id="baseline:x",
    )
    assert report.n_eligible == 3
    assert report.n_changed == 2
    assert report.n_correct_changes == 1
    assert report.n_wrong_changes == 1
    assert report.correct_exceeds_wrong is False
    assert report.prediction_identical_rejected is False


def test_evaluate_decision_changes_excludes_ineligible_families():
    arm_rankings = {"s1": _ranking("s1", ("a",), (1.0,))}
    baseline_rankings = {"s1": _ranking("s1", ("a",), (1.0,))}
    accepted_by_trace = {"s1": frozenset({"a"})}
    family_by_trace = {"s1": DecisionFamily.GOLD_SINGLETON}
    report = kd.evaluate_decision_changes(
        arm_rankings,
        baseline_rankings,
        accepted_by_trace,
        family_by_trace,
        arm=kd.DistillArm.VERIFIER_RANKING_KD,
        baseline_source_id="baseline:x",
    )
    assert report.n_eligible == 0
    assert report.pct_changed["estimate"] is None


def test_evaluate_decision_changes_rejects_prediction_identical_result():
    trace_ids = tuple(f"t{i}" for i in range(20))
    arm_rankings = {tid: _ranking(tid, ("a", "b"), (0.7, 0.3)) for tid in trace_ids}
    baseline_rankings = {tid: _ranking(tid, ("a", "b"), (0.9, 0.1)) for tid in trace_ids}
    accepted_by_trace = {tid: frozenset({"a"}) for tid in trace_ids}
    family_by_trace = _family_map(trace_ids, DecisionFamily.GOLD_SMALL)

    report = kd.evaluate_decision_changes(
        arm_rankings,
        baseline_rankings,
        accepted_by_trace,
        family_by_trace,
        arm=kd.DistillArm.VERIFIER_RANKING_KD,
        baseline_source_id="baseline:x",
    )
    assert report.n_changed == 0
    assert report.prediction_identical_rejected is True
    assert report.meets_change_threshold is False
    assert report.correct_exceeds_wrong is False


def test_evaluate_decision_changes_meets_five_percent_threshold():
    trace_ids = tuple(f"t{i}" for i in range(20))
    arm_rankings = {tid: _ranking(tid, ("a", "b"), (0.9, 0.1)) for tid in trace_ids}
    # Change exactly one of twenty (5%) and make it correct.
    baseline_rankings = {tid: _ranking(tid, ("a", "b"), (0.9, 0.1)) for tid in trace_ids}
    baseline_rankings["t0"] = _ranking("t0", ("b", "a"), (0.9, 0.1))
    accepted_by_trace = {tid: frozenset({"a"}) for tid in trace_ids}
    family_by_trace = _family_map(trace_ids, DecisionFamily.GOLD_SMALL)

    report = kd.evaluate_decision_changes(
        arm_rankings,
        baseline_rankings,
        accepted_by_trace,
        family_by_trace,
        arm=kd.DistillArm.VERIFIER_RANKING_KD,
        baseline_source_id="baseline:x",
    )
    assert report.n_changed == 1
    assert report.n_eligible == 20
    assert report.meets_change_threshold is True
    assert report.correct_exceeds_wrong is True


# --------------------------------------------------------------------------- #
# evaluate_structural_regressions
# --------------------------------------------------------------------------- #


def test_evaluate_structural_regressions_flags_illegal_action():
    fx = _build(trace_id="t-struct-illegal", n_candidates=2, accepted_indices=(0,))
    rankings = {fx.trace.trace_id: _ranking(fx.trace.trace_id, ("not-a-legal-id",), (1.0,))}
    traces_by_id = {fx.trace.trace_id: fx.trace}
    family_by_trace = {fx.trace.trace_id: DecisionFamily.GOLD_SMALL}
    report = kd.evaluate_structural_regressions(
        rankings, traces_by_id, family_by_trace, arm=kd.DistillArm.VERIFIER_RANKING_KD
    )
    assert report.no_illegal_action_introduced is False


def test_evaluate_structural_regressions_flags_altered_singleton():
    fx = _build(trace_id="t-struct-singleton", n_candidates=1, accepted_indices=(0,))
    rankings = {fx.trace.trace_id: _ranking(fx.trace.trace_id, (), None)}
    traces_by_id = {fx.trace.trace_id: fx.trace}
    family_by_trace = {fx.trace.trace_id: DecisionFamily.GOLD_SINGLETON}
    report = kd.evaluate_structural_regressions(
        rankings, traces_by_id, family_by_trace, arm=kd.DistillArm.VERIFIER_RANKING_KD
    )
    assert report.singleton_forced_bypass_preserved is False


def test_evaluate_structural_regressions_passes_for_legal_preserved_singleton():
    fx = _build(trace_id="t-struct-ok", n_candidates=1, accepted_indices=(0,))
    legal_id = fx.trace.legal_set.operator_actions[0].application_id
    rankings = {fx.trace.trace_id: _ranking(fx.trace.trace_id, (legal_id,), (1.0,))}
    traces_by_id = {fx.trace.trace_id: fx.trace}
    family_by_trace = {fx.trace.trace_id: DecisionFamily.GOLD_SINGLETON}
    report = kd.evaluate_structural_regressions(
        rankings, traces_by_id, family_by_trace, arm=kd.DistillArm.VERIFIER_RANKING_KD
    )
    assert report.singleton_forced_bypass_preserved is True
    assert report.no_illegal_action_introduced is True


def test_evaluate_structural_regressions_reports_defer_rate_only_for_defer_arm():
    fx = _build(trace_id="t-struct-defer", n_candidates=2, accepted_indices=(0,))
    ids = tuple(a.application_id for a in fx.trace.legal_set.operator_actions)
    rankings = {fx.trace.trace_id: _ranking(fx.trace.trace_id, ids, (0.6, 0.4))}
    traces_by_id = {fx.trace.trace_id: fx.trace}
    family_by_trace = {fx.trace.trace_id: DecisionFamily.GOLD_SMALL}

    non_defer_report = kd.evaluate_structural_regressions(
        rankings, traces_by_id, family_by_trace, arm=kd.DistillArm.VERIFIER_RANKING_KD
    )
    assert non_defer_report.defer_rate is None

    defer_report = kd.evaluate_structural_regressions(
        rankings,
        traces_by_id,
        family_by_trace,
        arm=kd.DistillArm.DEFER,
        defer_reason_by_trace={fx.trace.trace_id: "kd_target_margin_below_threshold(x)"},
    )
    assert defer_report.defer_rate == 1.0
    assert defer_report.defer_count == 1


# --------------------------------------------------------------------------- #
# End-to-end orchestration
# --------------------------------------------------------------------------- #


def _pool(prefix: str, n: int, *, accepted_index: int = 2) -> list:
    return [
        _build(
            trace_id=f"{prefix}-{i}",
            n_candidates=5,
            accepted_indices=(accepted_index,),
            current_scores_by_index={j: (1.0 if j == accepted_index else 0.1) for j in range(5)},
            supervision=SupervisionSource.GOLD if i % 2 == 0 else SupervisionSource.ON_POLICY,
        ).trace
        for i in range(n)
    ]


def test_compare_operator_verifier_kd_defer_rejects_eval_train_leakage():
    pool = _pool("leak", 4)
    with pytest.raises(ValueError, match="leak into eval_traces"):
        kd.compare_operator_verifier_kd_defer(
            eval_traces=pool,
            verifier_training_traces=[pool[0]],
            held_out_traces_by_seed={0: _pool("leak-held", 4)},
            teacher=SyntheticDescriptorTeacherV1(),
        )


def test_compare_operator_verifier_kd_defer_rejects_held_out_overlap_with_eval():
    eval_pool = _pool("ov-eval", 4)
    with pytest.raises(ValueError, match="leaks into eval_traces"):
        kd.compare_operator_verifier_kd_defer(
            eval_traces=eval_pool,
            verifier_training_traces=_pool("ov-train", 4),
            held_out_traces_by_seed={0: [eval_pool[0]]},
            teacher=SyntheticDescriptorTeacherV1(),
        )


def test_compare_operator_verifier_kd_defer_rejects_held_out_seed_overlap():
    shared = _pool("shared", 2)
    with pytest.raises(ValueError, match="overlaps another seed"):
        kd.compare_operator_verifier_kd_defer(
            eval_traces=_pool("eval2", 4),
            verifier_training_traces=_pool("train2", 4),
            held_out_traces_by_seed={0: shared, 1: shared},
            teacher=SyntheticDescriptorTeacherV1(),
        )


def test_compare_operator_verifier_kd_defer_requires_eval_traces():
    with pytest.raises(ValueError):
        kd.compare_operator_verifier_kd_defer(
            eval_traces=[],
            verifier_training_traces=[],
            held_out_traces_by_seed={0: _pool("x", 2)},
            teacher=SyntheticDescriptorTeacherV1(),
        )


def test_compare_operator_verifier_kd_defer_requires_held_out_pools():
    with pytest.raises(ValueError):
        kd.compare_operator_verifier_kd_defer(
            eval_traces=_pool("x", 2),
            verifier_training_traces=_pool("y", 2),
            held_out_traces_by_seed={},
            teacher=SyntheticDescriptorTeacherV1(),
        )


def test_compare_operator_verifier_kd_defer_end_to_end_report_shape(tmp_path):
    eval_pool = _pool("e2e-eval", 12)
    train_pool = _pool("e2e-train", 12)
    held_out = {
        0: _pool("e2e-held0", 8),
        1: _pool("e2e-held1", 8),
    }
    report = kd.compare_operator_verifier_kd_defer(
        eval_traces=eval_pool,
        verifier_training_traces=train_pool,
        held_out_traces_by_seed=held_out,
        teacher=SyntheticDescriptorTeacherV1(),
        n_bootstrap=200,
    )
    assert report.n_eval_traces == 12
    assert report.n_verifier_training_traces == 12
    assert set(report.held_out_seeds) == {0, 1}
    assert set(report.arms) == {arm.value for arm in kd.ARM_ORDER}
    # hard_accepted_set is the reference control -- never itself gated.
    assert report.arms["hard_accepted_set"].arm_go is False
    assert "reference control arm" in report.arms["hard_accepted_set"].reasons[0]
    # The stop rule never partially passes: go implies at least one of the
    # two advanced arms cleared every gate.
    assert report.stop_rule.go == bool(report.stop_rule.passing_arms)
    if report.stop_rule.go:
        for arm_id in report.stop_rule.passing_arms:
            assert report.arms[arm_id].arm_go is True
    else:
        assert report.stop_rule.reasons

    payload = report.to_dict()
    round_tripped = json.loads(json.dumps(payload))
    assert round_tripped["schema"] == "dsh4-03.v1"
    assert round_tripped["stop_rule"]["recommendation"] in {"go", "no_go_defer"}

    out_path = tmp_path / "report.json"
    kd.write_operator_verifier_kd_defer_report(out_path, report)
    written = json.loads(out_path.read_text(encoding="utf-8"))
    assert written["schema"] == "dsh4-03.v1"


def test_compare_operator_verifier_kd_defer_uses_current_scorer_baseline_by_default():
    eval_pool = _pool("default-baseline", 4)
    report = kd.compare_operator_verifier_kd_defer(
        eval_traces=eval_pool,
        verifier_training_traces=_pool("default-baseline-train", 4),
        held_out_traces_by_seed={0: _pool("default-baseline-held", 4)},
        teacher=SyntheticDescriptorTeacherV1(),
        n_bootstrap=50,
    )
    assert report.nondistilled_baseline_source_id == CurrentScorerBaselineV1().source_id
