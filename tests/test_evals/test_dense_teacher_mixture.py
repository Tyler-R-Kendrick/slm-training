"""Regression tests for SPV2-04 dense teacher mixture snapshots."""

from __future__ import annotations

import pytest

from slm_training.evals.dense_teacher_mixture import (
    AcquisitionPolicy,
    DenseTeacherExampleV1,
    acquire_teacher_labeled_states,
    attach_teacher_distribution,
    build_dense_teacher_snapshot,
    compare_dense_teacher_mixtures,
    compute_acquisition_score,
)
from slm_training.evals.solver_state_supervision import (
    SupervisionSource,
    SolverStateTrainingExampleV1,
)


def _example(
    state_fingerprint: str,
    source: SupervisionSource,
    split: str = "train",
    group: str | None = None,
    acceptable: tuple[int, ...] = (0,),
    verdict: str = "SUPPORTED",
    certified: bool = True,
) -> SolverStateTrainingExampleV1:
    return SolverStateTrainingExampleV1(
        problem_id="p1",
        state_fingerprint=state_fingerprint,
        supervision_source=source,
        legal_actions=tuple({"value": v, "family": f"fam-{v % 2}"} for v in range(4)),
        acceptable_actions=tuple({"value": v, "family": f"fam-{v % 2}"} for v in acceptable),
        support_verdict=verdict,
        cost_to_go=1.0 if verdict == "SUPPORTED" else None,
        cost_observed=verdict == "SUPPORTED",
        split_group_id=group or state_fingerprint,
        split=split,
        lineage_id="lineage-" + state_fingerprint,
        program_family_id="family-1",
        replay_certified=certified,
    )


def _teacher_trace(state_id: str, probs: tuple[float, ...]) -> object:
    """Minimal teacher-trace stand-in."""
    class _Trace:
        pass

    t = _Trace()
    t.state_id = state_id
    t.legal_action_ids = tuple(range(len(probs)))
    t.teacher_logits = None
    t.teacher_probs = probs
    return t


def test_attach_teacher_distribution_aligns_probs() -> None:
    rows = [
        _example("s1", SupervisionSource.ON_POLICY),
        _example("s2", SupervisionSource.ON_POLICY),
    ]
    traces = [
        _teacher_trace("s1", (0.1, 0.2, 0.3, 0.4)),
        _teacher_trace("s2", (0.4, 0.3, 0.2, 0.1)),
    ]
    dense = attach_teacher_distribution(rows, traces)
    assert len(dense) == 2
    assert dense[0].teacher_probs == pytest.approx((0.1, 0.2, 0.3, 0.4))
    assert dense[1].teacher_probs == pytest.approx((0.4, 0.3, 0.2, 0.1))


def test_gold_rows_have_no_teacher_distribution() -> None:
    rows = [_example("s1", SupervisionSource.GOLD)]
    traces = [_teacher_trace("s1", (0.25, 0.25, 0.25, 0.25))]
    dense = attach_teacher_distribution(rows, traces)
    assert dense[0].teacher_probs is None


def test_missing_teacher_trace_leaves_probs_none() -> None:
    rows = [_example("s1", SupervisionSource.ON_POLICY)]
    dense = attach_teacher_distribution(rows, [])
    assert dense[0].teacher_probs is None


def test_teacher_probs_sum_to_one() -> None:
    rows = [_example("s1", SupervisionSource.ON_POLICY)]
    traces = [_teacher_trace("s1", (1.0, 2.0, 3.0, 4.0))]
    dense = attach_teacher_distribution(rows, traces)
    assert dense[0].teacher_probs is not None
    assert sum(dense[0].teacher_probs) == pytest.approx(1.0)
    assert len(dense[0].teacher_probs) == 4


def test_acquisition_uniform_returns_requested_budget() -> None:
    dense = [
        DenseTeacherExampleV1(
            problem_id="p1",
            state_fingerprint=f"s{i}",
            supervision_source=SupervisionSource.ON_POLICY,
            legal_actions=tuple({"value": v} for v in range(4)),
            acceptable_actions=({"value": 0},),
            support_verdict="SUPPORTED",
            cost_to_go=1.0,
            cost_observed=True,
            split_group_id="g1",
            split="train",
            lineage_id="l1",
            program_family_id="f1",
            replay_certified=False,
            teacher_probs=(0.25, 0.25, 0.25, 0.25),
        )
        for i in range(10)
    ]
    selected = acquire_teacher_labeled_states(
        dense, budget=4, policy=AcquisitionPolicy.UNIFORM, seed=7
    )
    assert len(selected) == 4
    assert all(r.teacher_probs is not None for r in selected)


def test_high_divergence_prefers_peaked_distribution() -> None:
    uniform_row = DenseTeacherExampleV1(
        problem_id="p1",
        state_fingerprint="uniform",
        supervision_source=SupervisionSource.ON_POLICY,
        legal_actions=tuple({"value": v} for v in range(4)),
        acceptable_actions=({"value": 0},),
        support_verdict="SUPPORTED",
        cost_to_go=1.0,
        cost_observed=True,
        split_group_id="g1",
        split="train",
        lineage_id="l1",
        program_family_id="f1",
        replay_certified=False,
        teacher_probs=(0.25, 0.25, 0.25, 0.25),
    )
    peaked_row = DenseTeacherExampleV1(
        problem_id="p1",
        state_fingerprint="peaked",
        supervision_source=SupervisionSource.ON_POLICY,
        legal_actions=tuple({"value": v} for v in range(4)),
        acceptable_actions=({"value": 0},),
        support_verdict="SUPPORTED",
        cost_to_go=1.0,
        cost_observed=True,
        split_group_id="g1",
        split="train",
        lineage_id="l1",
        program_family_id="f1",
        replay_certified=False,
        teacher_probs=(0.7, 0.1, 0.1, 0.1),
    )
    assert compute_acquisition_score(peaked_row, AcquisitionPolicy.HIGH_DIVERGENCE) > compute_acquisition_score(
        uniform_row, AcquisitionPolicy.HIGH_DIVERGENCE
    )


def test_snapshot_includes_canonical_arms() -> None:
    rows = [
        _example(f"g{i}", SupervisionSource.GOLD)
        for i in range(20)
    ] + [
        _example(f"o{i}", SupervisionSource.ON_POLICY)
        for i in range(20)
    ]
    traces = [
        _teacher_trace(f"o{i}", (0.4, 0.3, 0.2, 0.1)) for i in range(20)
    ]
    snapshot = build_dense_teacher_snapshot(
        rows,
        traces,
        round_id="test-round",
        decision_budget=32,
        teacher_label_budget=16,
        acquisition_policy=AcquisitionPolicy.UNIFORM,
        seed=0,
    )
    expected = {
        "gold_only",
        "mixed_no_teacher",
        "mixed_teacher_argmax",
        "mixed_teacher_kl",
        "targeted_teacher_kl",
        "on_policy_teacher_kl",
    }
    assert set(snapshot.arms) == expected
    for arm, arm_rows in snapshot.arms.items():
        assert len(arm_rows) <= 32, f"{arm} exceeds decision budget"


def test_snapshot_respects_decision_budget_across_seeds() -> None:
    rows = [
        _example(f"g{i}", SupervisionSource.GOLD) for i in range(30)
    ] + [
        _example(f"o{i}", SupervisionSource.ON_POLICY) for i in range(30)
    ]
    traces = [_teacher_trace(f"o{i}", (0.4, 0.3, 0.2, 0.1)) for i in range(30)]
    comparison = compare_dense_teacher_mixtures(
        rows,
        traces,
        seeds=(0, 1, 2),
        decision_budget=24,
        teacher_label_budget=12,
        acquisition_policy=AcquisitionPolicy.UNIFORM,
        round_id="budget-test",
    )
    assert comparison["seeds"] == [0, 1, 2]
    for snapshot in comparison["snapshots"]:
        for arm, arm_rows in snapshot["arms"].items():
            assert len(arm_rows) <= 24, f"{arm} exceeds budget in {snapshot['round_id']}"


def test_held_out_rows_excluded_from_arms() -> None:
    rows = [
        _example("s1", SupervisionSource.GOLD, split="train", group="g1"),
        _example("s2", SupervisionSource.GOLD, split="test", group="g2"),
        _example("s3", SupervisionSource.ON_POLICY, split="train", group="g3"),
    ]
    traces = [_teacher_trace("s3", (0.4, 0.3, 0.2, 0.1))]
    snapshot = build_dense_teacher_snapshot(
        rows,
        traces,
        round_id="heldout-test",
        decision_budget=10,
        teacher_label_budget=5,
        acquisition_policy=AcquisitionPolicy.UNIFORM,
        seed=0,
    )
    for arm_rows in snapshot.arms.values():
        assert all(r.split != "test" for r in arm_rows)


def test_cross_split_leak_is_rejected() -> None:
    rows = [
        _example("s1", SupervisionSource.GOLD, split="train", group="g1"),
        _example("s1_val", SupervisionSource.GOLD, split="val", group="g1"),
        _example("s2", SupervisionSource.ON_POLICY, split="train", group="g2"),
    ]
    traces = [
        _teacher_trace("s2", (0.4, 0.3, 0.2, 0.1)),
    ]
    snapshot = build_dense_teacher_snapshot(
        rows,
        traces,
        round_id="leak-test",
        decision_budget=10,
        teacher_label_budget=5,
        acquisition_policy=AcquisitionPolicy.UNIFORM,
        seed=0,
    )
    for arm_rows in snapshot.arms.values():
        fingerprints = {r.state_fingerprint for r in arm_rows}
        assert "s1" not in fingerprints
        assert "s2" in fingerprints or arm_rows == []


def test_teacher_argmax_arm_has_single_acceptable_action() -> None:
    rows = [
        _example("g0", SupervisionSource.GOLD, acceptable=(0, 1)),
        _example("o0", SupervisionSource.ON_POLICY, acceptable=(0,)),
    ]
    traces = [_teacher_trace("o0", (0.1, 0.6, 0.2, 0.1))]
    snapshot = build_dense_teacher_snapshot(
        rows,
        traces,
        round_id="argmax-test",
        decision_budget=10,
        teacher_label_budget=5,
        acquisition_policy=AcquisitionPolicy.UNIFORM,
        seed=0,
    )
    argmax_rows = [
        r for r in snapshot.arms["mixed_teacher_argmax"] if r.state_fingerprint == "o0"
    ]
    assert argmax_rows
    assert len(argmax_rows[0].acceptable_actions) == 1
    assert argmax_rows[0].acceptable_actions[0]["value"] == 1
