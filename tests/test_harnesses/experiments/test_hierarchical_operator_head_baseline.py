from __future__ import annotations

from slm_training.harnesses.experiments.hierarchical_operator_head_baseline import (
    BIND_ROLE_OPERATOR_ID,
    SET_VALUE_OPERATOR_ID,
    HierarchicalOperatorHeadArm,
    build_hierarchical_operator_decisions,
    run_hierarchical_operator_head_baseline,
)

_SHAPES = ((1, 2), (2, 1))


def test_decisions_are_built_from_the_live_legal_set_with_typed_gold() -> None:
    decisions = build_hierarchical_operator_decisions(shapes=_SHAPES, repeats=2, seed=1)
    assert len(decisions) == 4
    for case in decisions:
        assert case.legal_operator_ids == {SET_VALUE_OPERATOR_ID, BIND_ROLE_OPERATOR_ID}
        n_values, n_roles = case.shape
        expected_gold = BIND_ROLE_OPERATOR_ID if n_roles > n_values else SET_VALUE_OPERATOR_ID
        assert case.gold_operator_id == expected_gold
    # every decision id is unique even though shapes repeat
    assert len({case.decision_id for case in decisions}) == len(decisions)


def test_ambiguous_shape_is_rejected() -> None:
    import pytest

    with pytest.raises(ValueError, match="ambiguous shape"):
        build_hierarchical_operator_decisions(shapes=((2, 2),), repeats=1, seed=1)


def test_matched_causal_baseline_reproduces_stop_rule_evidence() -> None:
    result = run_hierarchical_operator_head_baseline(
        shapes=_SHAPES,
        train_repeats=3,
        held_out_repeats=2,
        seeds=(7,),
        steps=12,
        learning_rate=0.05,
        d_model=8,
        hidden_dim=8,
    )
    assert result["issue"] == "SLM-383"
    assert result["predecessor_experiment_id"] == "E803"
    assert set(result["arms"]) == {arm.value for arm in HierarchicalOperatorHeadArm}

    # Zero false legal admissions across every arm/seed: the scorer only ever
    # ranks the two real live operator ids built from the legal set.
    assert all(
        run["false_legal_admissions"] == 0
        for values in result["arms"].values()
        for run in values
    )

    # Weight-zero and enabled share architecture/capacity (same raw parameter
    # count) -- weight-zero is a frozen no-op control, not a smaller model.
    hierarchical_parameter_counts = {
        run["parameter_count"]
        for arm in (HierarchicalOperatorHeadArm.WEIGHT_ZERO, HierarchicalOperatorHeadArm.ENABLED)
        for run in result["arms"][arm.value]
    }
    assert len(hierarchical_parameter_counts) == 1
    assert next(iter(hierarchical_parameter_counts)) > 0
    # ...but weight-zero has zero *trainable* parameters (frozen), while the
    # enabled arm actually trains -- the capacity-control distinction.
    assert all(
        run["trainable_parameter_count"] == 0
        for run in result["arms"][HierarchicalOperatorHeadArm.WEIGHT_ZERO.value]
    )
    assert all(
        run["trainable_parameter_count"] > 0
        for run in result["arms"][HierarchicalOperatorHeadArm.ENABLED.value]
    )

    # The acceptance/verdict machinery is well-formed and self-consistent.
    assert result["verdict"] in {"accept", "reject"}
    assert result["accepted"] == (result["verdict"] == "accept")
    assert isinstance(result["acceptance"]["zero_false_legal_admissions"], bool)
    assert isinstance(
        result["acceptance"]["enabled_causally_improves_beyond_token_baseline"], bool
    )


def test_enabled_beats_the_weight_zero_no_op_capacity_control() -> None:
    """The trained head must not be prediction-identical to a frozen null model.

    This is the specific case SLM-383's stop rule calls out ("a converged but
    prediction-identical... head remains default-off and is rejected") applied
    to the *architecture*, not the causal-improvement claim against the token
    baseline (which a separate acceptance gate governs).
    """
    result = run_hierarchical_operator_head_baseline(
        shapes=((1, 3), (3, 1), (2, 3), (3, 2)),
        train_repeats=4,
        held_out_repeats=2,
        seeds=(11, 29),
        steps=24,
        learning_rate=0.05,
        d_model=16,
        hidden_dim=16,
    )
    enabled_runs = result["arms"][HierarchicalOperatorHeadArm.ENABLED.value]
    zero_runs = result["arms"][HierarchicalOperatorHeadArm.WEIGHT_ZERO.value]
    assert all(
        enabled["operator_accuracy"] > zero["operator_accuracy"]
        for enabled, zero in zip(enabled_runs, zero_runs, strict=True)
    )
    assert result["acceptance"]["enabled_beats_weight_zero_capacity_control"] is True
