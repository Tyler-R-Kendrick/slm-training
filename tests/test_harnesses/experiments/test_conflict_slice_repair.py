"""Regression tests for the EFS2-03 conflict-slice repair harness."""

from __future__ import annotations

import pytest

from slm_training.harnesses.experiments.conflict_slice_repair import (
    ConflictSliceV1,
    TopologyNode,
    _tree_fingerprint,
    apply_repair_policy,
    compare_repair_policies,
)


def _simple_tree() -> TopologyNode:
    n3 = TopologyNode(
        node_id=3, node_type="LITERAL", parent_id=1, active=True, decision_level=2
    )
    n2 = TopologyNode(
        node_id=2, node_type="LITERAL", parent_id=1, active=True, decision_level=2
    )
    n1 = TopologyNode(
        node_id=1, node_type="SLOT", parent_id=0, children=(n2, n3), active=True,
        decision_level=1,
    )
    n0 = TopologyNode(
        node_id=0, node_type="ROOT", children=(n1,), active=True, decision_level=0
    )
    return n0


def _slice_for(
    tree: TopologyNode,
    failing: tuple[int, ...] = (2,),
    frontier: tuple[int, ...] = (1,),
    protected: tuple[int, ...] = (0,),
    completeness: str = "EXACT",
) -> ConflictSliceV1:
    return ConflictSliceV1(
        conflict_id="test",
        stage="grammar",  # type: ignore[arg-type]
        reason_code="test_reason",
        failing_node_ids=failing,
        dependency_frontier=frontier,
        protected_node_ids=protected,
        completeness_class=completeness,  # type: ignore[arg-type]
        original_state_fingerprint=_tree_fingerprint(tree),
    )


def test_conflict_slice_authorization() -> None:
    exact = ConflictSliceV1(
        conflict_id="c1",
        stage="grammar",  # type: ignore[arg-type]
        reason_code="r",
        failing_node_ids=(1,),
        dependency_frontier=(),
        protected_node_ids=(),
        completeness_class="EXACT",
        original_state_fingerprint="fp",
    )
    assert exact.can_authorize_repair()
    heuristic = ConflictSliceV1(
        conflict_id=exact.conflict_id,
        stage=exact.stage,
        reason_code=exact.reason_code,
        failing_node_ids=exact.failing_node_ids,
        dependency_frontier=exact.dependency_frontier,
        protected_node_ids=exact.protected_node_ids,
        completeness_class="HEURISTIC",  # type: ignore[arg-type]
        original_state_fingerprint=exact.original_state_fingerprint,
    )
    assert not heuristic.can_authorize_repair()


def test_none_policy_leaves_tree_unchanged() -> None:
    tree = _simple_tree()
    slice_ = _slice_for(tree)
    trace = apply_repair_policy(tree, slice_, "none")
    assert trace.repaired_tree.to_dict() == tree.to_dict()
    assert trace.remasked_node_ids == ()


def test_full_remask_drops_all_active_nodes() -> None:
    tree = _simple_tree()
    slice_ = _slice_for(tree, protected=(0,))
    trace = apply_repair_policy(tree, slice_, "full_remask")
    # Root (0) is protected, node 1 (SLOT) and 2,3 (LITERAL) are remasked.
    assert set(trace.remasked_node_ids) == {1, 2, 3}
    repaired_root = next(
        node for node in [trace.repaired_tree]
    )
    assert repaired_root.node_type == "ROOT"


def test_suffix_rollback_remasks_high_levels() -> None:
    tree = _simple_tree()
    slice_ = _slice_for(tree, failing=(3,), protected=(0,))
    trace = apply_repair_policy(tree, slice_, "suffix_rollback")
    # Node 3 is at decision_level 2; node 1 is level 1, node 0 level 0.
    assert 3 in trace.remasked_node_ids
    assert 1 not in trace.remasked_node_ids
    assert 0 not in trace.remasked_node_ids


def test_conflict_slice_touches_failing_and_frontier() -> None:
    tree = _simple_tree()
    slice_ = _slice_for(tree, failing=(2,), frontier=(1,), protected=(0,))
    trace = apply_repair_policy(tree, slice_, "conflict_slice")
    assert set(trace.remasked_node_ids) == {1, 2}


def test_conflict_slice_expanded_includes_parent_unless_protected() -> None:
    tree = _simple_tree()
    slice_ = _slice_for(tree, failing=(2,), frontier=(1,), protected=(0,))
    trace = apply_repair_policy(tree, slice_, "conflict_slice_expanded")
    # Parent 0 is protected and is excluded from remasking before application.
    assert 0 not in trace.remasked_node_ids
    assert 1 in trace.remasked_node_ids
    assert 2 in trace.remasked_node_ids
    assert trace.protected_mutations == 0


def test_protected_nodes_are_never_remasked() -> None:
    tree = _simple_tree()
    slice_ = _slice_for(tree, failing=(2,), frontier=(0,), protected=(0,))
    trace = apply_repair_policy(tree, slice_, "conflict_slice")
    # Protected nodes are excluded before application, so they are never touched.
    assert 0 not in trace.remasked_node_ids
    assert trace.protected_mutations == 0


def test_heuristic_slice_refuses_localized_repair_without_raising() -> None:
    tree = _simple_tree()
    slice_ = _slice_for(tree, completeness="HEURISTIC")
    trace = apply_repair_policy(tree, slice_, "conflict_slice")
    assert trace.remasked_node_ids == ()
    assert not trace.recovered
    assert trace.repeated_conflict


def test_fingerprint_mismatch_fails_closed() -> None:
    tree = _simple_tree()
    slice_ = ConflictSliceV1(
        conflict_id="mismatch",
        stage="grammar",  # type: ignore[arg-type]
        reason_code="r",
        failing_node_ids=(2,),
        dependency_frontier=(),
        protected_node_ids=(),
        completeness_class="EXACT",
        original_state_fingerprint="deadbeef",
    )
    with pytest.raises(ValueError, match="state fingerprint mismatch"):
        apply_repair_policy(tree, slice_, "conflict_slice")


def test_compare_policies_returns_all_policies() -> None:
    tree = _simple_tree()
    slice_ = _slice_for(tree, protected=(0,))
    outcomes = compare_repair_policies(tree, slice_, seeds=(0, 1))
    assert set(outcomes.keys()) == {
        "none",
        "suffix_rollback",
        "full_remask",
        "conflict_slice",
        "conflict_slice_expanded",
    }
    # EXACT conflict_slice should recover in fixture.
    assert outcomes["conflict_slice"].recovery_rate == 1.0
    assert outcomes["none"].recovery_rate == 0.0


def test_deterministic_across_seeds() -> None:
    tree = _simple_tree()
    slice_ = _slice_for(tree, protected=(0,))
    o1 = compare_repair_policies(tree, slice_, seeds=(0, 1, 2))
    o2 = compare_repair_policies(tree, slice_, seeds=(0, 1, 2))
    for policy in o1:
        assert o1[policy].recovery_rate == o2[policy].recovery_rate
        assert o1[policy].mean_remasked_nodes == o2[policy].mean_remasked_nodes


def test_clone_preserves_tree() -> None:
    from slm_training.harnesses.experiments.conflict_slice_repair import (
        _with_replaced_node,
    )

    tree = _simple_tree()
    clone = tree.clone()
    assert clone.to_dict() == tree.to_dict()
    # Rebuilding the clone should not affect the original.
    modified = _with_replaced_node(
        clone,
        1,
        TopologyNode(
            node_id=clone.children[0].node_id,
            node_type=clone.children[0].node_type,
            parent_id=clone.children[0].parent_id,
            children=(),
            active=clone.children[0].active,
            protected=clone.children[0].protected,
            certified=clone.children[0].certified,
            decision_level=clone.children[0].decision_level,
        ),
    )
    assert len(modified.children[0].children) == 0
    assert len(tree.children[0].children) == 2


def test_max_remask_nodes_budget() -> None:
    tree = _simple_tree()
    slice_ = _slice_for(tree, failing=(2, 3), frontier=(1,), protected=(0,))
    trace = apply_repair_policy(
        tree, slice_, "conflict_slice", max_remask_nodes=1
    )
    assert len(trace.remasked_node_ids) == 1
