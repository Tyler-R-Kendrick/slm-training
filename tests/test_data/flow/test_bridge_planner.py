"""Tests for SLM-189 (FFE2-01) bridge planner protocols and engine."""

from __future__ import annotations

from slm_training.data.flow.bridge_planner import (
    BridgePlanV1,
    BridgePlannerResultV1,
    BridgeStepV1,
    CERTIFICATE_FAILURE,
    INVALID_SOURCE,
    REACHED,
    UNKNOWN_BUDGET,
    UNREACHABLE_COMPLETE,
    build_edit_dependency_dag,
    plan_bridge,
    replay_plan,
)
from slm_training.harnesses.experiments.slm188_edit_algebra import (
    CanonicalEdit,
    build_sketch_seed,
    canonicalize,
    plan_edit_sequence,
)

HERO = 'root = Stack([hero], "column")\nhero_title = TextContent(":hero.title")\nhero_body = TextContent(":hero.body")\nhero = Card([hero_title, hero_body])'


def _hero_target() -> str:
    return canonicalize(HERO, validate=True)


def test_status_constants() -> None:
    assert REACHED == "reached"
    assert UNREACHABLE_COMPLETE == "unreachable_complete"
    assert UNKNOWN_BUDGET == "unknown_budget"
    assert INVALID_SOURCE == "invalid_source"
    assert CERTIFICATE_FAILURE == "certificate_failure"


def test_bridge_step_v1_round_trip() -> None:
    step = BridgeStepV1(
        step_index=0,
        edit={"edit_id": "e1", "action": "BindSlotPointer"},
        source_fingerprint="abc",
        target_fingerprint="def",
        transition_certificate={"schema": "TransitionCertificateV1"},
        cost={"edits": 1.0},
        wall_micros=42,
    )
    recovered = BridgeStepV1.from_dict(step.to_dict())
    assert recovered.step_index == step.step_index
    assert recovered.edit == step.edit
    assert recovered.source_fingerprint == step.source_fingerprint
    assert recovered.target_fingerprint == step.target_fingerprint
    assert recovered.transition_certificate == step.transition_certificate
    assert recovered.cost == step.cost
    assert recovered.wall_micros == step.wall_micros


def test_bridge_plan_v1_round_trip() -> None:
    target = _hero_target()
    source = build_sketch_seed(target)
    edits, _ = plan_edit_sequence(source, target)
    result = plan_bridge(source, target, arm="canonical_greedy", source_seed_id="minimal")
    assert result.plan is not None
    plan = result.plan
    recovered = BridgePlanV1.from_dict(plan.to_dict())
    assert recovered.schema == plan.schema
    assert recovered.path_length == plan.path_length
    assert len(recovered.edits) == len(plan.edits)
    assert recovered.termination_status == plan.termination_status


def test_bridge_planner_result_v1_round_trip() -> None:
    target = _hero_target()
    source = build_sketch_seed(target)
    result = plan_bridge(source, target, arm="canonical_greedy", source_seed_id="minimal")
    recovered = BridgePlannerResultV1.from_dict(result.to_dict())
    assert recovered.status == result.status
    assert recovered.replay_ok == result.replay_ok
    assert recovered.plan is not None
    assert recovered.plan.path_length == result.plan.path_length


def test_build_edit_dependency_dag_parent_before_child() -> None:
    insert_hero = CanonicalEdit(
        edit_id="insert-hero",
        action="InsertStatement",
        target_name="hero",
        production="Card",
    )
    insert_child = CanonicalEdit(
        edit_id="insert-child",
        action="InsertChild",
        target_name="hero",
        child_name="title",
    )
    dag = build_edit_dependency_dag([insert_child, insert_hero])
    assert "insert-child" in dag["insert-hero"]


def test_build_edit_dependency_dag_binder_before_reference() -> None:
    bind = CanonicalEdit(
        edit_id="bind-title",
        action="BindSlotPointer",
        target_name="title",
        slot=":hero.title",
    )
    ref = CanonicalEdit(
        edit_id="ref-title",
        action="InsertChild",
        target_name="hero",
        child_name="title",
        dependency_footprint=("title",),
    )
    dag = build_edit_dependency_dag([ref, bind])
    assert "ref-title" in dag["bind-title"]


def test_build_edit_dependency_dag_independent_edits_commute() -> None:
    a = CanonicalEdit(
        edit_id="a",
        action="BindSlotPointer",
        target_name="x",
        slot=":x",
    )
    b = CanonicalEdit(
        edit_id="b",
        action="BindSlotPointer",
        target_name="y",
        slot=":y",
    )
    dag = build_edit_dependency_dag([a, b])
    assert "b" not in dag.get("a", [])
    assert "a" not in dag.get("b", [])


def test_replay_plan_reaches_target_and_fingerprints_match() -> None:
    target = _hero_target()
    source = build_sketch_seed(target)
    edits, _ = plan_edit_sequence(source, target)
    final, fingerprints, ok, detail = replay_plan(source, edits)
    assert ok
    assert detail == "ok"
    assert final is not None
    assert canonicalize(final, validate=False) == canonicalize(target, validate=False)
    assert len(fingerprints) == len(edits) + 1
    assert fingerprints[-1] == __import__("slm_training.dsl.canonicalize", fromlist=["canonical_fingerprint"]).canonical_fingerprint(target)


def test_random_shortest_and_dependency_dag_replay_ok() -> None:
    target = _hero_target()
    source = build_sketch_seed(target)
    for arm in ("random_shortest", "dependency_dag"):
        result = plan_bridge(
            source,
            target,
            arm=arm,
            source_seed_id="minimal",
            rng_seed=1,
        )
        assert result.status == REACHED, f"{arm} did not reach target"
        assert result.replay_ok
        assert result.plan is not None
        assert result.plan.path_length > 0


def test_unknown_arms_return_unknown_budget() -> None:
    target = _hero_target()
    source = build_sketch_seed(target)
    for arm in ("contract_first", "source_adaptive", "solver_guided"):
        result = plan_bridge(source, target, arm=arm, source_seed_id="minimal")
        assert result.status == UNKNOWN_BUDGET
        assert result.plan is None


def test_cost_attribution_keys_present() -> None:
    target = _hero_target()
    source = build_sketch_seed(target)
    result = plan_bridge(source, target, arm="canonical_greedy", source_seed_id="minimal")
    keys = {
        "ast_alignment",
        "candidate_enum",
        "closure_query",
        "path_search",
        "canonicalization",
        "verifier",
        "certificate",
        "memory",
        "cache_hits",
    }
    assert keys.issubset(set(result.cost_attribution))
    assert result.plan is not None
    assert keys.issubset(set(result.plan.cost_vector))
