"""Tests for slm_training.harnesses.experiments.semantic_regret_matrix (SLM-143)."""

from __future__ import annotations

import pytest

from slm_training.harnesses.experiments.semantic_regret_matrix import (
    BoundedCompletionState,
    CompletionSnapshot,
    RegretMetrics,
    SemanticRegretMatrixReport,
    build_fixture_graph,
    build_unreachable_graph,
    compiler_choice_adapter,
    compute_regret_from_trace,
    enumerate_bounded_completions,
    make_semantic_plan_fixture,
    plan_regret_delta,
    selector_candidate_adapter,
    x22_adapter,
)
from slm_training.versioning import UNKNOWN


def test_enumerate_fixture_graph() -> None:
    graph = build_fixture_graph()
    snapshot = enumerate_bounded_completions("start", graph)
    assert snapshot.terminal_values == (0.0, 0.6, 1.0, 1.5, 2.0)
    assert snapshot.accepted_values == (0.6, 1.0, 2.0)
    assert snapshot.pruned_values == (1.5,)
    assert snapshot.coverage_complete is True
    assert snapshot.bound_hit is False


def test_toy_graph_exact_regret_values() -> None:
    graph = build_fixture_graph()
    report = compute_regret_from_trace("start", ["prune_high"], graph)
    assert len(report.snapshots) == 1
    metrics = report.snapshots[0]
    assert metrics.state_id == "start"
    assert metrics.legal_actions == ("accept_good", "accept_ok", "continue", "prune_high")
    assert metrics.chosen_action == "prune_high"
    assert metrics.oracle_best_value == 2.0
    assert metrics.representation_regret == 0.0
    assert metrics.candidate_coverage is True
    assert metrics.acceptable_action_rank == 1
    # prune_high is not accepted, so local/global regret measure the gap to the
    # best accepted completion (2.0). Effective chosen value is 1.5 (terminal).
    assert metrics.local_regret == pytest.approx(0.5)
    assert metrics.global_rank_regret == pytest.approx(0.5)
    # Pruning regret: best pruned (1.5) minus best retained (2.0) = -0.5.
    assert metrics.pruning_regret == pytest.approx(-0.5)
    assert metrics.unknown is False


def test_multiple_acceptable_actions_zero_regret() -> None:
    graph = build_fixture_graph()
    for action in ("accept_good", "accept_ok"):
        report = compute_regret_from_trace("start", [action], graph)
        metrics = report.snapshots[0]
        assert metrics.candidate_coverage is True
        assert metrics.local_regret == 0.0
        assert metrics.unknown is False


def test_unreachable_target_representation_regret() -> None:
    graph = build_unreachable_graph()
    report = compute_regret_from_trace(
        "start",
        ["accept_low"],
        graph,
        oracle_best_value=2.0,
    )
    metrics = report.snapshots[0]
    assert metrics.representation_regret == UNKNOWN
    assert metrics.candidate_coverage is True
    assert metrics.local_regret == 0.0


def test_pruned_action_not_counted_as_scoring_regret() -> None:
    graph = build_fixture_graph()
    report = compute_regret_from_trace("start", ["prune_high"], graph)
    metrics = report.snapshots[0]
    # prune_high has value 1.5, higher than accept_good (1.0), but it is pruned
    # and not accepted, so the scoring regrets still measure against the best
    # accepted completion (2.0), not against the pruned value.
    assert metrics.local_regret == pytest.approx(0.5)
    assert metrics.global_rank_regret == pytest.approx(0.5)
    assert metrics.acceptable_action_rank == 1


def test_unknown_propagates_when_bound_hit() -> None:
    graph = {
        "start": BoundedCompletionState(
            state_id="start",
            actions={
                "a0": ("s1", 0.0, False, None),
            },
        ),
        "s1": BoundedCompletionState(
            state_id="s1",
            actions={
                "a1": ("s2", 0.0, False, None),
            },
        ),
        "s2": BoundedCompletionState(
            state_id="s2",
            actions={
                "a2": ("s3", 0.0, False, None),
            },
        ),
        "s3": BoundedCompletionState(
            state_id="s3",
            actions={
                "term": (None, 1.0, True, None),
            },
        ),
    }
    report = compute_regret_from_trace("start", ["a0", "a1", "a2", "term"], graph, max_nodes=2)
    assert report.unknown is True
    assert any(snapshot.unknown for snapshot in report.snapshots)


def test_plan_factor_substitution_changes_only_plan_terms() -> None:
    graph = build_fixture_graph()
    baseline = compute_regret_from_trace("start", ["prune_high"], graph)
    oracle = compute_regret_from_trace("start", ["continue", "target"], graph)
    delta = plan_regret_delta(baseline, oracle)

    assert delta["baseline_policy"] == ["prune_high"]
    assert delta["oracle_policy"] == ["continue", "target"]
    assert delta["summary"]["n_snapshots"] == 1

    per = delta["per_snapshot"][0]
    assert per["state_id"] == "start"
    # Oracle reduces both local and global-rank regret.
    assert per["local_regret_delta"] == pytest.approx(-0.5)
    assert per["global_rank_regret_delta"] == pytest.approx(-0.5)
    # Representation and pruning regret do not depend on the chosen policy in
    # this fixture, so their deltas are zero.
    assert per["representation_regret_delta"] == 0.0
    assert per["pruning_regret_delta"] == 0.0


def test_adapters_return_unknown() -> None:
    for adapter in (compiler_choice_adapter, x22_adapter, selector_candidate_adapter):
        payload = adapter()
        assert payload["status"] == UNKNOWN
        assert "reason" in payload


def test_make_semantic_plan_fixture_round_trip() -> None:
    plan = make_semantic_plan_fixture(
        provenance="predicted", archetype_id="test", role_ids=["r1"]
    )
    assert plan.identity.provenance == "predicted"
    assert plan.archetype.id == "test"
    assert len(plan.role_slots) == 1
    assert plan.role_slots[0].role_id == "r1"


def test_matrix_report_to_dict() -> None:
    graph = build_fixture_graph()
    report = compute_regret_from_trace("start", ["accept_good"], graph)
    matrix = SemanticRegretMatrixReport(arms={"accept": report})
    data = matrix.to_dict()
    assert data["schema"] == "SemanticRegretReportV1"
    assert "arms" in data
    assert "accept" in data["arms"]


def test_completion_snapshot_to_dict() -> None:
    snapshot = CompletionSnapshot(
        start_state_id="s",
        terminal_values=(1.0, 2.0),
        accepted_values=(2.0,),
        pruned_values=(),
        coverage_complete=True,
        nodes_visited=1,
        bound_hit=False,
    )
    data = snapshot.to_dict()
    assert data["terminal_values"] == [1.0, 2.0]
    assert data["coverage_complete"] is True


def test_regret_metrics_to_dict_round_trip() -> None:
    metrics = RegretMetrics(
        state_id="s",
        legal_actions=("a", "b"),
        chosen_action="a",
        oracle_best_value=1.0,
        representation_regret=0.0,
        candidate_coverage=True,
        acceptable_action_rank=1,
        local_regret=0.0,
        pruning_regret=0.0,
        global_rank_regret=0.0,
        plan_regret=0.0,
        unknown=False,
    )
    data = metrics.to_dict()
    assert data["state_id"] == "s"
    assert data["legal_actions"] == ["a", "b"]


def test_regret_report_to_dict_contains_completion() -> None:
    graph = build_fixture_graph()
    report = compute_regret_from_trace("start", ["accept_good"], graph)
    data = report.to_dict()
    assert data["start_state_id"] == "start"
    assert data["completion_snapshot"]["start_state_id"] == "start"
    assert len(data["snapshots"]) == 1
