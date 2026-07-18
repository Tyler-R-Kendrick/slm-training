"""Tests for CAP1-03 task-confusability graph and neural state quotient."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.dsl.analysis.arity.task_quotient import (
    AlignedActionRecord,
    ConfusabilityGraph,
    TaskDistortionSpec,
    analyze_task_quotient,
    build_state_profiles,
    capacity_feasible,
    color_graph,
)


def _records_for_state(
    fingerprint: str, actions: dict[str, int], semantic: str | None = None
) -> list[AlignedActionRecord]:
    records: list[AlignedActionRecord] = []
    total = sum(actions.values())
    for action_id, count in actions.items():
        records.append(
            AlignedActionRecord(
                state_fingerprint=fingerprint,
                action_id=action_id,
                aligned_family=action_id,
                probability=count / total,
                semantic_fingerprint=semantic,
            )
        )
    return records


def test_build_state_profiles_aggregates_counts() -> None:
    records = _records_for_state("s1", {"a": 3, "b": 1})
    profiles = build_state_profiles(records)
    assert len(profiles) == 1
    profile = profiles["s1"]
    assert profile.action_distribution == {"a": 0.75, "b": 0.25}


def test_mutually_confusable_states_yield_chi_equals_m() -> None:
    # Three states with disjoint action distributions: all pairs confusable.
    records: list[AlignedActionRecord] = []
    for i, actions in enumerate([{"a": 1}, {"b": 1}, {"c": 1}]):
        records.extend(_records_for_state(f"s{i}", actions))
    spec = TaskDistortionSpec(spec_id="test", policy_metric="tv", policy_tolerance=0.0)
    report = analyze_task_quotient(records, spec, refine=False)
    assert report.state_count == 3
    assert report.edge_count == 3
    assert report.coloring.num_colors == 3
    assert report.coloring.exact is True
    assert report.coloring.verify(report.graph) == []


def test_equivalent_states_share_color() -> None:
    # Two states with identical distributions should share a color.
    records: list[AlignedActionRecord] = []
    for fp in ("s1", "s2"):
        records.extend(_records_for_state(fp, {"a": 1, "b": 1}))
    spec = TaskDistortionSpec(spec_id="test", policy_metric="tv", policy_tolerance=0.5)
    report = analyze_task_quotient(records, spec, refine=False)
    assert report.coloring.num_colors == 1
    assert report.coloring.verify(report.graph) == []


def test_hard_forbidden_confusion_creates_edge() -> None:
    records: list[AlignedActionRecord] = []
    for fp in ("s1", "s2"):
        records.extend(_records_for_state(fp, {"a": 1}, semantic="forbidden"))
    spec = TaskDistortionSpec(
        spec_id="test",
        policy_metric="tv",
        policy_tolerance=1.0,
        hard_forbidden_confusions=("forbidden",),
    )
    report = analyze_task_quotient(records, spec, refine=False)
    assert report.edge_count == 1


def test_refine_splits_harmful_merge() -> None:
    # Two states look identical on average (TV=0.8 under aggregate) but are
    # separated by a stricter per-pair regret check during refinement.
    records: list[AlignedActionRecord] = []
    records.extend(_records_for_state("s1", {"a": 9, "b": 1}))
    records.extend(_records_for_state("s2", {"a": 1, "b": 9}))
    spec = TaskDistortionSpec(
        spec_id="test",
        policy_metric="tv",
        policy_tolerance=0.5,
        average_tolerance=0.9,
    )
    report = analyze_task_quotient(records, spec, refine=True, max_refinement_iterations=4)
    # After refinement, the within-color violation should be split.
    assert report.coloring.verify(report.graph) == []
    assert len(report.counterexamples) >= 1


def test_coloring_is_exact_for_small_graphs() -> None:
    graph = ConfusabilityGraph(
        vertices={"a", "b", "c", "d"},
        edges={frozenset({"a", "b"}), frozenset({"b", "c"}), frozenset({"c", "d"})},
    )
    coloring = color_graph(graph, exact_max_vertices=128)
    assert coloring.exact is True
    assert coloring.num_colors == 2
    assert coloring.verify(graph) == []


def test_capacity_feasibility() -> None:
    feasible = capacity_feasible(16, [(2, 4), (3, 3), (4, 2)])
    assert feasible[(2, 4)] is True  # 2**4 == 16
    assert feasible[(3, 3)] is True  # 3**3 == 27 >= 16
    assert feasible[(4, 2)] is True  # 4**2 == 16


def test_analyze_task_quotient_report_has_required_keys() -> None:
    records: list[AlignedActionRecord] = []
    for i, actions in enumerate([{"a": 1}, {"b": 1}]):
        records.extend(_records_for_state(f"s{i}", actions))
    spec = TaskDistortionSpec(spec_id="test", policy_metric="tv", policy_tolerance=0.0)
    report = analyze_task_quotient(records, spec)
    data = report.to_dict()
    assert set(data) >= {
        "spec",
        "state_count",
        "edge_count",
        "density",
        "coloring",
        "class_size_histogram",
        "counterexamples",
        "capacity_feasibility",
        "estimated",
    }


def test_cli_smoke(tmp_path: Path) -> None:
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "state_fingerprint": f"s{i}",
                    "action_id": chr(ord("a") + i),
                }
            )
            for i in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    md = tmp_path / "report.md"
    from scripts.analyze_task_quotient import main

    rc = main(
        [
            "--records",
            str(records_path),
            "--out",
            str(out),
            "--markdown-out",
            str(md),
            "--policy-metric",
            "tv",
            "--policy-tolerance",
            "0.0",
        ]
    )
    assert rc == 0
    assert out.exists()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["state_count"] == 2
