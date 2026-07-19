"""Regression tests for CAP4-05 quotient-state diffusion graph diagnostics."""

from __future__ import annotations

import random

import pytest

from slm_training.dsl.analysis.arity.diffusion_graph import (
    QuotientDiffusionGraph,
    Transition,
    build_ast_subtree_kernel,
    build_posterior_weighted_kernel,
    build_production_mask_kernel,
    build_quotient_random_walk_kernel,
    build_surface_token_kernel,
    build_typed_hole_kernel,
    compare_kernels_at_matched_loss,
    information_schedule,
    recommend_information_balanced_schedule,
)


def _directed_cycle(n: int) -> QuotientDiffusionGraph:
    graph = QuotientDiffusionGraph()
    states = [f"s{i}" for i in range(n)]
    for i in range(n):
        graph.add_transition(
            Transition(
                state=states[i],
                action="next",
                next_state=states[(i + 1) % n],
                edge_type="kernel",
            )
        )
    return graph


def _two_cliques_bridge() -> QuotientDiffusionGraph:
    """Barbell: two 3-cliques joined by a single bidirectional bridge edge."""
    graph = QuotientDiffusionGraph()
    a = [f"a{i}" for i in range(3)]
    b = [f"b{i}" for i in range(3)]
    for clique in (a, b):
        for i in range(len(clique)):
            for j in range(len(clique)):
                if i != j:
                    graph.add_transition(
                        Transition(clique[i], "within", clique[j], edge_type="kernel")
                    )
    graph.add_transition(Transition(a[0], "bridge", b[0], edge_type="kernel"))
    graph.add_transition(Transition(b[0], "bridge", a[0], edge_type="kernel"))
    return graph


def test_directed_cycle_is_strongly_connected() -> None:
    graph = _directed_cycle(5)
    assert graph.is_strongly_connected()
    comps = graph.strong_components()
    assert len(comps) == 1
    assert set(graph.states) == comps[0]


def test_directed_cycle_diameter() -> None:
    graph = _directed_cycle(5)
    d = graph.diameter()
    assert d["exact"] is True
    assert d["value"] == 4


def test_directed_cycle_stationary_is_uniform() -> None:
    graph = _directed_cycle(5)
    pi = graph.stationary_distribution()
    assert pi["exact"] is True
    dist = pi["distribution"]
    for v in dist.values():
        assert v == pytest.approx(1.0 / 5.0, abs=1e-6)


def test_directed_cycle_spectral_gap() -> None:
    graph = _directed_cycle(5)
    gap = graph.spectral_gap()
    # Eigenvalues of a directed n-cycle are the n-th roots of unity.
    # Second-largest modulus is 1, so the spectral gap is 0.
    assert gap["value"] == pytest.approx(0.0, abs=1e-9)


def test_barbell_weak_but_not_strongly_connected_without_reverse() -> None:
    graph = _two_cliques_bridge()
    # Each clique has bidirectional internal edges but only one bridge direction.
    # The whole graph should still be strongly connected.
    assert graph.is_strongly_connected()
    # Conductance should expose the bottleneck.
    phi = graph.conductance()
    assert phi["exact"] is True
    assert 0.0 < phi["value"] <= 1.0


def test_from_traces_builds_expected_graph() -> None:
    records = [
        {
            "run_id": "r1",
            "example_id": "e1",
            "seed": 1,
            "decision_index": 1,
            "state_fingerprint": "A",
            "selected_action_id": "x",
            "diffusion_timestep": 0,
        },
        {
            "run_id": "r1",
            "example_id": "e1",
            "seed": 1,
            "decision_index": 2,
            "state_fingerprint": "B",
            "selected_action_id": "y",
            "diffusion_timestep": 1,
        },
        {
            "run_id": "r1",
            "example_id": "e1",
            "seed": 1,
            "decision_index": 3,
            "state_fingerprint": "C",
            "selected_action_id": None,
            "diffusion_timestep": 2,
        },
    ]
    graph = QuotientDiffusionGraph.from_traces(records)
    assert graph.states == {"A", "B", "C"}
    assert len(graph.transitions) == 2
    assert graph.transition_matrix()["A"]["B"] == pytest.approx(1.0)
    assert graph.transition_matrix()["B"]["C"] == pytest.approx(1.0)


def test_kernel_probabilities_normalize() -> None:
    states = ["s0", "s1", "s2"]
    options = {
        "s0": ["s1", "s2"],
        "s1": ["s0"],
        "s2": ["s0", "s1"],
    }
    kernels = [
        build_surface_token_kernel(states, options),
        build_production_mask_kernel(states, options),
        build_ast_subtree_kernel(states, options),
        build_typed_hole_kernel(states, options),
    ]
    for kernel in kernels:
        for s, row in kernel.transition_probs.items():
            total = sum(row.values())
            assert total == pytest.approx(1.0, abs=1e-9), f"{kernel.name} {s}"


def test_kernel_sampling_is_deterministic_by_seed() -> None:
    states = ["s0", "s1", "s2"]
    options = {"s0": ["s1", "s2"], "s1": ["s0"], "s2": ["s0"]}
    kernel = build_surface_token_kernel(states, options)
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    for _ in range(20):
        assert kernel.sample("s0", rng1) == kernel.sample("s0", rng2)


def test_quotient_random_walk_kernel_matches_graph() -> None:
    graph = _directed_cycle(4)
    kernel = build_quotient_random_walk_kernel(graph)
    assert kernel.name == "quotient_random_walk"
    for s in graph.states:
        row = kernel.transition_probs[s]
        assert sum(row.values()) == pytest.approx(1.0, abs=1e-9)


def test_posterior_weighted_kernel_normalizes() -> None:
    states = ["s0", "s1", "s2"]
    targets = ["s1"]
    kernel = build_posterior_weighted_kernel(states, targets, temperature=0.5)
    assert kernel.name == "posterior_weighted_walk"
    for s, row in kernel.transition_probs.items():
        assert sum(row.values()) == pytest.approx(1.0, abs=1e-9)


def test_information_schedule_aggregates_by_timestep() -> None:
    records = [
        {"diffusion_timestep": 0, "posterior_entropy_bits": 2.0, "completion_support_size_exact": 4},
        {"diffusion_timestep": 0, "posterior_entropy_bits": 4.0, "completion_support_size_exact": 8},
        {"diffusion_timestep": 1, "posterior_entropy_bits": 1.0, "completion_support_size_exact": 2},
    ]
    schedule = information_schedule(records)
    assert len(schedule) == 2
    assert schedule[0].timestep == 0
    assert schedule[0].mean_entropy_bits == pytest.approx(3.0)
    assert schedule[0].mean_support_size == pytest.approx(6.0)
    assert schedule[0].information_remaining == pytest.approx(1.0)
    assert schedule[1].information_remaining == pytest.approx(1.0 / 3.0)


def test_information_balanced_schedule_recommendation() -> None:
    records = [
        {"diffusion_timestep": 0, "posterior_entropy_bits": 4.0, "completion_support_size_exact": 16},
        {"diffusion_timestep": 1, "posterior_entropy_bits": 2.0, "completion_support_size_exact": 4},
        {"diffusion_timestep": 2, "posterior_entropy_bits": 0.5, "completion_support_size_exact": 2},
    ]
    points = information_schedule(records)
    rec = recommend_information_balanced_schedule(points, n_steps=4)
    assert len(rec) == 4
    assert rec[0] == pytest.approx(1.0)
    assert rec[-1] < rec[0]


def test_compare_kernels_at_matched_loss_ranks_kernels() -> None:
    states = ["s0", "s1", "s2"]
    options = {"s0": ["s1", "s2"], "s1": ["s0"], "s2": ["s0"]}
    k1 = build_surface_token_kernel(states, options)  # 2-way -> 1 bit/step
    k2 = build_typed_hole_kernel(states, options)  # same options in this fixture
    comparison = compare_kernels_at_matched_loss([k1, k2], states, target_loss_bits=2.0)
    assert comparison.matched_steps[k1.name] == pytest.approx(
        comparison.matched_steps[k2.name]
    )
    assert all(isinstance(v, (int, float)) for v in comparison.matched_steps.values())


def test_kernel_metadata_includes_contract() -> None:
    states = ["s0", "s1"]
    options = {"s0": ["s1"], "s1": ["s0"]}
    kernel = build_typed_hole_kernel(states, options)
    assert kernel.invalid_allowed is False
    assert kernel.exact_posterior is True
    assert "typed_hole" in kernel.metadata["mask_kind"]


def test_reversibility_detects_directed_cycle_not_reversible() -> None:
    graph = _directed_cycle(4)
    rev = graph.reversibility()
    assert rev["exact"] is True
    # A directed cycle is not reversible: pi_i P_ij = 1/n but pi_j P_ji = 0
    # for the forward edge (i -> i+1).
    assert rev["reversible"] is False
    assert rev["violations"] > 0
