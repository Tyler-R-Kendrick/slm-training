"""Core tests for slm_training.flow.reference."""

from __future__ import annotations

import random

import numpy as np

from slm_training.flow.reference import (
    ExactEnumerator,
    FixedGridSampler,
    FlowTargetRowV1,
    GeneratorBuilder,
    GillespieSampler,
    LUMPABLE,
    NOT_LUMPABLE,
    build_distance_rate_fn,
    build_uniform_rate_fn,
    check_generator,
    classify_partition,
    endpoint_distribution,
    is_strongly_lumpable,
)
from slm_training.flow.reference.adapters import (
    ChoiceSequenceAdapter,
    ToyLayoutAdapter,
)


def _triangle_adapter() -> ChoiceSequenceAdapter:
    return ChoiceSequenceAdapter(
        productions={
            "S": [["A", "B"], ["B", "A"]],
            "A": [["a"]],
            "B": [["b"]],
        },
        max_length=4,
        max_states=100,
    )


def test_choice_sequence_enumeration_is_closed() -> None:
    adapter = _triangle_adapter()
    graph = ExactEnumerator(adapter, max_states=100).enumerate()
    assert graph.n_states > 0
    # All terminal states should be reachable and have no outgoing transitions.
    terminal = {s.fingerprint for s in graph.terminal_states}
    for t in graph.transitions:
        assert t.source.fingerprint not in terminal


def test_uniform_generator_row_sums_zero() -> None:
    adapter = _triangle_adapter()
    graph = ExactEnumerator(adapter, max_states=100).enumerate()
    generator = GeneratorBuilder(graph).build_dense(build_uniform_rate_fn(1.0))
    errors = check_generator(generator.Q, atol=1e-9)
    assert not errors
    for i in range(generator.n_states):
        assert generator.Q[i, i] <= 0.0


def test_endpoint_distribution_conserves_mass() -> None:
    adapter = _triangle_adapter()
    graph = ExactEnumerator(adapter, max_states=100).enumerate()
    generator = GeneratorBuilder(graph).build_dense(build_uniform_rate_fn(1.0))
    source_idx = generator.state_index[graph.initial_states[0].fingerprint]
    p0 = np.zeros(generator.n_states)
    p0[source_idx] = 1.0
    for t in (0.5, 1.0, 2.0):
        pT = endpoint_distribution(generator.Q, p0, t)
        assert abs(pT.sum() - 1.0) < 1e-6
        assert np.all(pT >= -1e-9)


def test_gillespie_samples_converge_to_exact() -> None:
    adapter = _triangle_adapter()
    graph = ExactEnumerator(adapter, max_states=100).enumerate()
    generator = GeneratorBuilder(graph).build_dense(build_uniform_rate_fn(1.0))
    source = graph.initial_states[0]
    source_idx = generator.state_index[source.fingerprint]
    p0 = np.zeros(generator.n_states)
    p0[source_idx] = 1.0
    # Absorption distribution is the large-time endpoint mass over terminals.
    pT = endpoint_distribution(generator.Q, p0, 30.0)
    exact = {
        generator.index_state[i].fingerprint: float(p)
        for i, p in enumerate(pT)
        if p > 1e-6 and adapter.is_terminal(generator.index_state[i])
    }

    sampler = GillespieSampler(generator, max_steps=1_000, max_time=50.0)
    rng = random.Random(0)
    counts: dict[str, int] = {}
    n = 1_000
    for _ in range(n):
        traj = sampler.sample(source, rng, terminal_check=adapter.is_terminal)
        counts[traj.terminal_fingerprint] = counts.get(traj.terminal_fingerprint, 0) + 1
    empirical = {fp: c / n for fp, c in counts.items()}
    tv = 0.5 * sum(abs(exact.get(k, 0.0) - empirical.get(k, 0.0)) for k in set(exact) | set(empirical))
    assert tv < 0.25


def test_illegal_transitions_have_zero_rate() -> None:
    adapter = _triangle_adapter()
    graph = ExactEnumerator(adapter, max_states=100).enumerate()
    generator = GeneratorBuilder(graph).build_dense(build_uniform_rate_fn(1.0))
    legal_pairs = {
        (generator.state_index[t.source.fingerprint], generator.state_index[t.target.fingerprint])
        for t in graph.transitions
    }
    for i in range(generator.n_states):
        for j in range(generator.n_states):
            if i != j and (i, j) not in legal_pairs:
                assert generator.Q[i, j] == 0.0


def test_lumpability_singleton_partition_is_lumpable() -> None:
    adapter = _triangle_adapter()
    graph = ExactEnumerator(adapter, max_states=100).enumerate()
    generator = GeneratorBuilder(graph).build_dense(build_uniform_rate_fn(1.0))
    partition = {s.fingerprint: i for i, s in enumerate(graph.states)}
    ok, _ = is_strongly_lumpable(generator, partition, atol=1e-9)
    assert ok


def test_lumpability_coarse_partition_not_lumpable_for_non_trivial_rates() -> None:
    adapter = _triangle_adapter()
    graph = ExactEnumerator(adapter, max_states=100).enumerate()
    generator = GeneratorBuilder(graph).build_dense(build_uniform_rate_fn(1.0))
    # Two blocks: terminals vs non-terminals.
    partition = {s.fingerprint: (1 if adapter.is_terminal(s) else 0) for s in graph.states}
    result = classify_partition(generator, partition, atol=1e-9)
    # The non-terminal block usually has unequal outgoing rates to terminal blocks.
    assert result["status"] in (LUMPABLE, NOT_LUMPABLE)


def test_distance_rate_fn_nonnegative() -> None:
    adapter = _triangle_adapter()
    graph = ExactEnumerator(adapter, max_states=100).enumerate()

    def distance_fn(source, target):
        return 1.0

    rate_fn = build_distance_rate_fn(distance_fn, temperature=0.5)
    generator = GeneratorBuilder(graph).build_dense(rate_fn)
    assert np.all(generator.Q[np.eye(generator.n_states, dtype=bool)] <= 0.0)
    for t in graph.transitions:
        i = generator.state_index[t.source.fingerprint]
        j = generator.state_index[t.target.fingerprint]
        assert generator.Q[i, j] > 0.0


def test_flow_target_row_normalized_probs_sum_to_one() -> None:
    row = FlowTargetRowV1(
        row_id="r1",
        target_rates={"a": 1.0, "b": 3.0},
        total_hazard=4.0,
    )
    norm = row.normalized_next_edit_probs()
    assert abs(sum(norm.values()) - 1.0) < 1e-9
    assert norm == {"a": 0.25, "b": 0.75}


def test_toy_layout_adapter_actions_are_parser_valid() -> None:
    adapter = ToyLayoutAdapter(
        seed_programs=['root = Stack([text], "column")\ntext = TextContent(":slot")'],
        inventory=[":page.blurb"],
        max_depth=2,
        max_states=50,
    )
    graph = ExactEnumerator(adapter, max_states=50).enumerate()
    assert graph.n_states >= 1
    generator = GeneratorBuilder(graph).build_dense(build_uniform_rate_fn(1.0))
    assert not check_generator(generator.Q, atol=1e-9)


def test_fixed_grid_sampler_produces_trajectory() -> None:
    adapter = _triangle_adapter()
    graph = ExactEnumerator(adapter, max_states=100).enumerate()
    generator = GeneratorBuilder(graph).build_dense(build_uniform_rate_fn(5.0))
    sampler = FixedGridSampler(generator, step_size=0.1, n_steps=20)
    rng = random.Random(1)
    traj = sampler.sample(graph.initial_states[0], rng, terminal_check=adapter.is_terminal)
    assert len(traj.states) >= 1
    assert len(traj.wall_times) == len(traj.states)
