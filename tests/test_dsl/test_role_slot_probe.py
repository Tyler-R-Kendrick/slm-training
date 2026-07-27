"""AP-025 (SLM-318): role-slot swap/dropout/truncation probe tests."""

from __future__ import annotations

import pytest

from slm_training.dsl.abstract_plan import AbstractPlanV1, RoleSpanV1
from slm_training.dsl.role_slot_probe import (
    RoleProbeConfig,
    causal_effect_matrix,
    dropout_role,
    run_role_probes,
    slot_roles,
    swap_role,
    truncate_role,
)


def _plan() -> AbstractPlanV1:
    return AbstractPlanV1(
        slot_count=8,
        max_slot_count=8,
        rounds=4,
        role_spans=(
            RoleSpanV1(role="intent", start=0, length=3),
            RoleSpanV1(role="topology", start=3, length=5, max_length=2),
        ),
    )


def test_slot_roles_maps_each_round_to_its_owning_role() -> None:
    plan = _plan()
    assert slot_roles(plan, (0, 3, 2, 7)) == ("intent", "topology", "intent", "topology")


def test_swap_role_exchanges_only_the_declared_roles_rounds() -> None:
    plan = _plan()
    a = (0, 3, 1, 6)  # intent, topology, intent, topology
    b = (2, 4, 0, 5)  # intent, topology, intent, topology
    new_a, new_b = swap_role(plan, a, b, "intent")
    # intent-owned rounds (0, 2) swap; topology-owned rounds (1, 3) untouched.
    assert new_a == (2, 3, 0, 6)
    assert new_b == (0, 4, 1, 5)


def test_swap_role_requires_equal_length_sequences() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="equal-length"):
        swap_role(plan, (0, 1), (0, 1, 2), "intent")


def test_swap_role_rejects_unknown_role() -> None:
    plan = _plan()
    with pytest.raises(KeyError):
        swap_role(plan, (0, 1, 2, 3), (0, 1, 2, 3), "unknown")


def test_dropout_role_replaces_owned_rounds_with_the_role_filler() -> None:
    plan = _plan()
    tokens = (0, 3, 1, 6)  # intent, topology, intent, topology
    dropped = dropout_role(plan, tokens, "intent")
    assert dropped == (0, 3, 0, 6)  # intent filler is slot 0 (the span start)
    dropped_topology = dropout_role(plan, tokens, "topology")
    assert dropped_topology == (0, 3, 1, 3)  # topology filler is slot 3


def test_truncate_role_keeps_only_the_first_n_occupied_rounds() -> None:
    plan = _plan()
    tokens = (3, 4, 5, 6)  # all topology; max_length=2
    truncated = truncate_role(plan, tokens, "topology")  # default: span.max_length
    assert truncated == (3, 4, 3, 3)  # first two kept, rest filled with slot 3


def test_truncate_role_honors_explicit_keep_override() -> None:
    plan = _plan()
    tokens = (3, 4, 5, 6)
    truncated = truncate_role(plan, tokens, "topology", keep=1)
    assert truncated == (3, 3, 3, 3)


def test_truncate_role_rejects_non_positive_keep() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="keep must be"):
        truncate_role(plan, (3, 4, 5, 6), "topology", keep=0)


def test_probes_require_role_factorized_plan() -> None:
    homogeneous = AbstractPlanV1(slot_count=8, max_slot_count=8, rounds=4)
    with pytest.raises(ValueError, match="role_spans"):
        swap_role(homogeneous, (0, 1, 2, 3), (0, 1, 2, 3), "intent")
    with pytest.raises(ValueError, match="role_spans"):
        dropout_role(homogeneous, (0, 1, 2, 3), "intent")
    with pytest.raises(ValueError, match="role_spans"):
        truncate_role(homogeneous, (0, 1, 2, 3), "intent")


class _DistinctCountScorer:
    def score(self, plan_tokens: tuple[int, ...]) -> float:
        return len(set(plan_tokens)) / len(plan_tokens)


def test_run_role_probes_covers_every_role_and_intervention() -> None:
    plan = _plan()
    pairs = [((0, 3, 1, 6), (2, 4, 0, 5))]
    results = run_role_probes(plan, pairs, _DistinctCountScorer())
    seen = {(r.role, r.intervention) for r in results}
    assert seen == {
        (role, intervention)
        for role in ("intent", "topology")
        for intervention in ("swap", "dropout", "truncate")
    }
    # Every result is scored against the same tokens_a baseline.
    baseline = _DistinctCountScorer().score((0, 3, 1, 6))
    assert all(r.baseline_score == baseline for r in results)


def test_run_role_probes_config_restricts_roles() -> None:
    plan = _plan()
    pairs = [((0, 3, 1, 6), (2, 4, 0, 5))]
    results = run_role_probes(
        plan, pairs, _DistinctCountScorer(), config=RoleProbeConfig(roles=("intent",))
    )
    assert {r.role for r in results} == {"intent"}


def test_causal_effect_matrix_aggregates_mean_absolute_delta() -> None:
    plan = _plan()
    pairs = [((0, 3, 1, 6), (2, 4, 0, 5)), ((1, 4, 2, 7), (0, 3, 1, 6))]
    results = run_role_probes(plan, pairs, _DistinctCountScorer())
    matrix = causal_effect_matrix(results)
    assert set(matrix.keys()) == {"intent", "topology"}
    for role_matrix in matrix.values():
        assert set(role_matrix.keys()) == {"swap", "dropout", "truncate"}
        assert all(v >= 0.0 for v in role_matrix.values())
