"""DSH3-32 (SLM-407): operator-inference systems benchmark regression tests.

These are fixture-scale invariant checks, not a benchmark-numbers assertion:
the point is that the singleton-bypass toggle and the compiler-only arm
actually change the *measured* forward count, matching the repo's
non-negotiable "deterministic/singleton bypass outranks any learned score"
invariant (AGENTS.md).
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slm_training.dsl.operators import LegalSetCoverage
from slm_training.evals.operator_systems_benchmark import (
    OPERATOR_SERVING_ARMS,
    UNRUN_ARMS,
    build_operator_fixture,
    run_compiler_only_forced,
    run_operator_systems_benchmark,
    run_serialized_actions,
    run_typed_policy,
    run_x22_tree_edit,
)
from slm_training.evals.operator_systems_benchmark import _build_x22_model


def _singleton_fixture():
    return build_operator_fixture(
        stratum="singleton",
        operator_count=1,
        descriptor_count=3,
        allowed_indices=frozenset({0}),
    )


def _ambiguous_fixture():
    return build_operator_fixture(
        stratum="ambiguous",
        operator_count=2,
        descriptor_count=3,
        allowed_indices=frozenset({0, 1}),
    )


def _partial_fixture():
    return build_operator_fixture(
        stratum="partial",
        operator_count=1,
        descriptor_count=3,
        allowed_indices=frozenset({0}),
        repeated=True,
    )


def test_singleton_stratum_bypass_on_costs_zero_forwards() -> None:
    fixture = _singleton_fixture()
    _, row = run_typed_policy(fixture, cache_state="cold", cached=None, bypass=True)
    assert row.coverage == LegalSetCoverage.COMPLETE.value
    assert row.model_forwards == 0
    assert row.actions_committed == 1
    assert not row.failed


def test_singleton_stratum_bypass_off_forces_one_forward() -> None:
    """The exact forward the singleton-bypass invariant is saving in practice."""
    fixture = _singleton_fixture()
    _, row = run_typed_policy(fixture, cache_state="cold", cached=None, bypass=False)
    assert row.coverage == LegalSetCoverage.COMPLETE.value
    assert row.model_forwards == 1
    assert row.actions_committed == 1


def test_ambiguous_stratum_scores_regardless_of_bypass_toggle() -> None:
    fixture = _ambiguous_fixture()
    _, on_row = run_typed_policy(fixture, cache_state="cold", cached=None, bypass=True)
    _, off_row = run_typed_policy(fixture, cache_state="cold", cached=None, bypass=False)
    assert on_row.operator_row_count >= 2
    assert on_row.model_forwards == 1
    assert off_row.model_forwards == 1


def test_partial_stratum_never_forces_regardless_of_bypass_toggle() -> None:
    """PARTIAL coverage never forces -- the singleton-bypass toggle must not
    override that separate, non-negotiable invariant."""
    fixture = _partial_fixture()
    _, on_row = run_typed_policy(fixture, cache_state="cold", cached=None, bypass=True)
    _, off_row = run_typed_policy(fixture, cache_state="cold", cached=None, bypass=False)
    assert on_row.coverage == LegalSetCoverage.PARTIAL.value
    assert on_row.model_forwards == 0
    assert on_row.actions_committed == 0
    assert off_row.model_forwards == 0
    assert off_row.actions_committed == 0


@pytest.mark.parametrize("fixture_factory", [_singleton_fixture, _ambiguous_fixture])
def test_compiler_only_forced_never_invokes_the_model(fixture_factory) -> None:
    fixture = fixture_factory()
    legal_set, row = run_compiler_only_forced(fixture, cache_state="cold", cached=None)
    assert row.model_forwards == 0
    assert row.dry_runs > 0
    assert row.dry_runs == sum(entry.evaluated_combinations for entry in legal_set.entries)


def test_compiler_only_forced_repeated_slot_never_dry_runs() -> None:
    """Repeated-slot operators short-circuit to UNKNOWN/PARTIAL without ever
    calling ``library.dry_run`` -- zero dry runs is the correct measurement,
    not a missing one (``enumerate_operator_legal_set`` never enumerates an
    unbounded repeated-slot product)."""
    fixture = _partial_fixture()
    legal_set, row = run_compiler_only_forced(fixture, cache_state="cold", cached=None)
    assert row.model_forwards == 0
    assert row.dry_runs == 0
    assert row.coverage == LegalSetCoverage.PARTIAL.value


def test_serialized_actions_never_commits_and_never_invokes_the_model() -> None:
    fixture = _ambiguous_fixture()
    _, row = run_serialized_actions(fixture, cache_state="cold", cached=None)
    assert row.model_forwards == 0
    assert row.actions_committed == 0
    assert row.legal_set_size > 1


def test_warm_cache_state_skips_re_enumeration_dry_runs() -> None:
    fixture = _singleton_fixture()
    legal_set, cold_row = run_compiler_only_forced(fixture, cache_state="cold", cached=None)
    assert cold_row.dry_runs > 0
    _, warm_row = run_compiler_only_forced(fixture, cache_state="warm", cached=legal_set)
    assert warm_row.dry_runs == 0
    assert warm_row.actions_committed == cold_row.actions_committed


def test_x22_arm_records_a_real_positive_forward_count() -> None:
    model = _build_x22_model()
    row = run_x22_tree_edit(model)
    assert row.model_forwards >= 1
    assert row.policy_rows_scored >= row.model_forwards
    assert row.validator_calls == 1


def test_run_operator_systems_benchmark_end_to_end_small() -> None:
    report = run_operator_systems_benchmark(repeats=2)
    assert report.arms_measured == OPERATOR_SERVING_ARMS
    assert report.arms_unrun == UNRUN_ARMS
    assert "full_generation" not in report.arms_measured
    assert report.claim_class == "wiring"
    assert report.ship_eligible is False
    assert len(report.rows) > 0
    arm_ids = {row.arm_id for row in report.rows}
    assert arm_ids == set(OPERATOR_SERVING_ARMS)
    # The singleton stratum must show the bypass toggle actually changing the
    # measured forward count end to end, not just in the unit-level arm call.
    singleton_rows = [
        row
        for row in report.rows
        if row.stratum == "singleton" and row.arm_id.startswith("typed_policy_")
    ]
    assert any(row.arm_id.endswith("_on") and row.model_forwards == 0 for row in singleton_rows)
    assert any(row.arm_id.endswith("_off") and row.model_forwards == 1 for row in singleton_rows)


def test_run_operator_systems_benchmark_rejects_too_few_repeats() -> None:
    with pytest.raises(ValueError):
        run_operator_systems_benchmark(repeats=1)
