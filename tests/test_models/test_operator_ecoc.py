"""Regression tests for cost-aware ternary ECOC abstention under controlled
PARTIAL legal-set coverage (DSH3-29 / SLM-404)."""

from __future__ import annotations

import pytest

from slm_training.dsl.operators import (
    EffectDeltaKind,
    LegalSetCoverage,
    OperatorSupportVerdict,
)
from slm_training.models.action_code_registry import ActionCodeRegistry
from slm_training.models.local_action_head import TernaryECOCHead
from slm_training.models.operator_ecoc_coverage import (
    OperatorEcocCoverageReportV1,
    build_partial_ladder,
    evaluate_ecoc_under_coverage,
    partial_coverage_levels,
)
from slm_training.models.operator_policy_objective import (
    OperatorPolicyObjectiveV1,
    OperatorPolicyTargetV1,
    TargetUtilityLabel,
    build_controlled_partial_fixture,
)
from slm_training.models.operator_policy_view import (
    OperatorActionViewV1,
    validate_no_forbidden_fields,
)
from slm_training.models.semantic_cost import build_operator_action_cost_matrix

HIDDEN_DIM = 8


def _action_view(
    row: int,
    operator_id: str,
    cost: float,
    effect_signature: tuple[EffectDeltaKind, ...],
    locality: str,
) -> OperatorActionViewV1:
    return OperatorActionViewV1(
        row=row,
        operator_id=operator_id,
        operator_version="v1",
        locality=locality,
        cost=cost,
        effect_signature=effect_signature,
        argument_slots=(),
        verdict=OperatorSupportVerdict.SUPPORTED,
        coverage=LegalSetCoverage.COMPLETE,
    )


def _objective(gold: str, others: tuple[str, ...]) -> OperatorPolicyObjectiveV1:
    targets = [
        OperatorPolicyTargetV1(
            gold, OperatorSupportVerdict.SUPPORTED, TargetUtilityLabel.POSITIVE
        )
    ]
    targets.extend(
        OperatorPolicyTargetV1(
            key, OperatorSupportVerdict.SUPPORTED, TargetUtilityLabel.NEGATIVE
        )
        for key in others
    )
    return OperatorPolicyObjectiveV1(
        coverage=LegalSetCoverage.COMPLETE, targets=tuple(targets)
    )


# --- Catastrophic-pair fixture shared by several tests -----------------------
#
# alpha_catastrophic/beta_catastrophic have disjoint two-element effect
# signatures (symmetric difference 4), while delta_safe/gamma_safe are
# identical to each other and share the same locality/cost as everyone else.
# This makes (alpha_catastrophic, beta_catastrophic) the uniquely most
# expensive pair to confuse; every other pair is cheaper.
_ROWS = (
    _action_view(
        0,
        "alpha_catastrophic",
        0.0,
        (EffectDeltaKind.SCOPE, EffectDeltaKind.CARDINALITY),
        "node",
    ),
    _action_view(
        1,
        "beta_catastrophic",
        0.0,
        (EffectDeltaKind.PROPERTY, EffectDeltaKind.TOPOLOGY),
        "node",
    ),
    _action_view(2, "delta_safe", 0.0, (), "node"),
    _action_view(3, "gamma_safe", 0.0, (), "node"),
)


def test_cost_matrix_is_symmetric_deterministic_and_forbidden_field_free() -> None:
    for row in _ROWS:
        # Sanity: the input rows are themselves clean before we derive costs.
        validate_no_forbidden_fields(row.to_dict())
    costs_a = build_operator_action_cost_matrix(_ROWS)
    costs_b = build_operator_action_cost_matrix(_ROWS)
    assert costs_a == costs_b
    for (a, b), value in costs_a.items():
        assert costs_a[(b, a)] == value
    # The matrix is keyed only by operator_id strings with float costs --
    # nothing else from OperatorActionViewV1 ever enters it.
    assert all(isinstance(a, str) and isinstance(b, str) for a, b in costs_a)
    assert all(isinstance(value, float) for value in costs_a.values())
    # The uniquely catastrophic pair costs strictly more than every other pair.
    catastrophic = costs_a[("alpha_catastrophic", "beta_catastrophic")]
    for (a, b), value in costs_a.items():
        if {a, b} != {"alpha_catastrophic", "beta_catastrophic"}:
            assert value < catastrophic


def test_cost_matrix_rejects_duplicate_operator_ids() -> None:
    duplicated = _ROWS + (_ROWS[0],)
    with pytest.raises(ValueError, match="unique"):
        build_operator_action_cost_matrix(duplicated)


@pytest.mark.parametrize(
    "total,expected",
    [
        (1, (1,)),
        (2, (2, 2, 1, 1)),
        (3, (3, 2, 2, 1)),
        (4, (4, 3, 2, 1)),
        (10, (10, 8, 5, 2)),
    ],
)
def test_partial_coverage_levels_matches_expected_ladder(
    total: int, expected: tuple[int, ...]
) -> None:
    # De-duplicated: adjacent equal budgets collapse (e.g. total=2 gives 2,2,1,1
    # before dedup -> (2, 1) after).
    deduped: list[int] = []
    for value in expected:
        if not deduped or deduped[-1] != value:
            deduped.append(value)
    assert partial_coverage_levels(total) == tuple(deduped)


@pytest.mark.parametrize("total", range(1, 13))
def test_partial_coverage_levels_is_monotonic_non_increasing(total: int) -> None:
    levels = partial_coverage_levels(total)
    assert all(a >= b for a, b in zip(levels, levels[1:]))
    assert levels[0] == total
    assert all(1 <= level <= total for level in levels)


def test_partial_coverage_levels_rejects_non_positive_total() -> None:
    with pytest.raises(ValueError):
        partial_coverage_levels(0)


def test_build_partial_ladder_decreasing_budgets_and_shadow_isolation() -> None:
    objective = _objective(
        gold="alpha_catastrophic",
        others=("beta_catastrophic", "delta_safe", "gamma_safe"),
    )
    cost_matrix = build_operator_action_cost_matrix(_ROWS)
    ladder = build_partial_ladder(objective, cost_matrix, seed=0)
    expected_budgets = partial_coverage_levels(len(objective.targets))
    assert tuple(fixture.budget for fixture in ladder) == expected_budgets
    budgets = [fixture.budget for fixture in ladder]
    assert all(a >= b for a, b in zip(budgets, budgets[1:]))
    for fixture in ladder:
        assert fixture.public.coverage is LegalSetCoverage.PARTIAL
        assert fixture.shadow_complete.coverage is LegalSetCoverage.COMPLETE
        assert fixture.shadow_complete == objective
        assert len(fixture.public.targets) == fixture.budget
        # The public projection never reveals anything about the hidden
        # remainder of the shadow domain.
        assert "shadow" not in str(fixture.model_input())
        assert len(fixture.model_input()["candidate_support"]) == fixture.budget


def test_build_partial_ladder_is_deterministic_for_a_fixed_seed() -> None:
    objective = _objective(
        gold="alpha_catastrophic",
        others=("beta_catastrophic", "delta_safe", "gamma_safe"),
    )
    cost_matrix = build_operator_action_cost_matrix(_ROWS)
    ladder_a = build_partial_ladder(objective, cost_matrix, seed=7)
    ladder_b = build_partial_ladder(objective, cost_matrix, seed=7)
    assert tuple(f.truncation_order for f in ladder_a) == tuple(
        f.truncation_order for f in ladder_b
    )


def test_build_partial_ladder_requires_complete_objective() -> None:
    partial_objective = OperatorPolicyObjectiveV1(
        coverage=LegalSetCoverage.PARTIAL,
        targets=(
            OperatorPolicyTargetV1(
                "a", OperatorSupportVerdict.SUPPORTED, TargetUtilityLabel.POSITIVE
            ),
        ),
    )
    with pytest.raises(ValueError, match="COMPLETE"):
        build_partial_ladder(partial_objective, {})


# --- The core DSH3-29 hypothesis-mechanism test ------------------------------


def test_cost_aware_assignment_reduces_cost_weighted_risk_under_partial_coverage() -> None:
    """A cost-aware ECOC codeword assignment must not be riskier than uniform.

    Both heads see the identical PARTIAL fixture (all four actions, but
    projected through the controlled-partial machinery, so ``public.coverage``
    is PARTIAL). The cost-aware assignment separates the uniquely
    catastrophic (alpha, beta) pair to Hamming distance 3, so a 2-trit
    corruption probe toward the decoy is caught as ``detected_error``. The
    uniform assignment happens to leave that pair at the code's minimum
    distance of 2, so the identical probe silently reproduces the decoy's own
    codeword: a real, deterministic ``retained_wrong_action``.
    """
    cost_matrix = build_operator_action_cost_matrix(_ROWS)
    objective = _objective(
        gold="alpha_catastrophic",
        others=("beta_catastrophic", "delta_safe", "gamma_safe"),
    )
    order = tuple(sorted(target.action_key for target in objective.targets))
    fixture = build_controlled_partial_fixture(
        objective, truncation_order=order, budget=len(order)
    )
    assert fixture.public.coverage is LegalSetCoverage.PARTIAL
    ladder = (fixture,)

    head_cost_aware = TernaryECOCHead(
        HIDDEN_DIM, registry=ActionCodeRegistry(), costs=cost_matrix, use_detection=True
    )
    head_uniform = TernaryECOCHead(
        HIDDEN_DIM, registry=ActionCodeRegistry(), costs=None, use_detection=True
    )

    report_cost_aware = evaluate_ecoc_under_coverage(
        head_cost_aware, ladder, cost_matrix, cost_matrix_source="declaration_effect_locality/v1"
    )
    report_uniform = evaluate_ecoc_under_coverage(
        head_uniform, ladder, cost_matrix, cost_matrix_source="uniform"
    )

    level_cost_aware = report_cost_aware.coverage_levels[0]
    level_uniform = report_uniform.coverage_levels[0]

    assert level_cost_aware.detected_errors == 1
    assert level_cost_aware.retained_wrong_actions == 0
    assert level_uniform.retained_wrong_actions == 1
    assert level_uniform.detected_errors == 0

    assert report_cost_aware.overall_cost_weighted_risk == 0.0
    assert report_uniform.overall_cost_weighted_risk == pytest.approx(
        cost_matrix[("alpha_catastrophic", "beta_catastrophic")]
    )
    assert (
        report_cost_aware.overall_cost_weighted_risk
        < report_uniform.overall_cost_weighted_risk
    )


def test_evaluate_ecoc_under_coverage_requires_a_positive_shadow_target() -> None:
    objective = OperatorPolicyObjectiveV1(
        coverage=LegalSetCoverage.COMPLETE,
        targets=(
            OperatorPolicyTargetV1(
                "a", OperatorSupportVerdict.SUPPORTED, TargetUtilityLabel.NEGATIVE
            ),
        ),
    )
    fixture = build_controlled_partial_fixture(objective, truncation_order=("a",), budget=1)
    head = TernaryECOCHead(HIDDEN_DIM, registry=ActionCodeRegistry())
    with pytest.raises(ValueError, match="POSITIVE"):
        evaluate_ecoc_under_coverage(head, (fixture,), {})


def test_operator_ecoc_coverage_report_to_dict_round_trips_expected_fields() -> None:
    cost_matrix = build_operator_action_cost_matrix(_ROWS)
    objective = _objective(
        gold="alpha_catastrophic",
        others=("beta_catastrophic", "delta_safe", "gamma_safe"),
    )
    ladder = build_partial_ladder(objective, cost_matrix, seed=0)
    head = TernaryECOCHead(
        HIDDEN_DIM, registry=ActionCodeRegistry(), costs=cost_matrix, use_detection=True
    )
    report = evaluate_ecoc_under_coverage(
        head, ladder, cost_matrix, cost_matrix_source="declaration_effect_locality/v1"
    )
    assert isinstance(report, OperatorEcocCoverageReportV1)
    payload = report.to_dict()
    assert payload["schema"] == "operator_ecoc_coverage_report/v1"
    assert payload["cost_matrix_source"] == "declaration_effect_locality/v1"
    assert len(payload["coverage_levels"]) == len(ladder)
    assert payload["shadow_set_provenance"]
    for level_payload, fixture in zip(payload["coverage_levels"], ladder):
        assert level_payload["budget"] == fixture.budget
        assert level_payload["defer_count"] == (
            level_payload["model_abstain_count"] + level_payload["compiler_defer_count"]
        )
        total = (
            level_payload["invalid_codewords"]
            + level_payload["defer_count"]
            + level_payload["retained_wrong_actions"]
            + level_payload["detected_errors"]
            + level_payload["correct"]
        )
        assert total == level_payload["total_decisions"]
    assert isinstance(payload["overall_cost_weighted_risk"], float)
    assert isinstance(payload["overall_plain_risk"], float)
