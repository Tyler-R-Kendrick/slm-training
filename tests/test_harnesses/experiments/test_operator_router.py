import pytest

from slm_training.harnesses.experiments.operator_router import (
    ExecutionModeV1,
    ExecutionRouterInputV1,
    ExecutionRouterOutcomeV1,
    choose_execution_mode,
    oracle_execution_mode,
)


def test_router_rejects_future_or_outcome_features() -> None:
    with pytest.raises(ValueError, match="forbidden_field"):
        ExecutionRouterInputV1(
            features={"outcome": 1.0},
            available_modes=frozenset({ExecutionModeV1.PRIMITIVE_OPERATOR}),
        )


def test_router_never_selects_unavailable_or_partial_mode() -> None:
    decision = choose_execution_mode(
        ExecutionRouterInputV1(features={}, available_modes=frozenset()),
        scores={ExecutionModeV1.FULL_REGENERATE: 100.0},
    )
    assert decision.mode is ExecutionModeV1.DEFER
    assert decision.model_forwards == 0


def test_forced_mode_bypasses_scoring() -> None:
    decision = choose_execution_mode(
        ExecutionRouterInputV1(
            features={"legal_candidates": 1.0},
            available_modes=frozenset({ExecutionModeV1.PRIMITIVE_OPERATOR}),
            forced_mode=ExecutionModeV1.PRIMITIVE_OPERATOR,
        ),
        scores={ExecutionModeV1.PRIMITIVE_OPERATOR: -100.0},
    )
    assert (decision.mode, decision.source, decision.model_forwards) == (
        ExecutionModeV1.PRIMITIVE_OPERATOR,
        "forced",
        0,
    )


def test_oracle_is_lexicographic_and_diagnostic_only() -> None:
    assert oracle_execution_mode(
        (
            ExecutionRouterOutcomeV1(ExecutionModeV1.FULL_REGENERATE, True, True, 3, 1, 1),
            ExecutionRouterOutcomeV1(ExecutionModeV1.PRIMITIVE_OPERATOR, True, True, 1, 9, 9),
            ExecutionRouterOutcomeV1(ExecutionModeV1.BULK_OPERATOR, False, True, 0, 0, 0),
        )
    ) is ExecutionModeV1.PRIMITIVE_OPERATOR
