"""SLM-400 regression tests for coverage-aware operator supervision."""

from __future__ import annotations

import pytest
import torch

from slm_training.dsl.operators.legal_set import (
    LegalSetCoverage,
    OperatorSupportVerdict,
)
from slm_training.models.operator_policy_objective import (
    DeferFallbackAttributionV1,
    ObjectiveLossKind,
    OperatorPolicyObjectiveV1,
    OperatorPolicyRouteV1,
    OperatorPolicyTargetV1,
    TargetUtilityLabel,
    build_controlled_partial_fixture,
    false_hard_eliminations,
    operator_policy_loss,
    operator_policy_report,
    route_operator_policy_inference,
)


def _complete() -> OperatorPolicyObjectiveV1:
    return OperatorPolicyObjectiveV1(
        coverage=LegalSetCoverage.COMPLETE,
        targets=(
            OperatorPolicyTargetV1(
                "a", OperatorSupportVerdict.SUPPORTED, TargetUtilityLabel.POSITIVE
            ),
            OperatorPolicyTargetV1(
                "b", OperatorSupportVerdict.SUPPORTED, TargetUtilityLabel.NEGATIVE
            ),
            OperatorPolicyTargetV1(
                "c", OperatorSupportVerdict.UNSUPPORTED, TargetUtilityLabel.UNKNOWN
            ),
        ),
    )


def test_unknown_never_enters_a_negative_denominator_or_model_input() -> None:
    partial = OperatorPolicyObjectiveV1(
        coverage=LegalSetCoverage.PARTIAL,
        targets=(
            OperatorPolicyTargetV1(
                "a", OperatorSupportVerdict.SUPPORTED, TargetUtilityLabel.POSITIVE
            ),
            OperatorPolicyTargetV1(
                "b", OperatorSupportVerdict.UNKNOWN, TargetUtilityLabel.UNKNOWN
            ),
        ),
    )
    for kind in (ObjectiveLossKind.KNOWN_SUPPORTED_CE, ObjectiveLossKind.PU_RISK):
        logits = torch.tensor([0.0, 100.0], requires_grad=True)
        loss, metrics = operator_policy_loss(logits, partial, kind=kind)
        loss.backward()
        assert metrics["denominator"] == 1.0
        assert logits.grad is not None and logits.grad.tolist()[1] == 0.0
    assert "utility" not in str(partial.model_input())
    with pytest.raises(ValueError, match="UNKNOWN compiler support"):
        OperatorPolicyTargetV1(
            "bad", OperatorSupportVerdict.UNKNOWN, TargetUtilityLabel.NEGATIVE
        )


def test_controlled_partial_keeps_shadow_hidden_and_never_forces_singleton() -> None:
    fixture = build_controlled_partial_fixture(
        _complete(), truncation_order=("a", "b", "c"), budget=1
    )
    permuted = build_controlled_partial_fixture(
        _complete(), truncation_order=("c", "b", "a"), budget=1
    )
    assert fixture.public.model_input() == fixture.model_input()
    assert "shadow" not in str(fixture.model_input())
    assert (
        route_operator_policy_inference(fixture.public)
        is OperatorPolicyRouteV1.PARTIAL_WITNESSED
    )
    assert (
        route_operator_policy_inference(permuted.public)
        is OperatorPolicyRouteV1.PARTIAL_UNKNOWN
    )
    assert fixture.shadow_complete == permuted.shadow_complete
    assert false_hard_eliminations(fixture) == 0


def test_routes_complete_partial_timeout_and_budget_to_the_safe_path() -> None:
    singleton = OperatorPolicyObjectiveV1(
        coverage=LegalSetCoverage.COMPLETE,
        targets=(
            OperatorPolicyTargetV1(
                "a", OperatorSupportVerdict.SUPPORTED, TargetUtilityLabel.POSITIVE
            ),
        ),
    )
    unknown = OperatorPolicyObjectiveV1(coverage=LegalSetCoverage.PARTIAL, targets=())
    assert (
        route_operator_policy_inference(singleton)
        is OperatorPolicyRouteV1.COMPLETE_SINGLETON
    )
    assert (
        route_operator_policy_inference(_complete())
        is OperatorPolicyRouteV1.COMPLETE_AMBIGUOUS
    )
    assert (
        route_operator_policy_inference(unknown)
        is OperatorPolicyRouteV1.PARTIAL_UNKNOWN
    )
    assert (
        route_operator_policy_inference(singleton, timed_out=True)
        is OperatorPolicyRouteV1.TIMEOUT
    )
    assert (
        route_operator_policy_inference(singleton, budget_exhausted=True)
        is OperatorPolicyRouteV1.BUDGET_EXHAUSTED
    )


def test_pu_and_defer_are_safe_comparators_with_exact_fallback_attribution() -> None:
    objective = _complete()
    logits = torch.tensor([0.1, 0.3, 99.0], requires_grad=True)
    pu_loss, pu = operator_policy_loss(
        logits, objective, kind=ObjectiveLossKind.PU_RISK
    )
    defer_loss, defer = operator_policy_loss(
        logits, objective, kind=ObjectiveLossKind.DEFER, defer_logit=torch.tensor([0.0])
    )
    (pu_loss + defer_loss).backward()
    assert pu["denominator"] == 2.0
    assert defer["denominator"] == 3.0
    partial = build_controlled_partial_fixture(
        objective, truncation_order=("a", "b", "c"), budget=1
    )
    report = operator_policy_report(
        [(objective, logits.detach())],
        partial_fixtures=(partial,),
        defer_attributions=(
            DeferFallbackAttributionV1(
                OperatorPolicyRouteV1.PARTIAL_WITNESSED, True, True
            ),
            DeferFallbackAttributionV1(
                OperatorPolicyRouteV1.PARTIAL_UNKNOWN, True, False
            ),
        ),
    )
    assert report["credited_defers"] == 1.0
    assert report["uncredited_defers"] == 1.0
    assert report["false_hard_eliminations"] == 0.0
    assert report["partial_singleton_bypasses"] == 0.0


def test_known_supported_ce_accepts_multiple_positives() -> None:
    objective = OperatorPolicyObjectiveV1(
        coverage=LegalSetCoverage.COMPLETE,
        targets=(
            OperatorPolicyTargetV1(
                "a", OperatorSupportVerdict.SUPPORTED, TargetUtilityLabel.POSITIVE
            ),
            OperatorPolicyTargetV1(
                "b", OperatorSupportVerdict.SUPPORTED, TargetUtilityLabel.POSITIVE
            ),
            OperatorPolicyTargetV1(
                "c", OperatorSupportVerdict.SUPPORTED, TargetUtilityLabel.NEGATIVE
            ),
        ),
    )
    loss, metrics = operator_policy_loss(
        torch.tensor([0.0, 0.0, 0.0]),
        objective,
        kind=ObjectiveLossKind.KNOWN_SUPPORTED_CE,
    )
    assert loss.item() == pytest.approx(-torch.log(torch.tensor(2 / 3)).item())
    assert metrics["denominator"] == 3.0
