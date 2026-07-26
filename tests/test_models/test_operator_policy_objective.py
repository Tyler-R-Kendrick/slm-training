"""Tests for the DSH3-25 orthogonal support/utility objective and routing."""

from __future__ import annotations

import pytest
import torch

from slm_training.dsl.operators import (
    BindingPhase,
    LegalSetCoverage,
    OperatorSupportVerdict,
    RefKind,
)
from slm_training.models.operator_policy_objective import (
    InferenceRoute,
    ObjectiveArm,
    OperatorPolicyObjectiveV1,
    build_operator_policy_objectives,
    classify_route,
    defer_abstention_loss,
    known_supported_ce_loss,
    positive_unlabeled_risk_loss,
    run_matched_objective_fixture,
)
from slm_training.models.operator_policy_view import (
    OperatorActionViewV1,
    OperatorArgumentSlotViewV1,
    OperatorPolicyInputV1,
    ReferenceModelViewV1,
)


def _reference_row(row: int) -> ReferenceModelViewV1:
    return ReferenceModelViewV1(
        row=row,
        ref_kind=RefKind.VALUE,
        value_type="openui.string",
        compiler_facts=(),
        has_parent=False,
        parent_row=None,
        relative_position=None,
    )


def _action(
    row: int,
    *,
    candidate_rows: tuple[int, ...],
    verdict: OperatorSupportVerdict = OperatorSupportVerdict.SUPPORTED,
    coverage: LegalSetCoverage = LegalSetCoverage.COMPLETE,
    domain_complete: bool = True,
) -> OperatorActionViewV1:
    return OperatorActionViewV1(
        row=row,
        operator_id=f"openui.fixture_op_{row}",
        operator_version="v1",
        locality="node",
        cost=1.0,
        effect_signature=(),
        argument_slots=(
            OperatorArgumentSlotViewV1(
                slot_id="value",
                ref_kind=RefKind.VALUE,
                binding_phase=BindingPhase.APPLICATION,
                required=True,
                repeated=False,
                candidate_rows=candidate_rows,
                domain_complete=domain_complete,
            ),
        ),
        verdict=verdict,
        coverage=coverage,
    )


def test_classify_route_priority_order() -> None:
    assert (
        classify_route(
            verdict=OperatorSupportVerdict.UNSUPPORTED,
            coverage=LegalSetCoverage.COMPLETE,
            domain_complete=True,
            has_positive_witness=True,
            is_forced_singleton=True,
        )
        is InferenceRoute.UNSUPPORTED
    ), "unsupported outranks even a forced singleton"
    assert (
        classify_route(
            verdict=OperatorSupportVerdict.SUPPORTED,
            coverage=LegalSetCoverage.PARTIAL,
            domain_complete=False,
            has_positive_witness=False,
            is_forced_singleton=True,
        )
        is InferenceRoute.COMPLETE_SINGLETON
    ), "forced singleton outranks partial coverage"
    assert (
        classify_route(
            verdict=OperatorSupportVerdict.SUPPORTED,
            coverage=LegalSetCoverage.COMPLETE,
            domain_complete=True,
            has_positive_witness=False,
            is_forced_singleton=False,
        )
        is InferenceRoute.COMPLETE_AMBIGUOUS
    )
    assert (
        classify_route(
            verdict=OperatorSupportVerdict.SUPPORTED,
            coverage=LegalSetCoverage.PARTIAL,
            domain_complete=False,
            has_positive_witness=True,
            is_forced_singleton=False,
        )
        is InferenceRoute.PARTIAL_WITNESSED
    )
    assert (
        classify_route(
            verdict=OperatorSupportVerdict.SUPPORTED,
            coverage=LegalSetCoverage.PARTIAL,
            domain_complete=False,
            has_positive_witness=False,
            is_forced_singleton=False,
        )
        is InferenceRoute.PARTIAL_UNKNOWN
    )


def test_objective_derives_unknown_as_the_exhaustive_complement() -> None:
    objective = OperatorPolicyObjectiveV1(
        action_row=0,
        slot_id="value",
        compiler_support=OperatorSupportVerdict.SUPPORTED,
        coverage=LegalSetCoverage.PARTIAL,
        domain_complete=False,
        candidate_rows=(0, 1, 2, 3),
        positive_candidate_rows=(1,),
        route=InferenceRoute.PARTIAL_WITNESSED,
    )
    assert objective.unknown_candidate_rows == (0, 2, 3)
    payload = objective.to_dict()
    assert sorted(payload["positive_candidate_rows"] + payload["unknown_candidate_rows"]) == [
        0,
        1,
        2,
        3,
    ]


@pytest.mark.parametrize(
    "route,positives",
    [(InferenceRoute.PARTIAL_UNKNOWN, (1,)), (InferenceRoute.PARTIAL_WITNESSED, ())],
)
def test_objective_rejects_route_label_mismatches(route, positives) -> None:
    with pytest.raises(ValueError):
        OperatorPolicyObjectiveV1(
            action_row=0,
            slot_id="value",
            compiler_support=OperatorSupportVerdict.SUPPORTED,
            coverage=LegalSetCoverage.PARTIAL,
            domain_complete=False,
            candidate_rows=(0, 1),
            positive_candidate_rows=positives,
            route=route,
        )


def test_build_operator_policy_objectives_from_policy_input() -> None:
    reference_rows = tuple(_reference_row(row) for row in range(4))
    action_rows = (
        _action(0, candidate_rows=(0, 1), coverage=LegalSetCoverage.COMPLETE),
        _action(
            1,
            candidate_rows=(2, 3),
            coverage=LegalSetCoverage.PARTIAL,
            domain_complete=False,
        ),
    )
    policy_input = OperatorPolicyInputV1(
        reference_rows=reference_rows,
        action_rows=action_rows,
        ordinary_action_count=0,
        coverage=LegalSetCoverage.PARTIAL,
    )
    objectives = build_operator_policy_objectives(policy_input, {0: (1,)})
    by_action = {objective.action_row: objective for objective in objectives}
    assert by_action[0].route is InferenceRoute.COMPLETE_AMBIGUOUS
    assert by_action[0].positive_candidate_rows == (1,)
    assert by_action[1].route is InferenceRoute.PARTIAL_UNKNOWN
    assert by_action[1].positive_candidate_rows == ()
    assert by_action[1].unknown_candidate_rows == (2, 3)


def test_known_supported_ce_and_pu_risk_disagree_on_confident_unknowns() -> None:
    objective = OperatorPolicyObjectiveV1(
        action_row=0,
        slot_id="value",
        compiler_support=OperatorSupportVerdict.SUPPORTED,
        coverage=LegalSetCoverage.PARTIAL,
        domain_complete=False,
        candidate_rows=(0, 1, 2),
        positive_candidate_rows=(0,),
        route=InferenceRoute.PARTIAL_WITNESSED,
    )
    # Unknown candidates (1, 2) score very high -- a confident guess a naive
    # CE arm must punish as if certainly wrong, but PU risk should not treat
    # the same way, since they are unlabeled, not certified negative.
    logits = torch.tensor([0.0, 5.0, 5.0])
    ce_loss = known_supported_ce_loss(logits, objective)
    pu_loss = positive_unlabeled_risk_loss(logits, objective, prior=0.5)
    assert ce_loss is not None and pu_loss is not None
    assert float(ce_loss) != pytest.approx(float(pu_loss))


def test_defer_abstention_matches_ce_except_on_partial_unknown() -> None:
    witnessed = OperatorPolicyObjectiveV1(
        action_row=0,
        slot_id="value",
        compiler_support=OperatorSupportVerdict.SUPPORTED,
        coverage=LegalSetCoverage.PARTIAL,
        domain_complete=False,
        candidate_rows=(0, 1),
        positive_candidate_rows=(0,),
        route=InferenceRoute.PARTIAL_WITNESSED,
    )
    unknown = OperatorPolicyObjectiveV1(
        action_row=1,
        slot_id="value",
        compiler_support=OperatorSupportVerdict.SUPPORTED,
        coverage=LegalSetCoverage.PARTIAL,
        domain_complete=False,
        candidate_rows=(0, 1),
        positive_candidate_rows=(),
        route=InferenceRoute.PARTIAL_UNKNOWN,
    )
    logits = torch.tensor([1.0, -1.0])
    assert defer_abstention_loss(logits, unknown) is None
    ce = known_supported_ce_loss(logits, witnessed)
    defer = defer_abstention_loss(logits, witnessed)
    assert ce is not None and defer is not None
    assert float(ce) == pytest.approx(float(defer))


def test_run_matched_objective_fixture_reports_routes_without_false_eliminations() -> None:
    def fixture_builder(unknown_fraction, seed):
        reference_rows = tuple(_reference_row(row) for row in range(4))
        has_witness = unknown_fraction < 0.5
        coverage = LegalSetCoverage.COMPLETE if unknown_fraction == 0.0 else LegalSetCoverage.PARTIAL
        action = _action(
            0,
            candidate_rows=(0, 1, 2, 3),
            coverage=coverage,
            domain_complete=unknown_fraction == 0.0,
        )
        policy_input = OperatorPolicyInputV1(
            reference_rows=reference_rows,
            action_rows=(action,),
            ordinary_action_count=0,
            coverage=coverage,
        )
        positive_rows_by_action_row = {0: (0,)} if has_witness else {}
        torch.manual_seed(seed)
        logits_by_action_row = {0: torch.randn(4)}
        return policy_input, positive_rows_by_action_row, logits_by_action_row

    report = run_matched_objective_fixture(
        fixture_builder=fixture_builder,
        unknown_fractions=(0.0, 0.25, 0.5, 0.75),
    )
    payload = report.to_dict()
    assert payload["claim_class"] == "wiring"
    assert payload["false_hard_eliminations"] == 0
    assert payload["disposition"] == "wiring_underpowered"
    assert set(payload["unknown_fractions"]) == {
        "unknown_fraction_0.00",
        "unknown_fraction_0.25",
        "unknown_fraction_0.50",
        "unknown_fraction_0.75",
    }
    witnessed_entry = payload["unknown_fractions"]["unknown_fraction_0.25"]
    assert witnessed_entry["routes"] == {"partial_witnessed": 1}
    unknown_entry = payload["unknown_fractions"]["unknown_fraction_0.75"]
    assert unknown_entry["routes"] == {"partial_unknown": 1}
    assert unknown_entry["mean_loss"][ObjectiveArm.DEFER_ABSTENTION.value] is None
    assert unknown_entry["mean_loss"][ObjectiveArm.KNOWN_SUPPORTED_CE.value] is None
