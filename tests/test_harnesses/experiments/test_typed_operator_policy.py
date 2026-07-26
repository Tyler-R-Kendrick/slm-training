"""Regression coverage for SLM-403's default-off typed policy scorer."""

from __future__ import annotations

from slm_training.dsl.operators import (
    BindingPhase,
    CompilerFact,
    LegalSetCoverage,
    OperatorSupportVerdict,
    RefKind,
)
from slm_training.harnesses.experiments.typed_operator_policy import (
    TypedOperatorPolicyExampleV1,
    TypedOperatorPolicyScorer,
    decide_typed_operator_policy,
    train_typed_operator_policy,
    typed_operator_policy_loss,
)
from slm_training.models.operator_policy_view import (
    OperatorActionViewV1,
    OperatorArgumentSlotViewV1,
    OperatorPolicyInputV1,
    ReferenceModelViewV1,
)


def _example(*, coverage: LegalSetCoverage, action_count: int) -> TypedOperatorPolicyExampleV1:
    references = (
        ReferenceModelViewV1(
            row=0,
            ref_kind=RefKind.VALUE,
            value_type="openui.string",
            compiler_facts=(CompilerFact.VALUE_VISIBLE,),
            has_parent=False,
            parent_row=None,
            relative_position=None,
        ),
    )
    actions = tuple(
        OperatorActionViewV1(
            row=row,
            operator_id=f"openui.fixture_{row}",
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
                    candidate_rows=(0,),
                    domain_complete=coverage is LegalSetCoverage.COMPLETE,
                ),
            ),
            verdict=(
                OperatorSupportVerdict.SUPPORTED
                if coverage is LegalSetCoverage.COMPLETE
                else OperatorSupportVerdict.UNKNOWN
            ),
            coverage=coverage,
        )
        for row in range(action_count)
    )
    return TypedOperatorPolicyExampleV1(
        row_id=f"fixture-{coverage.value}-{action_count}",
        view=OperatorPolicyInputV1(
            reference_rows=references,
            action_rows=actions,
            ordinary_action_count=0,
            coverage=coverage,
        ),
        accepted_action_row=0,
        accepted_argument_rows=(("value", 0),),
    )


def test_complete_policy_scores_only_live_action_and_slot_rows() -> None:
    example = _example(coverage=LegalSetCoverage.COMPLETE, action_count=2)
    scorer = TypedOperatorPolicyScorer.from_examples((example,), dim=8)

    loss = typed_operator_policy_loss(scorer, example)
    decision = decide_typed_operator_policy(scorer, example)

    assert loss.requires_grad
    assert decision.model_forwards == 1
    assert decision.selected_action_row in {0, 1}
    assert decision.selected_argument_rows == (("value", 0),)


def test_complete_policy_trains_with_a_bounded_matched_schedule() -> None:
    example = _example(coverage=LegalSetCoverage.COMPLETE, action_count=2)
    scorer = TypedOperatorPolicyScorer.from_examples((example,), dim=8)

    history = train_typed_operator_policy(
        scorer, (example,), steps=3, learning_rate=0.05
    )

    assert len(history) == 3
    assert history[-1] < history[0]


def test_complete_singleton_bypasses_the_learned_forward() -> None:
    example = _example(coverage=LegalSetCoverage.COMPLETE, action_count=1)
    scorer = TypedOperatorPolicyScorer.from_examples((example,), dim=8)

    decision = decide_typed_operator_policy(scorer, example)

    assert decision.selected_action_row == 0
    assert decision.model_forwards == 0


def test_partial_policy_defers_without_force_or_hard_prune() -> None:
    example = _example(coverage=LegalSetCoverage.PARTIAL, action_count=2)
    scorer = TypedOperatorPolicyScorer.from_examples((example,), dim=8)

    loss = typed_operator_policy_loss(scorer, example)
    decision = decide_typed_operator_policy(scorer, example)

    assert loss.item() == 0.0
    assert decision.selected_action_row is None
    assert decision.selected_argument_rows == ()
    assert decision.model_forwards == 0
