"""Regression coverage for SLM-403's default-off typed policy scorer."""

from __future__ import annotations

from pathlib import Path

import pytest

from slm_training.data.contract import GenerationRequest
from slm_training.dsl.operators import (
    BindingPhase,
    CompilerFact,
    LegalSetCoverage,
    OperatorSupportVerdict,
    RefKind,
)
from slm_training.dsl.schema import ExampleRecord, write_jsonl
from slm_training.harnesses.model_build import ModelBuildConfig
from slm_training.harnesses.model_build.eval_runner import evaluate
from slm_training.harnesses.model_build.plugin import StubModel
from slm_training.harnesses.experiments.typed_operator_policy import (
    TYPED_OPERATOR_POLICY_HEAD_FAMILIES,
    TypedOperatorPolicyDecisionV1,
    TypedOperatorPolicyEvidencePlugin,
    TypedOperatorPolicyEvidenceV1,
    TypedOperatorPolicyExampleV1,
    TypedOperatorPolicyScorer,
    decide_typed_operator_policy,
    train_typed_operator_policy,
    typed_operator_policy_loss,
)
from slm_training.models.operator_policy_objective import OperatorPolicyRouteV1
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


@pytest.mark.parametrize("head_family", TYPED_OPERATOR_POLICY_HEAD_FAMILIES)
def test_each_typed_head_preserves_complete_and_partial_authority(head_family: str) -> None:
    ambiguous = _example(coverage=LegalSetCoverage.COMPLETE, action_count=2)
    singleton = _example(coverage=LegalSetCoverage.COMPLETE, action_count=1)
    partial = _example(coverage=LegalSetCoverage.PARTIAL, action_count=2)
    scorer = TypedOperatorPolicyScorer.from_examples(
        (ambiguous,), dim=8, head_family=head_family  # type: ignore[arg-type]
    )

    loss = typed_operator_policy_loss(scorer, ambiguous)
    decision = decide_typed_operator_policy(scorer, ambiguous)
    singleton_decision = decide_typed_operator_policy(scorer, singleton)
    partial_decision = decide_typed_operator_policy(scorer, partial)

    assert loss.requires_grad
    assert decision.selected_action_row in {0, 1}
    assert decision.selected_argument_rows == (("value", 0),)
    assert singleton_decision.model_forwards == 0
    assert singleton_decision.selected_action_row == 0
    assert partial_decision.selected_action_row is None
    assert partial_decision.model_forwards == 0


def test_model_plugin_side_channel_preserves_openui_and_request_alignment() -> None:
    class Delegate:
        def generate_batch_requests(
            self, requests: list[GenerationRequest], **_kwargs: object
        ) -> list[str]:
            return ["root = Separator()" for _ in requests]

        def consume_generation_evidence(self) -> list[dict[str, object]]:
            return [{"grammar_constrained": True}]

    decision = TypedOperatorPolicyDecisionV1(
        route=OperatorPolicyRouteV1.COMPLETE_AMBIGUOUS,
        selected_action_row=1,
        selected_argument_rows=(("value", 0),),
        model_forwards=1,
    )
    plugin = TypedOperatorPolicyEvidencePlugin(
        Delegate(),  # type: ignore[arg-type]
        lambda _request: TypedOperatorPolicyEvidenceV1("local_flat", decision),
    )
    requests = [GenerationRequest(prompt="add a separator")]

    assert plugin.generate_batch_requests(requests) == ["root = Separator()"]
    assert plugin.consume_generation_evidence() == [
        {
            "grammar_constrained": True,
            "typed_operator_policy": {
                "schema": "typed_operator_policy_evidence/v1",
                "head_family": "local_flat",
                "route": "complete_ambiguous",
                "selected_action_row": 1,
                "selected_argument_rows": [
                    {"slot_id": "value", "reference_row": 0}
                ],
                "model_forwards": 1,
            },
        }
    ]


def test_model_plugin_side_channel_rejects_misaligned_delegate_evidence() -> None:
    class Delegate:
        def generate_batch_requests(
            self, requests: list[GenerationRequest], **_kwargs: object
        ) -> list[str]:
            return ["root = Separator()" for _ in requests]

        def consume_generation_evidence(self) -> list[dict[str, object]]:
            return [{}, {}]

    decision = TypedOperatorPolicyDecisionV1(
        route=OperatorPolicyRouteV1.PARTIAL_UNKNOWN,
        selected_action_row=None,
        selected_argument_rows=(),
        model_forwards=0,
    )
    plugin = TypedOperatorPolicyEvidencePlugin(
        Delegate(),  # type: ignore[arg-type]
        lambda _request: TypedOperatorPolicyEvidenceV1("local_flat", decision),
    )

    plugin.generate_batch_requests([GenerationRequest(prompt="add a separator")])
    with pytest.raises(ValueError, match="request-aligned"):
        plugin.consume_generation_evidence()


def test_model_plugin_side_channel_keeps_normal_eval_scoring(tmp_path: Path) -> None:
    train_dir = tmp_path / "train"
    test_dir = tmp_path / "eval"
    suite_dir = test_dir / "suites" / "smoke"
    suite_dir.mkdir(parents=True)
    gold = 'root = TextContent(":slot_0")'
    record = ExampleRecord(
        id="side-channel",
        prompt="add label",
        openui=gold,
        placeholders=[":slot_0"],
    )
    write_jsonl(train_dir / "records.jsonl", [record])
    write_jsonl(suite_dir / "records.jsonl", [record])
    decision = TypedOperatorPolicyDecisionV1(
        route=OperatorPolicyRouteV1.COMPLETE_AMBIGUOUS,
        selected_action_row=1,
        selected_argument_rows=(),
        model_forwards=1,
    )
    plugin = TypedOperatorPolicyEvidencePlugin(
        StubModel(memory={record.prompt: gold}),
        lambda _request: TypedOperatorPolicyEvidenceV1("local_flat", decision),
    )

    metrics = evaluate(
        ModelBuildConfig(
            train_dir=train_dir,
            test_dir=test_dir,
            suite="smoke",
            run_root=tmp_path / "runs",
            run_id="typed-policy-side-channel",
            model_name="stub",
            context_backend="scratch",
        ),
        model=plugin,
        publish_agentv=False,
    )

    assert metrics["meaningful_program_rate"] == 1.0
    assert metrics["details"][0]["prediction"] == gold
    assert metrics["details"][0]["topology_evidence"]["typed_operator_policy"] == {
        "schema": "typed_operator_policy_evidence/v1",
        "head_family": "local_flat",
        "route": "complete_ambiguous",
        "selected_action_row": 1,
        "selected_argument_rows": [],
        "model_forwards": 1,
    }
