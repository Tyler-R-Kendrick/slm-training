"""SLM-418 (DSH5-10) ninth slice: argument-level preference over the typed policy head."""

from __future__ import annotations

import math

import pytest

from slm_training.dsl.operators import (
    BindingPhase,
    BoundArgumentV1,
    LegalSetCoverage,
    OperatorSupportVerdict,
    RefKind,
    ReplayPreferenceRelation,
    enumerate_operator_legal_set,
    extract_replay_preference_rows,
    serialize_operator_action,
    undo_conversation,
)
from slm_training.harnesses.experiments.argument_preference import (
    TypedOperatorArgumentPreferenceExampleV1,
    build_argument_preference_example,
    train_typed_operator_argument_preference,
    typed_operator_argument_preference_loss,
)
from slm_training.harnesses.experiments.typed_operator_policy import TypedOperatorPolicyScorer
from slm_training.models.operator_policy_view import (
    OperatorActionViewV1,
    OperatorArgumentSlotViewV1,
    OperatorPolicyInputV1,
    ReferenceModelViewV1,
)
from tests.test_dsl.test_operator_conversation import _append, _fixture, _provenance
from tests.test_dsl.test_replay_preference import (
    _PRONOUN_OPERATOR_ID,
    _append_pronoun,
    _pronoun_focus_fixture,
)


def _legal_set_and_table(pack, library, trace, state_id):
    node = trace.node(state_id)
    legal_set = enumerate_operator_legal_set(
        pack=pack,
        library=library,
        state=node.state,
        reference_table=node.reference_table,
        provenance=_provenance(node.state),
        ordinary_nonoperator_actions=(),
    )
    return node.reference_table, legal_set


def _pronoun_focus_row_and_context():
    pack, library, trace, table_for, refs_by_index = _pronoun_focus_fixture()
    edited_once = _append_pronoun(pack, library, trace, refs_by_index[0], table_for=table_for)
    edited_twice = _append_pronoun(
        pack, library, edited_once, refs_by_index[0], table_for=table_for
    )
    report = extract_replay_preference_rows(
        edited_twice, pack=pack, library=library, provenance_for=_provenance
    )
    row = report.rows[0]
    assert row.semantic_relation is ReplayPreferenceRelation.PRONOUN_FOCUS_FOLLOWUP
    reference_table, legal_set = _legal_set_and_table(
        pack, library, edited_twice, row.input_state_id
    )
    return row, pack, library, reference_table, legal_set, refs_by_index


def test_build_argument_preference_example_from_a_real_pronoun_focus_row() -> None:
    row, pack, library, reference_table, legal_set, refs_by_index = _pronoun_focus_row_and_context()

    example = build_argument_preference_example(
        row, reference_table=reference_table, legal_set=legal_set, library=library
    )

    assert example is not None
    assert example.row_id == row.input_state_id
    assert example.slot_id == "value"
    action = example.view.action_rows[example.action_row]
    assert action.operator_id == _PRONOUN_OPERATOR_ID
    candidates = next(
        slot.candidate_rows for slot in action.argument_slots if slot.slot_id == "value"
    )
    assert example.chosen_reference_row in candidates
    assert example.rejected_reference_row in candidates
    assert example.chosen_reference_row != example.rejected_reference_row


def test_build_argument_preference_example_declines_non_pronoun_focus_relations() -> None:
    pack, library, root = _fixture()
    edited, _application = _append(pack, library, root)
    undone = undo_conversation(edited, provenance=_provenance(edited.current.state))
    report = extract_replay_preference_rows(
        undone, pack=pack, library=library, provenance_for=_provenance
    )
    row = report.rows[0]
    assert row.semantic_relation is ReplayPreferenceRelation.EDIT_THEN_UNDO
    reference_table, legal_set = _legal_set_and_table(pack, library, undone, row.input_state_id)

    example = build_argument_preference_example(
        row, reference_table=reference_table, legal_set=legal_set, library=library
    )

    assert example is None


def test_build_argument_preference_example_declines_a_row_with_no_differing_slot() -> None:
    """Defensive branch: a hand-built row whose chosen/rejected bind the same ref renders None."""
    from dataclasses import replace as dataclass_replace

    row, pack, library, reference_table, legal_set, refs_by_index = _pronoun_focus_row_and_context()
    same_action = serialize_operator_action(
        _PRONOUN_OPERATOR_ID, (BoundArgumentV1("value", refs_by_index[0]),)
    )
    degenerate_row = dataclass_replace(row, chosen_action=same_action, rejected_action=same_action)

    example = build_argument_preference_example(
        degenerate_row, reference_table=reference_table, legal_set=legal_set, library=library
    )

    assert example is None


def test_loss_is_finite_and_gradient_connected() -> None:
    row, pack, library, reference_table, legal_set, refs_by_index = _pronoun_focus_row_and_context()
    example = build_argument_preference_example(
        row, reference_table=reference_table, legal_set=legal_set, library=library
    )
    assert example is not None
    scorer = TypedOperatorPolicyScorer.from_examples((example,), dim=8)

    loss = typed_operator_argument_preference_loss(scorer, example)

    assert loss.requires_grad
    assert loss.isfinite()


def test_training_cannot_move_the_loss_when_candidates_are_feature_identical() -> None:
    """Honest null result, not a bug: this fixture's two sibling refs are
    provably indistinguishable to the scorer.

    ``ReferenceModelViewV1`` deliberately strips ``semantic_fingerprint``
    (the only thing distinguishing ``_PRONOUN_TARGETS[0]`` from ``[1]`` --
    see ``FORBIDDEN_FIELD_NAMES`` in
    ``slm_training.models.operator_policy_view``, anti-identity-leakage by
    design) and ``OperatorFeatureEncoder._reference_embeddings`` is a pure
    function of ``ref_kind``/``value_type``/``compiler_facts``/``has_parent``/
    ``relative_position`` only -- no row-index feature. Both candidate rows
    here share every one of those (same ``VALUE`` kind, same
    ``openui.string`` type, no parent, no position), so they get
    byte-identical embeddings through the shared-weight
    ``CandidateScoringHead`` regardless of any parameter update: the loss is
    stuck at ``-log_sigmoid(0) = ln(2)`` structurally, not from
    under-training. See ``test_training_reduces_the_pairwise_loss_when_candidates_differ``
    below for proof the loss function and gradient flow work correctly when
    the signal *is* representable.
    """
    row, pack, library, reference_table, legal_set, refs_by_index = _pronoun_focus_row_and_context()
    example = build_argument_preference_example(
        row, reference_table=reference_table, legal_set=legal_set, library=library
    )
    assert example is not None
    scorer = TypedOperatorPolicyScorer.from_examples((example,), dim=8)

    before = float(typed_operator_argument_preference_loss(scorer, example).detach())
    history = train_typed_operator_argument_preference(
        scorer, (example,), steps=20, learning_rate=0.05
    )
    after = float(typed_operator_argument_preference_loss(scorer, example).detach())

    assert len(history) == 20
    assert after == pytest.approx(before, abs=1e-6)
    assert before == pytest.approx(math.log(2.0), abs=1e-4)


def test_training_reduces_the_pairwise_loss_when_candidates_differ() -> None:
    """When the two candidates differ in an allowed feature (here:
    ``relative_position``), the signal is representable and training
    actually reduces the loss -- proving the mechanism itself works; the
    null result above is specific to feature-identical candidates, not a
    defect in the loss or the training loop.
    """
    references = (
        ReferenceModelViewV1(
            row=0,
            ref_kind=RefKind.INDEX,
            value_type="openui.string",
            compiler_facts=(),
            has_parent=False,
            parent_row=None,
            relative_position=0,
        ),
        ReferenceModelViewV1(
            row=1,
            ref_kind=RefKind.INDEX,
            value_type="openui.string",
            compiler_facts=(),
            has_parent=False,
            parent_row=None,
            relative_position=1,
        ),
    )
    action = OperatorActionViewV1(
        row=0,
        operator_id="openui.fixture_argpref",
        operator_version="v1",
        locality="node",
        cost=1.0,
        effect_signature=(),
        argument_slots=(
            OperatorArgumentSlotViewV1(
                slot_id="value",
                ref_kind=RefKind.INDEX,
                binding_phase=BindingPhase.APPLICATION,
                required=True,
                repeated=False,
                candidate_rows=(0, 1),
                domain_complete=True,
            ),
        ),
        verdict=OperatorSupportVerdict.SUPPORTED,
        coverage=LegalSetCoverage.COMPLETE,
    )
    view = OperatorPolicyInputV1(
        reference_rows=references,
        action_rows=(action,),
        ordinary_action_count=0,
        coverage=LegalSetCoverage.COMPLETE,
    )
    example = TypedOperatorArgumentPreferenceExampleV1(
        row_id="synthetic-distinguishable",
        view=view,
        action_row=0,
        slot_id="value",
        chosen_reference_row=0,
        rejected_reference_row=1,
    )
    scorer = TypedOperatorPolicyScorer.from_examples((example,), dim=8)

    before = float(typed_operator_argument_preference_loss(scorer, example).detach())
    history = train_typed_operator_argument_preference(
        scorer, (example,), steps=30, learning_rate=0.1
    )
    after = float(typed_operator_argument_preference_loss(scorer, example).detach())

    assert len(history) == 30
    assert after < before
