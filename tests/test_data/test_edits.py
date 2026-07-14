"""Structural edit, task-row, and trajectory invariants."""

from __future__ import annotations

import pytest

from slm_training.data.edits import (
    EditIntent,
    EditKind,
    EditOperation,
    EditPatch,
    EditTrajectory,
    apply_patch,
    build_transition,
    diff_programs,
    emit_transition_records,
    emit_trajectory_records,
    invert_patch,
    minimal_statement_patch,
    unsupported_patch,
)
from slm_training.data.progspec import ProgramSpec
from slm_training.dsl.language_contract import contract_id

BEFORE = (
    'root = Stack([card, cta], "column")\n'
    "card = Card([title])\n"
    'title = TextContent(":hero.title")\n'
    'cta = Button(":hero.cta")'
)
AFTER = (
    'root = Stack([card, cta], "column")\n'
    "card = Card([title])\n"
    'title = TextContent(":hero.title", "large-heavy")\n'
    'cta = Button(":hero.cta")'
)


def _spec(openui: str = BEFORE) -> ProgramSpec:
    return ProgramSpec(
        id="program_edit_1",
        ast={"type": "root"},
        canonical_openui=openui,
        facts={"components": ["Stack", "Card", "TextContent", "Button"]},
        contract_id=contract_id(),
        program_family_id="family_edit",
        lineage_id="lineage_edit",
        split_group_id="group_edit",
    )


def test_formal_intent_taxonomy_covers_the_operator_contract() -> None:
    assert {intent.value for intent in EditIntent} == {
        "add",
        "remove",
        "replace",
        "move",
        "reorder",
        "wrap",
        "unwrap",
        "split",
        "merge",
        "duplicate",
        "rename_preserving_refs",
        "change_content",
        "change_prop",
        "change_layout",
        "change_responsive",
        "add_or_modify_state",
        "add_or_modify_query",
        "add_or_modify_mutation",
        "add_or_modify_action",
        "apply_to_all_matching",
        "noop_already_satisfied",
        "unsupported_request",
    }


def test_diff_apply_inverse_and_minimality() -> None:
    patch = diff_programs(BEFORE, AFTER, instruction="make the title prominent")
    assert patch.ast_operation_count == 1
    assert patch.operations[0].kind is EditKind.REPLACE
    assert patch.operations[0].name == "title"
    assert apply_patch(BEFORE, patch) == AFTER
    assert apply_patch(AFTER, invert_patch(BEFORE, patch)) == BEFORE
    assert minimal_statement_patch(BEFORE, patch) == (
        'title = TextContent(":hero.title", "large-heavy")'
    )


def test_rename_is_one_operation_and_preserves_literal_content() -> None:
    renamed = BEFORE.replace("[card, cta]", "[card, primary]").replace(
        "cta = Button", "primary = Button"
    )
    patch = diff_programs(
        BEFORE,
        renamed,
        instruction="rename the CTA binder",
        renames={"cta": "primary"},
    )
    assert patch.ast_operation_count == 1
    assert patch.operations[0].kind is EditKind.RENAME
    assert apply_patch(BEFORE, patch) == renamed
    assert '":hero.cta"' in renamed


def test_disconnect_garbage_collects_unreachable_statements_and_inverts() -> None:
    after = 'root = Button(":hero.cta")'
    patch = EditPatch(
        (
            EditOperation(
                EditKind.REPLACE,
                "root",
                before='Stack([card, cta], "column")',
                after='Button(":hero.cta")',
            ),
        ),
        instruction="replace the page with one button",
    )
    assert apply_patch(BEFORE, patch) == after
    inverse = invert_patch(BEFORE, patch)
    assert apply_patch(after, inverse) == BEFORE
    assert inverse.ast_operation_count == 4


def test_transition_runs_f2_patch_gate_and_render_gate() -> None:
    rendered: list[str] = []
    transition = build_transition(
        BEFORE,
        "make the title prominent",
        diff_programs(BEFORE, AFTER),
        render_verifier=lambda source: rendered.append(source) is None,
    )
    assert transition.after == AFTER
    assert transition.render_verified is True
    assert rendered == [AFTER]

    with pytest.raises(ValueError, match="render verification"):
        build_transition(
            BEFORE,
            "make the title prominent",
            diff_programs(BEFORE, AFTER),
            render_verifier=lambda _: False,
        )


def test_emit_all_three_task_modes_with_lineage_and_patch_evidence() -> None:
    transition = build_transition(
        BEFORE,
        "make the title prominent",
        diff_programs(BEFORE, AFTER),
        render_verifier=lambda _: True,
    )
    records = emit_transition_records(_spec(), transition)
    assert [record.meta["edit"]["mode"] for record in records] == [
        "GENERATE",
        "APPLY_PATCH",
        "PATCH",
    ]
    assert [record.meta["task"] for record in records] == [
        "generation",
        "edit",
        "patch",
    ]
    assert {record.meta["split_group_id"] for record in records} == {"group_edit"}
    assert {record.meta["verification_tier"] for record in records} == {"Silver"}
    assert {record.meta["failing_gate"] for record in records} == {None}
    assert records[-1].meta["edit"]["statement_patch"].startswith("title =")


def test_trajectory_undo_redo_reference_and_partial_rollback() -> None:
    rendered: list[str] = []
    trajectory = EditTrajectory(
        BEFORE,
        render_verifier=lambda source: rendered.append(source) is None,
    )
    expanded = (
        'root = Stack([card, cta], "column")\n'
        "card = Card([title, note])\n"
        'title = TextContent(":hero.title", "large-heavy")\n'
        'note = TextContent(":hero.note")\n'
        'cta = Button(":hero.cta")'
    )
    transition = trajectory.apply(
        "make the title prominent and add a note",
        diff_programs(BEFORE, expanded, intent=EditIntent.ADD),
        focus="title",
    )
    assert transition.ast_operation_count == 3
    assert trajectory.resolve_reference("it") == "title"
    assert trajectory.resolve_reference("that title") == "title"

    trajectory.rollback_last(
        "undo the title change but keep the note",
        revert_names=("title",),
    )
    assert 'title = TextContent(":hero.title")' in trajectory.current
    assert 'note = TextContent(":hero.note")' in trajectory.current

    partial = trajectory.current
    assert trajectory.undo() == expanded
    assert trajectory.redo() == partial
    assert "undo the title change" in trajectory.summary()
    assert len(rendered) >= 5

    records = emit_trajectory_records(_spec(), trajectory)
    assert len(records) == 2
    assert {record.source for record in records} == {"edit_trajectory"}
    assert [record.meta["edit"]["turn"] for record in records] == [1, 2]
    assert records[1].meta["edit"]["history_summary"].startswith("1.")


def test_invalid_and_unsupported_transitions_are_explicit() -> None:
    invalid = EditPatch(
        (
            EditOperation(
                EditKind.REPLACE,
                "root",
                before='Stack([card, cta], "column")',
                after="Stack([missing])",
            ),
        )
    )
    with pytest.raises(ValueError):
        build_transition(
            BEFORE,
            "use a missing child",
            invalid,
            render_verifier=lambda _: True,
        )

    noop = unsupported_patch("component is outside the pinned contract")
    assert noop.ast_operation_count == 0
    assert noop.intent is EditIntent.UNSUPPORTED_REQUEST
    assert apply_patch(BEFORE, noop) == BEFORE
    transition = build_transition(
        BEFORE, "add a video", noop, render_verifier=lambda _: True
    )
    assert transition.after == BEFORE
    assert transition.statement_patch == ""
