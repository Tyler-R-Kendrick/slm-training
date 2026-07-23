from __future__ import annotations

import hashlib
from dataclasses import FrozenInstanceError, replace

import pytest

from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    CompilerCoverage,
    ConversationOperation,
    ConversationStateNodeV1,
    ConversationTraceError,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorRejectedError,
    OperatorStateV1,
    RefKind,
    ReferenceDescriptorV1,
    ReferenceResolutionError,
    RegisteredOperatorV1,
    append_operator_turn,
    branch_fingerprint,
    build_reference_table,
    checkout_conversation_state,
    create_conversation_trace,
    fork_conversation,
    redo_conversation,
    replay_conversation_trace,
    resolve_branch_reference,
    undo_conversation,
)
from slm_training.dsl.pack import get_pack

SOURCE = 'root = TextContent(":hero.title")'
OPERATOR_ID = "openui.advance_fixture"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _provenance(state: OperatorStateV1) -> ApplicationProvenanceV1:
    return ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.conversation_fixture",
        compiler_version="v1",
        source_artifact_digest=_sha(state.source),
        request_id="request-1",
    )


def _table(
    state: OperatorStateV1,
    branch: str,
    *,
    seed: int,
):
    return build_reference_table(
        request_id="request-1",
        state_digest=state.state_digest,
        branch_digest=branch,
        descriptors=(
            ReferenceDescriptorV1(
                ref_kind=RefKind.VALUE,
                semantic_fingerprint=_sha("visible-value"),
                value_type="openui.string",
            ),
        ),
        seed=seed,
    )


def _fixture():
    base_pack = get_pack("openui")
    root_state = OperatorStateV1.from_source(base_pack, SOURCE)

    def execute(state, _arguments):
        replacements = (
            (":hero.title", ":hero.body"),
            (":hero.body", ":hero.caption"),
            (":hero.caption", ":hero.title"),
        )
        for before, after in replacements:
            if before in state.source:
                return OperatorMutationV1(
                    source=state.source.replace(before, after),
                    effect=ActionEffectV1(
                        compiler_coverage=CompilerCoverage.EXACT
                    ),
                )
        raise OperatorRejectedError("fixture.no_transition")

    declaration = AstOperatorV1(
        operator_id=OPERATOR_ID,
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(),
        preconditions=(),
        effect_signature=(),
        locality="node",
        cost=1.0,
    )
    library = OperatorLibraryV1((RegisteredOperatorV1(declaration, execute),))
    pack = replace(base_pack, operator_library=library)
    branch = branch_fingerprint(root_state.state_digest, _sha("root-branch"))
    root_table = _table(root_state, branch, seed=1)
    trace = create_conversation_trace(
        pack=pack,
        root_state=root_state,
        root_reference_table=root_table,
        provenance=_provenance(root_state),
    )
    return pack, library, trace


def _application(pack, library, state):
    result = library.apply(pack, state, OPERATOR_ID, (), _provenance(state))
    assert result.succeeded
    assert result.state is not None
    return result.application, result.state


def _append(pack, library, trace, *, seed: int = 2):
    application, output_state = _application(pack, library, trace.current.state)
    output_table = _table(
        output_state,
        trace.current.branch_digest,
        seed=seed,
    )
    return (
        append_operator_turn(
            trace,
            pack=pack,
            library=library,
            application=application,
            output_reference_table=output_table,
        ),
        application,
    )


def test_trace_replays_edit_undo_redo_checkout_fork_and_branch_edit() -> None:
    pack, library, root = _fixture()
    edited, application = _append(pack, library, root)
    original_child_id = edited.current_state_id
    assert edited.turns[-1].application_ids == (application.application_id,)
    assert edited.turns[-1].operation is ConversationOperation.AST_EDIT

    undone = undo_conversation(
        edited, provenance=_provenance(edited.current.state)
    )
    assert undone.current_state_id == root.root_state_id
    redone = redo_conversation(
        undone,
        target_state_id=original_child_id,
        provenance=_provenance(undone.current.state),
    )
    assert redone.current.state == edited.current.state
    checked_out = checkout_conversation_state(
        redone,
        target_state_id=root.root_state_id,
        provenance=_provenance(redone.current.state),
    )
    forked = fork_conversation(
        checked_out,
        branch_nonce_digest=_sha("fork-one"),
        reference_seed=11,
        provenance=_provenance(checked_out.current.state),
    )
    fork_root = forked.current
    assert fork_root.state == root.current.state
    assert fork_root.branch_digest != root.current.branch_digest
    assert fork_root.reference_table.fingerprint != (
        root.current.reference_table.fingerprint
    )
    assert {
        entry.descriptor.semantic_fingerprint
        for entry in fork_root.reference_table.entries
    }.isdisjoint(
        {
            entry.descriptor.semantic_fingerprint
            for entry in root.current.reference_table.entries
        }
    )

    old_ref = root.current.reference_table.entries[0].ref
    with pytest.raises(ReferenceResolutionError, match="ref.missing"):
        resolve_branch_reference(
            fork_root, old_ref, expected_kind=RefKind.VALUE
        )
    with pytest.raises(ReferenceResolutionError, match="ref.cross_branch"):
        root.current.reference_table.resolve(
            old_ref,
            state_digest=root.current.state.state_digest,
            branch_digest=fork_root.branch_digest,
            expected_kind=RefKind.VALUE,
        )

    fork_edited, fork_application = _append(pack, library, forked, seed=12)
    assert fork_application.application_id == application.application_id
    assert fork_edited.current.state == edited.current.state
    assert fork_edited.current.state_id != original_child_id
    assert fork_edited.node(original_child_id) == edited.current
    fork_second_edit, _ = _append(pack, library, fork_edited, seed=13)
    assert ":hero.caption" in fork_second_edit.current.state.source
    assert replay_conversation_trace(
        pack=pack, library=library, trace=fork_second_edit
    ) == fork_second_edit.current
    assert replay_conversation_trace(
        pack=pack,
        authority_resolver=lambda _node: (pack, library),
        trace=fork_second_edit,
    ) == fork_second_edit.current


def test_trace_construction_and_replay_are_deterministic() -> None:
    def build():
        pack, library, trace = _fixture()
        trace, _ = _append(pack, library, trace)
        trace = undo_conversation(
            trace, provenance=_provenance(trace.current.state)
        )
        trace = fork_conversation(
            trace,
            branch_nonce_digest=_sha("deterministic-fork"),
            reference_seed=17,
            provenance=_provenance(trace.current.state),
        )
        return pack, library, trace

    first_pack, first_library, first = build()
    _, _, second = build()
    assert first.to_dict() == second.to_dict()
    assert first.fingerprint == second.fingerprint
    assert replay_conversation_trace(
        pack=first_pack, library=first_library, trace=first
    ).state_id == first.current_state_id


def test_one_same_branch_next_turn_requires_explicit_sibling_fork() -> None:
    pack, library, root = _fixture()
    edited, application = _append(pack, library, root)
    at_root = checkout_conversation_state(
        edited,
        target_state_id=root.root_state_id,
        provenance=_provenance(edited.current.state),
    )
    _, output_state = _application(pack, library, at_root.current.state)
    with pytest.raises(ConversationTraceError, match="fork"):
        append_operator_turn(
            at_root,
            pack=pack,
            library=library,
            application=application,
            output_reference_table=_table(
                output_state, at_root.current.branch_digest, seed=2
            ),
        )

    first_fork = fork_conversation(
        at_root,
        branch_nonce_digest=_sha("sibling-one"),
        reference_seed=3,
        provenance=_provenance(at_root.current.state),
    )
    back = checkout_conversation_state(
        first_fork,
        target_state_id=root.root_state_id,
        provenance=_provenance(first_fork.current.state),
    )
    second_fork = fork_conversation(
        back,
        branch_nonce_digest=_sha("sibling-two"),
        reference_seed=3,
        provenance=_provenance(back.current.state),
    )
    siblings = [
        node
        for node in second_fork.state_nodes
        if node.parent_state_id == root.root_state_id
        and node.branch_digest != root.current.branch_digest
    ]
    assert len(siblings) == 2
    assert siblings[0].branch_digest != siblings[1].branch_digest
    replay_conversation_trace(pack=pack, library=library, trace=second_fork)


def test_history_boundaries_fail_closed_without_hidden_redo_state() -> None:
    pack, library, root = _fixture()
    with pytest.raises(ConversationTraceError, match="nothing to undo"):
        undo_conversation(root, provenance=_provenance(root.current.state))
    with pytest.raises(ConversationTraceError, match="already current"):
        checkout_conversation_state(
            root,
            target_state_id=root.root_state_id,
            provenance=_provenance(root.current.state),
        )

    edited, _ = _append(pack, library, root)
    undone = undo_conversation(
        edited, provenance=_provenance(edited.current.state)
    )
    with pytest.raises(ConversationTraceError, match="direct child"):
        redo_conversation(
            undone,
            target_state_id=root.root_state_id,
            provenance=_provenance(undone.current.state),
        )
    stale = replace(
        _provenance(undone.current.state),
        source_artifact_digest=_sha("stale"),
    )
    with pytest.raises(ConversationTraceError, match="source digest"):
        fork_conversation(
            undone,
            branch_nonce_digest=_sha("never-created"),
            reference_seed=1,
            provenance=stale,
        )


def test_append_rejects_stale_output_table_and_application_provenance() -> None:
    pack, library, root = _fixture()
    application, output_state = _application(
        pack, library, root.current.state
    )
    wrong_branch = branch_fingerprint(
        root.current.state.state_digest, _sha("wrong-branch")
    )
    with pytest.raises(ConversationTraceError, match="cross-branch"):
        append_operator_turn(
            root,
            pack=pack,
            library=library,
            application=application,
            output_reference_table=_table(output_state, wrong_branch, seed=2),
        )

    stale_application = replace(
        application,
        provenance=replace(
            application.provenance,
            source_artifact_digest=_sha("stale-source"),
        ),
    )
    with pytest.raises(ConversationTraceError, match="source provenance"):
        append_operator_turn(
            root,
            pack=pack,
            library=library,
            application=stale_application,
            output_reference_table=_table(
                output_state, root.current.branch_digest, seed=2
            ),
        )


def test_replay_detects_invalid_intermediate_and_stale_turn_provenance() -> None:
    pack, library, root = _fixture()
    edited, _ = _append(pack, library, root)
    output = edited.current
    bad_state = replace(output.state, state_digest=_sha("bad-state"))
    bad_table = replace(
        output.reference_table, state_digest=bad_state.state_digest
    )
    bad_node = ConversationStateNodeV1(
        parent_state_id=output.parent_state_id,
        branch_digest=output.branch_digest,
        state=bad_state,
        reference_table=bad_table,
    )
    bad_turn = replace(edited.turns[-1], output_state_id=bad_node.state_id)
    bad_trace = replace(
        edited,
        current_state_id=bad_node.state_id,
        state_nodes=(edited.state_nodes[0], bad_node),
        turns=(bad_turn,),
    )
    with pytest.raises(ConversationTraceError, match="replay state differs"):
        replay_conversation_trace(
            pack=pack, library=library, trace=bad_trace
        )

    turn = edited.turns[-1]
    assert turn.application is not None
    stale_provenance = replace(
        turn.provenance, source_artifact_digest=_sha("stale")
    )
    stale_application = replace(
        turn.application, provenance=stale_provenance
    )
    stale_turn = replace(
        turn,
        provenance=stale_provenance,
        application=stale_application,
    )
    stale_trace = replace(edited, turns=(stale_turn,))
    with pytest.raises(ConversationTraceError, match="source digest"):
        replay_conversation_trace(
            pack=pack, library=library, trace=stale_trace
        )


def test_trace_values_are_frozen_and_provenance_complete() -> None:
    pack, library, trace = _fixture()
    trace, _ = _append(pack, library, trace)
    with pytest.raises(FrozenInstanceError):
        trace.current_state_id = trace.root_state_id  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        trace.current.branch_digest = _sha("mutated")  # type: ignore[misc]
    payload = trace.to_dict()
    assert payload["root_provenance"]["source_artifact_digest"] == _sha(SOURCE)
    assert payload["turns"][0]["application_ids"] == [
        trace.turns[0].application.application_id  # type: ignore[union-attr]
    ]
    replay_conversation_trace(pack=pack, library=library, trace=trace)
    with pytest.raises(ConversationTraceError, match="exactly one"):
        replay_conversation_trace(pack=pack, trace=trace)
    with pytest.raises(ConversationTraceError, match="exactly one"):
        replay_conversation_trace(
            pack=pack,
            library=library,
            authority_resolver=lambda _node: (pack, library),
            trace=trace,
        )
