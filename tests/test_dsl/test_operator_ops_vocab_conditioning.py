from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from tests.casefiles import case_values

from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BindingPhase,
    BoundArgumentV1,
    CompilerCoverage,
    EffectDeltaKind,
    EffectDeltaV1,
    OperatorArgumentSlotV1,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorStateV1,
    RefKind,
    RegisteredOperatorV1,
    ValueRef,
)
from slm_training.dsl.operators.ops_vocab_conditioning import (
    OpsVocabConditioningError,
    conversation_trace_op_ids,
    format_history_ops_text,
    resolve_turn_op_ids,
)
from slm_training.dsl.ops_vocab import shared_token_ids
from slm_training.dsl.pack import get_pack

SOURCE = 'root = Card([TextContent(":hero.title"), TextContent(":hero.body")], "clear")'


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _declaration() -> AstOperatorV1:
    # A real OPS_VOCAB member (`ops_vocab._FAMILY_BY_OP["openui.set_property"]`),
    # so the resolved id round-trips through format_history_ops_text below too.
    return AstOperatorV1(
        operator_id="openui.set_property",
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(
            OperatorArgumentSlotV1("value", RefKind.VALUE, BindingPhase.APPLICATION),
        ),
        preconditions=(),
        effect_signature=(EffectDeltaKind.PROPERTY,),
        locality="node",
        cost=1.0,
    )


def _executor(state, arguments):
    ref = arguments[0].value
    return OperatorMutationV1(
        source=state.source.replace(":hero.title", ":hero.body"),
        effect=ActionEffectV1(
            property_deltas=(
                EffectDeltaV1(
                    EffectDeltaKind.PROPERTY, ref, ":hero.title", ":hero.body"
                ),
            ),
            compiler_coverage=CompilerCoverage.EXACT,
            estimated_completion_cost=1.0,
        ),
    )


def _library() -> OperatorLibraryV1:
    return OperatorLibraryV1((RegisteredOperatorV1(_declaration(), _executor),))


def _library_and_application() -> tuple[OperatorLibraryV1, dict]:
    library = _library()
    pack = replace(get_pack("openui"), operator_library=library)
    state = OperatorStateV1.from_source(pack, SOURCE)
    provenance = ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.compiler",
        compiler_version="2026.07",
        source_artifact_digest=_sha(SOURCE),
        request_id="req-1",
    )
    arguments = (BoundArgumentV1("value", ValueRef("req-1", "v1")),)
    applied = library.apply(pack, state, "openui.set_property", arguments, provenance)
    assert applied.succeeded
    return library, applied.application.to_dict()


def test_ast_edit_turn_resolves_the_real_operator_id() -> None:
    library, application = _library_and_application()
    turn = {"operation": "ast_edit", "application": application}
    assert resolve_turn_op_ids(turn, library) == ("openui.set_property",)


def test_ast_edit_turn_without_application_is_a_typed_error() -> None:
    library = _library()
    with pytest.raises(OpsVocabConditioningError, match="missing its application"):
        resolve_turn_op_ids({"operation": "ast_edit", "application": None}, library)


def test_transaction_commit_turn_without_transaction_is_a_typed_error() -> None:
    library = _library()
    with pytest.raises(OpsVocabConditioningError, match="missing its transaction"):
        resolve_turn_op_ids(
            {"operation": "transaction_commit", "transaction": None}, library
        )


def test_ast_edit_turn_with_unrecognized_fingerprint_is_a_typed_error() -> None:
    # A fingerprint the resolution library doesn't recognize must become a
    # real unrepresentable_intents finding for a caller (e.g. the adequacy
    # audit script), never an uncaught KeyError.
    library, application = _library_and_application()
    application = {**application, "operator_fingerprint": "0" * 64}
    with pytest.raises(OpsVocabConditioningError, match="operator fingerprint"):
        resolve_turn_op_ids(
            {"operation": "ast_edit", "application": application}, library
        )


@pytest.mark.parametrize(
    "operation, expected",
    case_values(__file__, "test_history_operations_map_onto_their_reserved_id"),
)
def test_history_operations_map_onto_their_reserved_id(
    operation: str, expected: str
) -> None:
    library = _library()
    assert resolve_turn_op_ids({"operation": operation}, library) == (expected,)


def test_conversation_trace_op_ids_preserves_turn_order() -> None:
    library, application = _library_and_application()
    trace = {
        "turns": [
            {"operation": "undo"},
            {"operation": "ast_edit", "application": application},
            {"operation": "redo"},
        ]
    }
    assert conversation_trace_op_ids(trace, library) == (
        "conversation.undo",
        "openui.set_property",
        "conversation.redo",
    )


def test_format_history_ops_text_uses_the_real_reserved_tokens() -> None:
    tokens = shared_token_ids()
    text = format_history_ops_text(("conversation.undo", "openui.set_property"))
    assert text == "<|op:conversation.undo|> <|op:openui.set_property|>"
    for token in text.split(" "):
        assert token in tokens  # every emitted token really is reserved (I13)


def test_format_history_ops_text_rejects_unknown_ids() -> None:
    with pytest.raises(OpsVocabConditioningError, match="openui.not_a_real_op"):
        format_history_ops_text(("openui.not_a_real_op",))
