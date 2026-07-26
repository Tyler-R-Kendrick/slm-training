"""First consumer of the shared ``OPS_VOCAB`` kernel (SLM-428 / VAR2-01).

Decode invariant I13 (`docs/design/decode-invariants.md`) reserves a compute-op
vocabulary shared verbatim by both towers, but until this module nothing reads
it: `dsl.ops_vocab.shared_token_ids` had exactly two callers, both inside
`ops_vocab.py` itself. This module turns a persisted conversation trace's
turn history into the literal reserved-op token text I13 already provisions
token-id space for, so the encoder side has something real to condition on.

Every function here reads the ``to_dict()`` shape
`harnesses/train_data/operator_corpus.py` actually persists (real corpus
records), never a synthetic fixture. Turn -> op id resolution goes through
each operator's content-addressed ``operator_fingerprint`` (never a bare,
model-emitted label): AST_EDIT and TRANSACTION_COMMIT turns resolve through
`OperatorLibraryV1.lookup_by_fingerprint`; every other `ConversationOperation`
maps 1:1 onto its `conversation.<value>` history-family id
(`dsl.ops_vocab._history_op_ids`).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from slm_training.dsl.operators.conversation import ConversationOperation
from slm_training.dsl.operators.registry import OperatorLibraryV1
from slm_training.dsl.ops_vocab import OPS_VOCAB

_TOKEN_BY_OP_ID: dict[str, str] = {symbol.op_id: symbol.token for symbol in OPS_VOCAB}


class OpsVocabConditioningError(ValueError):
    """A conversation turn cannot be mapped onto a reserved OPS_VOCAB op id."""


def resolve_turn_op_ids(
    turn: Mapping[str, Any], library: OperatorLibraryV1
) -> tuple[str, ...]:
    """The OPS_VOCAB op id(s) one persisted ``TurnArtifactV1.to_dict()`` turn realizes."""
    operation = turn["operation"]
    if operation == ConversationOperation.AST_EDIT.value:
        application = turn.get("application")
        if not application:
            raise OpsVocabConditioningError("ast_edit turn is missing its application")
        declaration = library.lookup_by_fingerprint(application["operator_fingerprint"])
        return (declaration.operator_id,)
    if operation == ConversationOperation.TRANSACTION_COMMIT.value:
        transaction = turn.get("transaction")
        if not transaction:
            raise OpsVocabConditioningError(
                "transaction_commit turn is missing its transaction"
            )
        return tuple(
            library.lookup_by_fingerprint(action["operator_fingerprint"]).operator_id
            for action in transaction["prepared_actions"]
        )
    # Every remaining ConversationOperation member is a history-family op by
    # construction (`dsl.ops_vocab._history_op_ids`) — never a model label.
    return (f"conversation.{operation}",)


def conversation_trace_op_ids(
    trace: Mapping[str, Any], library: OperatorLibraryV1
) -> tuple[str, ...]:
    """Every OPS_VOCAB op id a persisted ``ConversationTraceV1.to_dict()`` trace realizes, in turn order."""
    op_ids: list[str] = []
    for turn in trace.get("turns", ()):
        op_ids.extend(resolve_turn_op_ids(turn, library))
    return tuple(op_ids)


def format_history_ops_text(op_ids: Sequence[str]) -> str:
    """Render a turn-op-id sequence as literal reserved OPS_VOCAB tokens.

    Every id must already be an ``OPS_VOCAB`` member. An id this function
    cannot map is a real ``unrepresentable_intents`` adequacy-audit finding,
    never a silently dropped token (I13: derived, not authored).
    """
    unknown = sorted({op_id for op_id in op_ids if op_id not in _TOKEN_BY_OP_ID})
    if unknown:
        raise OpsVocabConditioningError(
            "operator ids missing from OPS_VOCAB: " + ", ".join(unknown)
        )
    return " ".join(_TOKEN_BY_OP_ID[op_id] for op_id in op_ids)


__all__ = [
    "OpsVocabConditioningError",
    "conversation_trace_op_ids",
    "format_history_ops_text",
    "resolve_turn_op_ids",
]
