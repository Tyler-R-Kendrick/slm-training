"""Immutable replayable conversation state graph for verified operators."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from slm_training.dsl.operators.contracts import (
    ApplicationProvenanceV1,
    OperatorApplicationV1,
    OperatorRef,
    RefKind,
    _fingerprint,
    _require_digest,
)
from slm_training.dsl.operators.references import (
    ReferenceDescriptorV1,
    ReferenceTableV1,
    branch_fingerprint,
    build_reference_table,
)
from slm_training.dsl.operators.registry import (
    OperatorLibraryV1,
    OperatorStateV1,
)
from slm_training.dsl.pack import DslPack


class ConversationOperation(str, Enum):
    AST_EDIT = "ast_edit"
    UNDO = "undo"
    REDO = "redo"
    CHECKOUT_STATE = "checkout_state"
    FORK = "fork"


class ConversationTraceError(ValueError):
    """The immutable state/turn graph violates its replay contract."""


OperatorAuthorityResolver = Callable[
    ["ConversationStateNodeV1"], tuple[DslPack, OperatorLibraryV1]
]


def _source_digest(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _state_dict(state: OperatorStateV1) -> dict[str, str]:
    return {
        "schema": state.schema,
        "pack_id": state.pack_id,
        "source": state.source,
        "state_digest": state.state_digest,
        "ast_digest": state.ast_digest,
    }


@dataclass(frozen=True)
class ConversationStateNodeV1:
    parent_state_id: str | None
    branch_digest: str
    state: OperatorStateV1
    reference_table: ReferenceTableV1
    schema: str = "conversation_state_node/v1"

    def __post_init__(self) -> None:
        if self.parent_state_id is not None:
            _require_digest(self.parent_state_id, "parent_state_id")
        _require_digest(self.branch_digest, "branch_digest")
        if self.reference_table.state_digest != self.state.state_digest:
            raise ConversationTraceError("state node reference table is stale")
        if self.reference_table.branch_digest != self.branch_digest:
            raise ConversationTraceError("state node reference table is cross-branch")

    @property
    def state_id(self) -> str:
        return _fingerprint(
            {
                "schema": self.schema,
                "parent_state_id": self.parent_state_id,
                "branch_digest": self.branch_digest,
                "state_digest": self.state.state_digest,
                "ast_digest": self.state.ast_digest,
                "reference_table_fingerprint": self.reference_table.fingerprint,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "state_id": self.state_id,
            "parent_state_id": self.parent_state_id,
            "branch_digest": self.branch_digest,
            "state": _state_dict(self.state),
            "reference_table": self.reference_table.to_dict(),
        }


@dataclass(frozen=True)
class TurnArtifactV1:
    operation: ConversationOperation
    input_state_id: str
    output_state_id: str
    provenance: ApplicationProvenanceV1
    application: OperatorApplicationV1 | None = None
    branch_nonce_digest: str | None = None
    reference_seed: int | None = None
    schema: str = "conversation_turn/v1"

    def __post_init__(self) -> None:
        _require_digest(self.input_state_id, "input_state_id")
        _require_digest(self.output_state_id, "output_state_id")
        if self.operation is ConversationOperation.AST_EDIT:
            if self.application is None:
                raise ConversationTraceError("AST edit turn requires an application")
            if not self.application.succeeded:
                raise ConversationTraceError(
                    "AST edit turn requires a successful application"
                )
            if self.application.provenance != self.provenance:
                raise ConversationTraceError(
                    "turn and application provenance disagree"
                )
        elif self.application is not None:
            raise ConversationTraceError(
                "history operations cannot carry AST applications"
            )
        fork_fields = (self.branch_nonce_digest, self.reference_seed)
        if self.operation is ConversationOperation.FORK:
            if any(value is None for value in fork_fields):
                raise ConversationTraceError(
                    "fork requires branch nonce and reference seed"
                )
            assert self.branch_nonce_digest is not None
            _require_digest(self.branch_nonce_digest, "branch_nonce_digest")
            assert self.reference_seed is not None
            if self.reference_seed < 0:
                raise ConversationTraceError("reference seed must be non-negative")
        elif any(value is not None for value in fork_fields):
            raise ConversationTraceError(
                "only fork turns may carry branch allocation fields"
            )

    @property
    def application_ids(self) -> tuple[str, ...]:
        if self.application is None:
            return ()
        return (self.application.application_id,)

    @property
    def turn_id(self) -> str:
        return _fingerprint(self.to_dict(include_turn_id=False))

    def to_dict(self, *, include_turn_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": self.schema,
            "operation": self.operation.value,
            "input_state_id": self.input_state_id,
            "output_state_id": self.output_state_id,
            "provenance": self.provenance.to_dict(),
            "application": (
                self.application.to_dict()
                if self.application is not None
                else None
            ),
            "application_ids": list(self.application_ids),
            "branch_nonce_digest": self.branch_nonce_digest,
            "reference_seed": self.reference_seed,
        }
        if include_turn_id:
            value["turn_id"] = self.turn_id
        return value


@dataclass(frozen=True)
class ConversationTraceV1:
    pack_id: str
    root_state_id: str
    current_state_id: str
    state_nodes: tuple[ConversationStateNodeV1, ...]
    turns: tuple[TurnArtifactV1, ...]
    root_provenance: ApplicationProvenanceV1
    schema: str = "conversation_trace/v1"

    def __post_init__(self) -> None:
        _require_digest(self.root_state_id, "root_state_id")
        _require_digest(self.current_state_id, "current_state_id")
        by_id = {node.state_id: node for node in self.state_nodes}
        if len(by_id) != len(self.state_nodes):
            raise ConversationTraceError("conversation state IDs must be unique")
        if self.root_state_id not in by_id or self.current_state_id not in by_id:
            raise ConversationTraceError("root/current state is missing")
        root = by_id[self.root_state_id]
        if root.parent_state_id is not None:
            raise ConversationTraceError("root state cannot have a parent")
        for node in self.state_nodes:
            if node.state_id != self.root_state_id and node.parent_state_id is None:
                raise ConversationTraceError("only the root may omit its parent")
            if (
                node.parent_state_id is not None
                and node.parent_state_id not in by_id
            ):
                raise ConversationTraceError("state node parent is missing")
            seen: set[str] = set()
            cursor = node
            while cursor.parent_state_id is not None:
                if cursor.state_id in seen:
                    raise ConversationTraceError("state parent graph contains a cycle")
                seen.add(cursor.state_id)
                cursor = by_id[cursor.parent_state_id]
            if cursor.state_id != self.root_state_id:
                raise ConversationTraceError("state node does not descend from root")
        if root.state.pack_id != self.pack_id:
            raise ConversationTraceError("trace root belongs to another pack")
        if self.root_provenance.pack_id != self.pack_id:
            raise ConversationTraceError("root provenance names another pack")
        if len({turn.turn_id for turn in self.turns}) != len(self.turns):
            raise ConversationTraceError("conversation turn IDs must be unique")
        parent_branches = [
            (node.parent_state_id, node.branch_digest)
            for node in self.state_nodes
            if node.parent_state_id is not None
        ]
        if len(set(parent_branches)) != len(parent_branches):
            raise ConversationTraceError(
                "a parent may have only one next state per branch"
            )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def node(self, state_id: str) -> ConversationStateNodeV1:
        try:
            return next(
                node for node in self.state_nodes if node.state_id == state_id
            )
        except StopIteration as exc:
            raise ConversationTraceError(f"unknown state ID {state_id}") from exc

    @property
    def current(self) -> ConversationStateNodeV1:
        return self.node(self.current_state_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "pack_id": self.pack_id,
            "root_state_id": self.root_state_id,
            "current_state_id": self.current_state_id,
            "state_nodes": [node.to_dict() for node in self.state_nodes],
            "turns": [turn.to_dict() for turn in self.turns],
            "root_provenance": self.root_provenance.to_dict(),
        }


def create_conversation_trace(
    *,
    pack: DslPack,
    root_state: OperatorStateV1,
    root_reference_table: ReferenceTableV1,
    provenance: ApplicationProvenanceV1,
) -> ConversationTraceV1:
    verified = OperatorStateV1.from_source(pack, root_state.source)
    if verified != root_state:
        raise ConversationTraceError("root state failed pack authority")
    if provenance.pack_id != pack.pack_id:
        raise ConversationTraceError("root provenance names another pack")
    if provenance.request_id != root_reference_table.request_id:
        raise ConversationTraceError("root provenance and reference request differ")
    if provenance.source_artifact_digest != _source_digest(root_state.source):
        raise ConversationTraceError("root provenance source digest is stale")
    root = ConversationStateNodeV1(
        parent_state_id=None,
        branch_digest=root_reference_table.branch_digest,
        state=root_state,
        reference_table=root_reference_table,
    )
    return ConversationTraceV1(
        pack_id=pack.pack_id,
        root_state_id=root.state_id,
        current_state_id=root.state_id,
        state_nodes=(root,),
        turns=(),
        root_provenance=provenance,
    )


def _history_provenance(
    trace: ConversationTraceV1, provenance: ApplicationProvenanceV1
) -> None:
    current = trace.current
    if provenance.pack_id != trace.pack_id:
        raise ConversationTraceError("history provenance names another pack")
    if provenance.request_id != current.reference_table.request_id:
        raise ConversationTraceError("history provenance request is stale")
    if provenance.source_artifact_digest != _source_digest(current.state.source):
        raise ConversationTraceError("history provenance source digest is stale")


def _append_turn(
    trace: ConversationTraceV1,
    turn: TurnArtifactV1,
    *,
    new_node: ConversationStateNodeV1 | None = None,
) -> ConversationTraceV1:
    if turn.input_state_id != trace.current_state_id:
        raise ConversationTraceError("turn input is not the current state")
    nodes = (
        (*trace.state_nodes, new_node)
        if new_node is not None
        else trace.state_nodes
    )
    return ConversationTraceV1(
        pack_id=trace.pack_id,
        root_state_id=trace.root_state_id,
        current_state_id=turn.output_state_id,
        state_nodes=nodes,
        turns=(*trace.turns, turn),
        root_provenance=trace.root_provenance,
    )


def append_operator_turn(
    trace: ConversationTraceV1,
    *,
    pack: DslPack,
    library: OperatorLibraryV1,
    application: OperatorApplicationV1,
    output_reference_table: ReferenceTableV1,
) -> ConversationTraceV1:
    current = trace.current
    if any(
        node.parent_state_id == current.state_id
        and node.branch_digest == current.branch_digest
        for node in trace.state_nodes
    ):
        raise ConversationTraceError(
            "same-branch next turn already exists; fork before adding a sibling"
        )
    if application.provenance.request_id != current.reference_table.request_id:
        raise ConversationTraceError("application request is stale for current refs")
    if application.provenance.source_artifact_digest != _source_digest(
        current.state.source
    ):
        raise ConversationTraceError("application source provenance is stale")
    if any(
        argument.value not in {
            entry.ref for entry in current.reference_table.entries
        }
        for argument in application.arguments
    ):
        raise ConversationTraceError("application uses stale or cross-branch refs")
    replayed = library.replay(pack, current.state, application)
    if not replayed.succeeded or replayed.state is None:
        raise ConversationTraceError("operator application did not replay")
    if output_reference_table.request_id != application.provenance.request_id:
        raise ConversationTraceError("output reference request differs")
    output = ConversationStateNodeV1(
        parent_state_id=current.state_id,
        branch_digest=current.branch_digest,
        state=replayed.state,
        reference_table=output_reference_table,
    )
    turn = TurnArtifactV1(
        operation=ConversationOperation.AST_EDIT,
        input_state_id=current.state_id,
        output_state_id=output.state_id,
        provenance=application.provenance,
        application=application,
    )
    return _append_turn(trace, turn, new_node=output)


def undo_conversation(
    trace: ConversationTraceV1,
    *,
    provenance: ApplicationProvenanceV1,
) -> ConversationTraceV1:
    _history_provenance(trace, provenance)
    parent_id = trace.current.parent_state_id
    if parent_id is None:
        raise ConversationTraceError("nothing to undo")
    turn = TurnArtifactV1(
        operation=ConversationOperation.UNDO,
        input_state_id=trace.current_state_id,
        output_state_id=parent_id,
        provenance=provenance,
    )
    return _append_turn(trace, turn)


def redo_conversation(
    trace: ConversationTraceV1,
    *,
    target_state_id: str,
    provenance: ApplicationProvenanceV1,
) -> ConversationTraceV1:
    _history_provenance(trace, provenance)
    target = trace.node(target_state_id)
    if target.parent_state_id != trace.current_state_id:
        raise ConversationTraceError("redo target is not a direct child")
    turn = TurnArtifactV1(
        operation=ConversationOperation.REDO,
        input_state_id=trace.current_state_id,
        output_state_id=target_state_id,
        provenance=provenance,
    )
    return _append_turn(trace, turn)


def checkout_conversation_state(
    trace: ConversationTraceV1,
    *,
    target_state_id: str,
    provenance: ApplicationProvenanceV1,
) -> ConversationTraceV1:
    _history_provenance(trace, provenance)
    trace.node(target_state_id)
    if target_state_id == trace.current_state_id:
        raise ConversationTraceError("checkout target is already current")
    turn = TurnArtifactV1(
        operation=ConversationOperation.CHECKOUT_STATE,
        input_state_id=trace.current_state_id,
        output_state_id=target_state_id,
        provenance=provenance,
    )
    return _append_turn(trace, turn)


def _fork_seed(branch_digest: str, requested_seed: int) -> int:
    if requested_seed < 0:
        raise ConversationTraceError("reference seed must be non-negative")
    return int(
        _fingerprint(
            {
                "schema": "conversation_fork_seed/v1",
                "branch_digest": branch_digest,
                "requested_seed": requested_seed,
            }
        )[:16],
        16,
    )


def clone_reference_table_for_branch(
    table: ReferenceTableV1,
    *,
    branch_digest: str,
    seed: int,
) -> ReferenceTableV1:
    """Reallocate branch-local opaque refs while preserving semantic descriptors."""
    _require_digest(branch_digest, "branch_digest")
    if branch_digest == table.branch_digest:
        raise ConversationTraceError("fork branch must differ from its source")
    semantic_map = {
        entry.descriptor.semantic_fingerprint: _fingerprint(
            {
                "schema": "conversation_fork_semantic_ref/v1",
                "source_semantic_fingerprint": (
                    entry.descriptor.semantic_fingerprint
                ),
                "branch_digest": branch_digest,
            }
        )
        for entry in table.entries
    }
    cloned_descriptors = tuple(
        replace(
            entry.descriptor,
            semantic_fingerprint=semantic_map[
                entry.descriptor.semantic_fingerprint
            ],
            parent_fingerprint=(
                semantic_map.get(
                    entry.descriptor.parent_fingerprint,
                    _fingerprint(
                        {
                            "schema": "conversation_fork_parent_ref/v1",
                            "source_parent_fingerprint": (
                                entry.descriptor.parent_fingerprint
                            ),
                            "branch_digest": branch_digest,
                        }
                    ),
                )
                if entry.descriptor.parent_fingerprint is not None
                else None
            ),
            parent_order_digest=(
                _fingerprint(
                    {
                        "schema": "conversation_fork_parent_order/v1",
                        "source_parent_order_digest": (
                            entry.descriptor.parent_order_digest
                        ),
                        "branch_digest": branch_digest,
                    }
                )
                if entry.descriptor.parent_order_digest is not None
                else None
            ),
        )
        for entry in table.entries
    )
    descriptor_fingerprint_map = {
        source.descriptor.fingerprint: cloned.fingerprint
        for source, cloned in zip(
            table.entries, cloned_descriptors, strict=True
        )
    }
    runtime_symbols = tuple(
        replace(
            symbol,
            symbol_fingerprint=_fingerprint(
                {
                    "schema": "conversation_fork_runtime_symbol/v1",
                    "source_symbol_fingerprint": symbol.symbol_fingerprint,
                    "branch_digest": branch_digest,
                }
            ),
            ref_fingerprint=descriptor_fingerprint_map[symbol.ref_fingerprint],
        )
        for symbol in table.runtime_symbols
    )
    return build_reference_table(
        request_id=table.request_id,
        state_digest=table.state_digest,
        branch_digest=branch_digest,
        descriptors=cloned_descriptors,
        seed=_fork_seed(branch_digest, seed),
        runtime_symbols=runtime_symbols,
    )


def fork_conversation(
    trace: ConversationTraceV1,
    *,
    branch_nonce_digest: str,
    reference_seed: int,
    provenance: ApplicationProvenanceV1,
) -> ConversationTraceV1:
    _history_provenance(trace, provenance)
    _require_digest(branch_nonce_digest, "branch_nonce_digest")
    source = trace.current
    branch = branch_fingerprint(
        trace.node(trace.root_state_id).state.state_digest,
        branch_nonce_digest,
    )
    if any(node.branch_digest == branch for node in trace.state_nodes):
        raise ConversationTraceError("fork branch already exists")
    cloned = clone_reference_table_for_branch(
        source.reference_table,
        branch_digest=branch,
        seed=reference_seed,
    )
    output = ConversationStateNodeV1(
        parent_state_id=source.state_id,
        branch_digest=branch,
        state=source.state,
        reference_table=cloned,
    )
    turn = TurnArtifactV1(
        operation=ConversationOperation.FORK,
        input_state_id=source.state_id,
        output_state_id=output.state_id,
        provenance=provenance,
        branch_nonce_digest=branch_nonce_digest,
        reference_seed=reference_seed,
    )
    return _append_turn(trace, turn, new_node=output)


def _validate_node(pack: DslPack, node: ConversationStateNodeV1) -> None:
    verified = OperatorStateV1.from_source(pack, node.state.source)
    if verified != node.state:
        raise ConversationTraceError("state node failed pack authority replay")
    if node.reference_table.state_digest != node.state.state_digest:
        raise ConversationTraceError("state node has stale references")
    if node.reference_table.branch_digest != node.branch_digest:
        raise ConversationTraceError("state node has cross-branch references")


def _validate_turn_provenance(
    trace: ConversationTraceV1,
    cursor: ConversationStateNodeV1,
    turn: TurnArtifactV1,
) -> None:
    provenance = turn.provenance
    if provenance.pack_id != trace.pack_id:
        raise ConversationTraceError("turn provenance names another pack")
    if provenance.request_id != cursor.reference_table.request_id:
        raise ConversationTraceError("turn provenance request is stale")
    if provenance.source_artifact_digest != _source_digest(cursor.state.source):
        raise ConversationTraceError("turn provenance source digest is stale")


def replay_conversation_trace(
    *,
    pack: DslPack,
    library: OperatorLibraryV1 | None = None,
    authority_resolver: OperatorAuthorityResolver | None = None,
    trace: ConversationTraceV1,
) -> ConversationStateNodeV1:
    """Replay every turn and exact intermediate state without hidden cursors."""
    if (library is None) == (authority_resolver is None):
        raise ConversationTraceError(
            "provide exactly one fixed library or state authority resolver"
        )
    if trace.pack_id != pack.pack_id:
        raise ConversationTraceError("trace belongs to another pack")
    by_id = {node.state_id: node for node in trace.state_nodes}
    root = trace.node(trace.root_state_id)
    _validate_node(pack, root)
    if (
        trace.root_provenance.pack_id != trace.pack_id
        or trace.root_provenance.request_id != root.reference_table.request_id
    ):
        raise ConversationTraceError("root provenance authority differs")
    if trace.root_provenance.source_artifact_digest != _source_digest(
        root.state.source
    ):
        raise ConversationTraceError("root provenance source digest is stale")
    created = {root.state_id}
    cursor = root
    for turn in trace.turns:
        if turn.input_state_id != cursor.state_id:
            raise ConversationTraceError("turn history is not cursor-contiguous")
        _validate_turn_provenance(trace, cursor, turn)
        if turn.output_state_id not in by_id:
            raise ConversationTraceError("turn output state is missing")
        output = by_id[turn.output_state_id]
        if turn.operation is ConversationOperation.AST_EDIT:
            assert turn.application is not None
            if output.parent_state_id != cursor.state_id:
                raise ConversationTraceError("AST edit output has wrong parent")
            if output.branch_digest != cursor.branch_digest:
                raise ConversationTraceError("AST edit crossed branches")
            if authority_resolver is None:
                turn_pack, turn_library = pack, library
            else:
                turn_pack, turn_library = authority_resolver(cursor)
            assert turn_library is not None
            if turn_pack.pack_id != trace.pack_id:
                raise ConversationTraceError(
                    "state authority resolver returned another pack"
                )
            replayed = turn_library.replay(
                turn_pack, cursor.state, turn.application
            )
            if replayed.state != output.state:
                raise ConversationTraceError("AST edit replay state differs")
            created.add(output.state_id)
        elif turn.operation is ConversationOperation.UNDO:
            if cursor.parent_state_id != output.state_id:
                raise ConversationTraceError("undo did not select exact parent")
        elif turn.operation is ConversationOperation.REDO:
            if output.parent_state_id != cursor.state_id:
                raise ConversationTraceError("redo did not select exact child")
        elif turn.operation is ConversationOperation.CHECKOUT_STATE:
            pass
        else:
            assert turn.operation is ConversationOperation.FORK
            assert turn.branch_nonce_digest is not None
            assert turn.reference_seed is not None
            expected_branch = branch_fingerprint(
                root.state.state_digest, turn.branch_nonce_digest
            )
            expected_table = clone_reference_table_for_branch(
                cursor.reference_table,
                branch_digest=expected_branch,
                seed=turn.reference_seed,
            )
            if (
                output.parent_state_id != cursor.state_id
                or output.state != cursor.state
                or output.branch_digest != expected_branch
                or output.reference_table != expected_table
            ):
                raise ConversationTraceError("fork replay differs")
            created.add(output.state_id)
        _validate_node(pack, output)
        cursor = output
    if cursor.state_id != trace.current_state_id:
        raise ConversationTraceError("trace current state differs from replay")
    if created != set(by_id):
        raise ConversationTraceError("trace contains orphan state nodes")
    return cursor


def resolve_branch_reference(
    node: ConversationStateNodeV1,
    ref: OperatorRef,
    *,
    expected_kind: RefKind,
    current_parent_order_digest: str | None = None,
) -> ReferenceDescriptorV1:
    """Resolve only through the node-owned branch/state table."""
    return node.reference_table.resolve(
        ref,
        state_digest=node.state.state_digest,
        branch_digest=node.branch_digest,
        expected_kind=expected_kind,
        current_parent_order_digest=current_parent_order_digest,
    )


__all__ = [
    "ConversationOperation",
    "ConversationStateNodeV1",
    "ConversationTraceError",
    "ConversationTraceV1",
    "OperatorAuthorityResolver",
    "TurnArtifactV1",
    "append_operator_turn",
    "checkout_conversation_state",
    "clone_reference_table_for_branch",
    "create_conversation_trace",
    "fork_conversation",
    "redo_conversation",
    "replay_conversation_trace",
    "resolve_branch_reference",
    "undo_conversation",
]
