"""Replay-authoritative collapse of verified operator conversations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any

from slm_training.dsl.operators.contracts import (
    OperatorApplicationV1,
    _fingerprint,
    _require_digest,
    _require_identifier,
)
from slm_training.dsl.operators.conversation import (
    ConversationOperation,
    ConversationStateNodeV1,
    ConversationTraceV1,
    ConversationTraceError,
    OperatorAuthorityResolver,
    replay_conversation_trace,
)
from slm_training.dsl.operators.registry import OperatorReplayError
from slm_training.dsl.pack import DslPack


class CollapseRejectionKind(str, Enum):
    TOO_SHORT = "too_short"
    HISTORY_OPERATION = "history_operation"
    NO_OP = "no_op"
    CYCLE = "cycle"
    REDUNDANT = "redundant"
    REPLAY_MISMATCH = "replay_mismatch"


class HardNegativeOutcome(str, Enum):
    CONFLICT = "conflict"
    DIFFERENT_RESULT = "different_result"


@dataclass(frozen=True)
class CollapsedHardNegativeV1:
    swapped_step_indices: tuple[int, int]
    application_ids: tuple[str, ...]
    outcome: HardNegativeOutcome
    observed_final_state_digest: str | None = None
    conflict_code: str | None = None
    schema: str = "collapsed_hard_negative/v1"

    def __post_init__(self) -> None:
        left, right = self.swapped_step_indices
        if right != left + 1 or left < 0:
            raise ValueError("hard negative must swap adjacent ordered steps")
        if len(self.application_ids) < 2:
            raise ValueError("hard negative requires the complete source sequence")
        for application_id in self.application_ids:
            _require_digest(application_id, "application_id")
        if len(set(self.application_ids)) != len(self.application_ids):
            raise ValueError("hard negative application IDs must be unique")
        if self.outcome is HardNegativeOutcome.CONFLICT:
            if not self.conflict_code or self.observed_final_state_digest is not None:
                raise ValueError("conflicting hard negative evidence is incomplete")
            _require_identifier(self.conflict_code, "conflict_code")
        elif self.observed_final_state_digest is None or self.conflict_code is not None:
            raise ValueError("different-result hard negative evidence is incomplete")
        if self.observed_final_state_digest is not None:
            _require_digest(
                self.observed_final_state_digest, "observed_final_state_digest"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "swapped_step_indices": list(self.swapped_step_indices),
            "application_ids": list(self.application_ids),
            "outcome": self.outcome.value,
            "observed_final_state_digest": self.observed_final_state_digest,
            "conflict_code": self.conflict_code,
        }


@dataclass(frozen=True)
class CollapsedInstructionV1:
    pack_id: str
    root_state_id: str
    final_state_id: str
    root_state_digest: str
    final_state_digest: str
    root_ast_digest: str
    final_ast_digest: str
    turn_ids: tuple[str, ...]
    operator_ids: tuple[str, ...]
    applications: tuple[OperatorApplicationV1, ...]
    hard_negatives: tuple[CollapsedHardNegativeV1, ...]
    nl_available: bool = False
    schema: str = "collapsed_instruction/v1"

    def __post_init__(self) -> None:
        _require_identifier(self.pack_id, "pack_id")
        _require_digest(self.root_state_id, "root_state_id")
        _require_digest(self.final_state_id, "final_state_id")
        for field, value in (
            ("root_state_digest", self.root_state_digest),
            ("final_state_digest", self.final_state_digest),
            ("root_ast_digest", self.root_ast_digest),
            ("final_ast_digest", self.final_ast_digest),
        ):
            _require_digest(value, field)
        if len(self.applications) < 2:
            raise ValueError("collapsed instruction requires at least two applications")
        if not (
            len(self.turn_ids)
            == len(self.operator_ids)
            == len(self.applications)
        ):
            raise ValueError("collapsed instruction source sequence lengths differ")
        if len(set(self.turn_ids)) != len(self.turn_ids):
            raise ValueError("collapsed instruction turn IDs must be unique")
        for turn_id in self.turn_ids:
            _require_digest(turn_id, "turn_id")
        for operator_id in self.operator_ids:
            _require_identifier(operator_id, "operator_id")
        if any(not application.succeeded for application in self.applications):
            raise ValueError("collapsed instruction requires successful applications")
        if self.nl_available:
            raise ValueError("NL collapse requires the unavailable CERT_CAP1 path")

    @property
    def application_ids(self) -> tuple[str, ...]:
        return tuple(
            application.application_id for application in self.applications
        )

    @property
    def collapse_id(self) -> str:
        return _fingerprint(self.to_dict(include_collapse_id=False))

    def to_dict(self, *, include_collapse_id: bool = True) -> dict[str, Any]:
        value = {
            "schema": self.schema,
            "pack_id": self.pack_id,
            "root_state_id": self.root_state_id,
            "final_state_id": self.final_state_id,
            "root_state_digest": self.root_state_digest,
            "final_state_digest": self.final_state_digest,
            "root_ast_digest": self.root_ast_digest,
            "final_ast_digest": self.final_ast_digest,
            "turn_ids": list(self.turn_ids),
            "operator_ids": list(self.operator_ids),
            "applications": [
                application.to_dict() for application in self.applications
            ],
            "application_ids": list(self.application_ids),
            "required_order": list(range(len(self.applications))),
            "hard_negatives": [
                negative.to_dict() for negative in self.hard_negatives
            ],
            "nl_available": self.nl_available,
        }
        if include_collapse_id:
            value["collapse_id"] = self.collapse_id
        return value


@dataclass(frozen=True)
class CollapseDecisionV1:
    collapse: CollapsedInstructionV1 | None = None
    rejection: CollapseRejectionKind | None = None
    rejected_step_indices: tuple[int, ...] = ()
    schema: str = "collapse_decision/v1"

    def __post_init__(self) -> None:
        if (self.collapse is None) == (self.rejection is None):
            raise ValueError("collapse decision requires exactly one outcome")
        if self.collapse is not None and self.rejected_step_indices:
            raise ValueError("accepted collapse cannot carry rejected steps")
        if any(index < 0 for index in self.rejected_step_indices):
            raise ValueError("rejected step indices must be non-negative")
        if len(set(self.rejected_step_indices)) != len(
            self.rejected_step_indices
        ):
            raise ValueError("rejected step indices must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "collapse": self.collapse.to_dict() if self.collapse else None,
            "rejection": self.rejection.value if self.rejection else None,
            "rejected_step_indices": list(self.rejected_step_indices),
        }


def replay_collapsed_instruction(
    *,
    authority_resolver: OperatorAuthorityResolver,
    trace: ConversationTraceV1,
    collapse: CollapsedInstructionV1,
) -> ConversationStateNodeV1:
    """Execute only the collapsed sequence and prove its trace boundary."""
    if (
        collapse.pack_id != trace.pack_id
        or collapse.root_state_id != trace.root_state_id
        or collapse.final_state_id != trace.current_state_id
    ):
        raise ConversationTraceError("collapse identity differs from source trace")
    if collapse.turn_ids != tuple(turn.turn_id for turn in trace.turns):
        raise ConversationTraceError("collapse turn lineage differs from source trace")
    if collapse.applications != tuple(
        turn.application for turn in trace.turns
    ):
        raise ConversationTraceError(
            "collapse applications differ from source trace"
        )
    cursor = trace.node(collapse.root_state_id)
    for operator_id, application in zip(
        collapse.operator_ids, collapse.applications, strict=True
    ):
        turn_pack, library = authority_resolver(cursor)
        if turn_pack.pack_id != collapse.pack_id:
            raise ConversationTraceError(
                "collapsed state authority returned another pack"
            )
        if _operator_id(library, application) != operator_id:
            raise ConversationTraceError("collapsed operator identity drifted")
        replayed = library.replay(turn_pack, cursor.state, application)
        if replayed.state is None:
            raise ConversationTraceError("collapsed application did not succeed")
        matching = [
            node
            for node in trace.state_nodes
            if node.parent_state_id == cursor.state_id
            and node.branch_digest == cursor.branch_digest
            and node.state == replayed.state
        ]
        if len(matching) != 1:
            raise ConversationTraceError(
                "collapsed application has no unique source state"
            )
        cursor = matching[0]
    if (
        cursor.state_id != collapse.final_state_id
        or cursor.state.state_digest != collapse.final_state_digest
        or cursor.state.ast_digest != collapse.final_ast_digest
    ):
        raise ConversationTraceError("collapsed final state differs from source trace")
    return cursor


def _operator_id(library, application: OperatorApplicationV1) -> str:
    matches = [
        declaration.operator_id
        for declaration in library.declarations
        if declaration.fingerprint == application.operator_fingerprint
    ]
    if len(matches) != 1:
        raise OperatorReplayError("recorded operator fingerprint is unavailable")
    return matches[0]


def _hard_negative(
    *,
    authority_resolver: OperatorAuthorityResolver,
    trace: ConversationTraceV1,
    applications: tuple[OperatorApplicationV1, ...],
    operator_ids: tuple[str, ...],
    swap_index: int,
) -> CollapsedHardNegativeV1 | None:
    order = list(range(len(applications)))
    order[swap_index], order[swap_index + 1] = (
        order[swap_index + 1],
        order[swap_index],
    )
    cursor = trace.node(trace.root_state_id)
    try:
        for index in order:
            turn_pack, library = authority_resolver(cursor)
            application = applications[index]
            replayed = library.apply(
                turn_pack,
                cursor.state,
                operator_ids[index],
                application.arguments,
                replace(
                    application.provenance,
                    source_artifact_digest=hashlib.sha256(
                        cursor.state.source.encode("utf-8")
                    ).hexdigest(),
                ),
            )
            if not replayed.succeeded or replayed.state is None:
                code = (
                    replayed.application.rejection.code
                    if replayed.application.rejection is not None
                    else "collapse.reordered_rejected"
                )
                raise OperatorReplayError(code)
            matching = [
                node
                for node in trace.state_nodes
                if node.state.state_digest == replayed.state.state_digest
                and node.branch_digest == cursor.branch_digest
            ]
            if len(matching) != 1:
                raise OperatorReplayError("collapse.reordered_state_unavailable")
            cursor = matching[0]
    except (KeyError, OperatorReplayError, ValueError) as exc:
        return CollapsedHardNegativeV1(
            swapped_step_indices=(swap_index, swap_index + 1),
            application_ids=tuple(
                applications[index].application_id for index in order
            ),
            outcome=HardNegativeOutcome.CONFLICT,
            conflict_code=str(exc) or type(exc).__name__,
        )
    final = trace.node(trace.current_state_id)
    if cursor.state.state_digest == final.state.state_digest:
        return None
    return CollapsedHardNegativeV1(
        swapped_step_indices=(swap_index, swap_index + 1),
        application_ids=tuple(
            applications[index].application_id for index in order
        ),
        outcome=HardNegativeOutcome.DIFFERENT_RESULT,
        observed_final_state_digest=cursor.state.state_digest,
    )


def collapse_conversation_trace(
    *,
    pack: DslPack,
    authority_resolver: OperatorAuthorityResolver,
    trace: ConversationTraceV1,
) -> CollapseDecisionV1:
    """Admit only exact ordered AST-edit traces with replayed hard negatives."""
    replay_conversation_trace(
        pack=pack, authority_resolver=authority_resolver, trace=trace
    )
    if len(trace.turns) < 2:
        return CollapseDecisionV1(rejection=CollapseRejectionKind.TOO_SHORT)
    history = tuple(
        index
        for index, turn in enumerate(trace.turns)
        if turn.operation is not ConversationOperation.AST_EDIT
    )
    if history:
        return CollapseDecisionV1(
            rejection=CollapseRejectionKind.HISTORY_OPERATION,
            rejected_step_indices=history,
        )
    applications = tuple(
        turn.application for turn in trace.turns if turn.application is not None
    )
    seen_states = {trace.node(trace.root_state_id).state.state_digest}
    seen_applications: set[str] = set()
    no_ops: list[int] = []
    cycles: list[int] = []
    redundant: list[int] = []
    operator_ids: list[str] = []
    for index, (turn, application) in enumerate(
        zip(trace.turns, applications, strict=True)
    ):
        input_node = trace.node(turn.input_state_id)
        output_node = trace.node(turn.output_state_id)
        _, library = authority_resolver(input_node)
        operator_ids.append(_operator_id(library, application))
        if input_node.state.state_digest == output_node.state.state_digest:
            no_ops.append(index)
        if output_node.state.state_digest in seen_states:
            cycles.append(index)
        if application.application_id in seen_applications:
            redundant.append(index)
        seen_states.add(output_node.state.state_digest)
        seen_applications.add(application.application_id)
    for kind, indices in (
        (CollapseRejectionKind.NO_OP, no_ops),
        (CollapseRejectionKind.CYCLE, cycles),
        (CollapseRejectionKind.REDUNDANT, redundant),
    ):
        if indices:
            return CollapseDecisionV1(
                rejection=kind, rejected_step_indices=tuple(indices)
            )
    hard_negatives = []
    for index in range(len(applications) - 1):
        input_node = trace.node(trace.turns[index].input_state_id)
        _, library = authority_resolver(input_node)
        left = library.lookup(operator_ids[index])
        right = library.lookup(operator_ids[index + 1])
        mutually_commuting = (
            right.operator_id in left.commutes_with
            and left.operator_id in right.commutes_with
        )
        if not mutually_commuting:
            negative = _hard_negative(
                authority_resolver=authority_resolver,
                trace=trace,
                applications=applications,
                operator_ids=tuple(operator_ids),
                swap_index=index,
            )
            if negative is not None:
                hard_negatives.append(negative)
    root = trace.node(trace.root_state_id)
    final = trace.node(trace.current_state_id)
    if final.state != replay_conversation_trace(
        pack=pack, authority_resolver=authority_resolver, trace=trace
    ).state:
        return CollapseDecisionV1(
            rejection=CollapseRejectionKind.REPLAY_MISMATCH
        )
    collapsed = CollapsedInstructionV1(
        pack_id=trace.pack_id,
        root_state_id=trace.root_state_id,
        final_state_id=trace.current_state_id,
        root_state_digest=root.state.state_digest,
        final_state_digest=final.state.state_digest,
        root_ast_digest=root.state.ast_digest,
        final_ast_digest=final.state.ast_digest,
        turn_ids=tuple(turn.turn_id for turn in trace.turns),
        operator_ids=tuple(operator_ids),
        applications=applications,
        hard_negatives=tuple(hard_negatives),
    )
    replay_collapsed_instruction(
        authority_resolver=authority_resolver,
        trace=trace,
        collapse=collapsed,
    )
    return CollapseDecisionV1(collapse=collapsed)


__all__ = [
    "CollapseDecisionV1",
    "CollapseRejectionKind",
    "CollapsedHardNegativeV1",
    "CollapsedInstructionV1",
    "HardNegativeOutcome",
    "collapse_conversation_trace",
    "replay_collapsed_instruction",
]
