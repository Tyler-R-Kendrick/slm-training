"""SLM-418 (DSH5-10): replay-grounded preference rows from undo/redo history.

Builds versioned preference rows over one exact input state from four
verified conversation patterns -- edit-then-undo, undo-then-redo, partial
rollback (a ``checkout`` to a strict ancestor), and checkout-another-state (a
``checkout`` to a state that is not an ancestor) -- where the chosen and
rejected control-or-operator actions are checked against the exact legal set
available at that state (``enumerate_operator_legal_set``), never against
transcript text.

This is an honestly partial slice of SLM-418. It does not implement:
fork-then-choose-one-branch, merge success/conflict, or pronoun/focus
follow-up patterns; it does not train an SFT/preference variant, measure
held-out benefit, or produce turn-depth / context-view ablations. See
``docs/design/dsh5-10-replay-preference-rows.md`` for the full disposition
and the remaining scope.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from slm_training.dsl.operators.conversation import (
    ConversationOperation,
    ConversationTraceV1,
)
from slm_training.dsl.operators.contracts import ApplicationProvenanceV1
from slm_training.dsl.operators.legal_set import (
    OperatorLegalSetV1,
    enumerate_operator_legal_set,
)
from slm_training.dsl.operators.registry import OperatorLibraryV1, OperatorStateV1
from slm_training.dsl.pack import DslPack
from slm_training.harness_core.versioning import build_version_stamp

ProvenanceFactory = Callable[[OperatorStateV1], ApplicationProvenanceV1]


class ReplayPreferenceRelation(str, Enum):
    EDIT_THEN_UNDO = "edit_then_undo"
    UNDO_THEN_REDO = "undo_then_redo"
    PARTIAL_ROLLBACK = "partial_rollback"
    CHECKOUT_OTHER_STATE = "checkout_other_state"


@dataclass(frozen=True)
class OperatorReplayPreferenceRowV1:
    """One replay-grounded preference row over a single exact input state.

    ``chosen_action``/``rejected_action`` are serialized members of the
    legal set computed at ``input_state_id`` (``OperatorLegalSetV1``), never
    derived from transcript text.
    """

    input_state_id: str
    chosen_action: str
    rejected_action: str
    chosen_output_state_id: str
    semantic_relation: ReplayPreferenceRelation
    correction_reason: str
    legal_set_fingerprint: str
    schema: str = "operator_replay_preference_row/v1"

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "input_state_id": self.input_state_id,
            "chosen_action": self.chosen_action,
            "rejected_action": self.rejected_action,
            "chosen_output_state_id": self.chosen_output_state_id,
            "semantic_relation": self.semantic_relation.value,
            "correction_reason": self.correction_reason,
            "legal_set_fingerprint": self.legal_set_fingerprint,
        }


@dataclass(frozen=True)
class OperatorEventMemoryReportV1:
    """Counts of replay-grounded preference rows extracted from one trace.

    SLM-418's full report also requires turn-depth and context-view
    ablations against a trained policy; those are out of scope for this
    slice (see the disposition doc).
    """

    rows: tuple[OperatorReplayPreferenceRowV1, ...]
    version_stamp: dict
    schema: str = "operator_event_memory_report/v1"

    @property
    def counts_by_relation(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in self.rows:
            key = row.semantic_relation.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "rows": [row.to_dict() for row in self.rows],
            "counts_by_relation": self.counts_by_relation,
            "version_stamp": self.version_stamp,
        }


def _available_history_actions(
    trace: ConversationTraceV1, state_id: str
) -> tuple[str, ...]:
    """Control actions available at ``state_id`` from the trace's own DAG.

    ``undo`` (a parent exists), ``redo:<child>`` (an already-materialized
    child on the current branch), and ``checkout:<other_state>`` (any other
    state in the trace, mirroring ``control_actions.py``'s own
    ``ConversationControlKind.CHECKOUT`` enumeration -- every node except the
    current one, regardless of branch or ancestor relationship) are modeled
    here. fork/copy availability is still out of scope for this slice.
    """
    node = trace.node(state_id)
    actions: list[str] = []
    if node.parent_state_id is not None:
        actions.append("undo")
    for other in trace.state_nodes:
        if (
            other.parent_state_id == state_id
            and other.branch_digest == node.branch_digest
        ):
            actions.append(f"redo:{other.state_id}")
    for other in sorted(trace.state_nodes, key=lambda candidate: candidate.state_id):
        if other.state_id != state_id:
            actions.append(f"checkout:{other.state_id}")
    return tuple(actions)


def _is_strict_ancestor(
    trace: ConversationTraceV1, ancestor_state_id: str, of_state_id: str
) -> bool:
    """Whether ``ancestor_state_id`` lies on ``of_state_id``'s parent chain.

    Walks the immutable parent chain (never transcript text) to distinguish
    a **partial rollback** (checkout to a strict ancestor -- the same
    direction ``undo`` moves in, just possibly more than one hop) from a
    **checkout to another state** (any other target: a sibling, a cousin on
    another branch, or a descendant).
    """
    cursor = trace.node(of_state_id).parent_state_id
    while cursor is not None:
        if cursor == ancestor_state_id:
            return True
        cursor = trace.node(cursor).parent_state_id
    return False


def _legal_set_at(
    trace: ConversationTraceV1,
    *,
    pack: DslPack,
    library: OperatorLibraryV1,
    state_id: str,
    provenance_for: ProvenanceFactory,
) -> OperatorLegalSetV1:
    node = trace.node(state_id)
    return enumerate_operator_legal_set(
        pack=pack,
        library=library,
        state=node.state,
        reference_table=node.reference_table,
        provenance=provenance_for(node.state),
        ordinary_nonoperator_actions=_available_history_actions(trace, state_id),
    )


def _pick_rejected(legal_set: OperatorLegalSetV1, chosen: str) -> str | None:
    """A deterministic legal alternative to ``chosen``, or ``None`` if none exists.

    Drawn from the full legal set -- operator actions and available history
    controls (``undo``/``redo:<state>``) alike -- so a row is never dropped
    just because the only unchosen alternative happens to be a control.
    """
    candidates = sorted(
        action for action in legal_set.all_serialized_actions if action != chosen
    )
    return candidates[0] if candidates else None


def extract_replay_preference_rows(
    trace: ConversationTraceV1,
    *,
    pack: DslPack,
    library: OperatorLibraryV1,
    provenance_for: ProvenanceFactory,
) -> OperatorEventMemoryReportV1:
    """Scan a trace for edit-then-undo, undo-then-redo, partial-rollback, and
    checkout-another-state patterns.

    Each match produces one row whose chosen and rejected actions are both
    verified members of the exact legal set at the shared input state. A
    row is only emitted when an unchosen alternative actually exists in that
    legal set -- undo/redo/checkout is never asserted preferred by default.
    """
    rows: list[OperatorReplayPreferenceRowV1] = []
    turns = trace.turns

    for turn in turns:
        if turn.operation is not ConversationOperation.CHECKOUT_STATE:
            continue
        decision_state_id = turn.input_state_id
        legal_set = _legal_set_at(
            trace,
            pack=pack,
            library=library,
            state_id=decision_state_id,
            provenance_for=provenance_for,
        )
        chosen = f"checkout:{turn.output_state_id}"
        rejected = _pick_rejected(legal_set, chosen)
        if chosen not in legal_set.all_serialized_actions or rejected is None:
            continue
        if _is_strict_ancestor(trace, turn.output_state_id, decision_state_id):
            relation = ReplayPreferenceRelation.PARTIAL_ROLLBACK
            correction_reason = "user_rolled_back_partially"
        else:
            relation = ReplayPreferenceRelation.CHECKOUT_OTHER_STATE
            correction_reason = "user_checked_out_alternate_state"
        rows.append(
            OperatorReplayPreferenceRowV1(
                input_state_id=decision_state_id,
                chosen_action=chosen,
                rejected_action=rejected,
                chosen_output_state_id=turn.output_state_id,
                semantic_relation=relation,
                correction_reason=correction_reason,
                legal_set_fingerprint=legal_set.fingerprint,
            )
        )

    for index in range(len(turns) - 1):
        current, following = turns[index], turns[index + 1]

        if (
            current.operation is ConversationOperation.AST_EDIT
            and following.operation is ConversationOperation.UNDO
            and following.input_state_id == current.output_state_id
        ):
            decision_state_id = current.output_state_id
            legal_set = _legal_set_at(
                trace,
                pack=pack,
                library=library,
                state_id=decision_state_id,
                provenance_for=provenance_for,
            )
            rejected = _pick_rejected(legal_set, "undo")
            if "undo" in legal_set.all_serialized_actions and rejected is not None:
                rows.append(
                    OperatorReplayPreferenceRowV1(
                        input_state_id=decision_state_id,
                        chosen_action="undo",
                        rejected_action=rejected,
                        chosen_output_state_id=following.output_state_id,
                        semantic_relation=ReplayPreferenceRelation.EDIT_THEN_UNDO,
                        correction_reason="user_undid_without_redo",
                        legal_set_fingerprint=legal_set.fingerprint,
                    )
                )

        if (
            current.operation is ConversationOperation.UNDO
            and following.operation is ConversationOperation.REDO
            and following.input_state_id == current.output_state_id
        ):
            decision_state_id = current.output_state_id
            legal_set = _legal_set_at(
                trace,
                pack=pack,
                library=library,
                state_id=decision_state_id,
                provenance_for=provenance_for,
            )
            chosen = f"redo:{following.output_state_id}"
            rejected = _pick_rejected(legal_set, chosen)
            if chosen in legal_set.all_serialized_actions and rejected is not None:
                rows.append(
                    OperatorReplayPreferenceRowV1(
                        input_state_id=decision_state_id,
                        chosen_action=chosen,
                        rejected_action=rejected,
                        chosen_output_state_id=following.output_state_id,
                        semantic_relation=ReplayPreferenceRelation.UNDO_THEN_REDO,
                        correction_reason="user_redid_after_reconsidering",
                        legal_set_fingerprint=legal_set.fingerprint,
                    )
                )

    return OperatorEventMemoryReportV1(
        rows=tuple(rows),
        version_stamp=build_version_stamp("dsl.operators.replay_preference"),
    )
