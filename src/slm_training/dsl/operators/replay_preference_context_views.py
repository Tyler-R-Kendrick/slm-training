"""SLM-418 (DSH5-10): turn-depth / context-view ablation dimensions.

``replay_preference.py``'s own module docstring and its
``OperatorEventMemoryReportV1`` docstring both flag the same gap: "SLM-418's
full report also requires turn-depth and context-view ablations against a
trained policy; those are out of scope for this slice." This module closes
the *structural* half of that gap -- it defines the five context-view input
representations the issue's own "Compare current state only, state plus
recent semantic receipts, state plus retrieved relevant events, complete
transcript plus state, and existing last-three-text-history baseline" bullet
asks for, and the five turn-depths (1/2/4/8/16) its matrix names, as precise
functions of already-computed state/event ids.

Every view below is built ONLY from state ids, output-state ids, and a small
closed structural ``action_kind`` vocabulary derived by string-prefix check on
an already-serialized legal action (``action_kind_of``) -- never from
transcript text as authority, and never a semantic embedding of any id (ids
are join keys only, per the issue's own explicit ban). This module trains
nothing; :mod:`slm_training.harnesses.preference.
replay_preference_context_view_variants` consumes it to train and compare
matched context-view variants, and
:class:`OperatorEventMemoryAblationReportV1` below records the *structural*
ablation grid (how much history each cell actually exposes) that a trained
comparison can be laid on top of without recomputing it.

Explicitly out of scope here (see the harness and eval modules for what
picks each item up, and the disposition doc for what stays unattempted):
learned scoring, SFT/preference training, held-out benefit measurement, and
CAP0/CAP1/CAP2 retention.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from slm_training.dsl.operators.conversation import (
    ConversationOperation,
    ConversationStateNodeV1,
    ConversationTraceV1,
)
from slm_training.dsl.operators.replay_preference import (
    OperatorEventMemoryReportV1,
    action_kind_of,  # noqa: F401 -- re-exported unchanged for existing callers
)
from slm_training.harness_core.versioning import build_version_stamp

#: The issue's own matrix: "1/2/4/8/16 turn depth."
TURN_DEPTHS: tuple[int, ...] = (1, 2, 4, 8, 16)

#: The issue's own phrase: "existing last-three-text-history baseline." Fixed
#: at 3 regardless of the requested ``turn_depth`` -- see
#: ``ContextView.LAST_THREE_TEXT_HISTORY`` below.
LAST_THREE_WINDOW: int = 3


class ContextView(str, Enum):
    """The five context-view input representations the issue's own bullet names.

    Each member's exact construction is documented on its branch inside
    ``build_context_view_window``; the summary:

    * ``CURRENT_STATE_ONLY`` -- no history receipt is ever visible.
    * ``STATE_PLUS_RECENT_RECEIPTS`` -- the last ``turn_depth`` receipts in
      the trace's own chronological turn order (may cross branch boundaries).
    * ``STATE_PLUS_RETRIEVED_EVENTS`` -- the last ``turn_depth`` receipts on
      the single-branch ancestry path that causally produced the decision
      state (walked via ``parent_state_id``; never merely recent).
    * ``FULL_TRACE_PLUS_STATE`` -- every receipt before the decision;
      ``turn_depth`` is recorded but always saturates.
    * ``LAST_THREE_TEXT_HISTORY`` -- a fixed 3-receipt window with state ids
      stripped, carrying only the structural ``action_kind`` token, so it
      can never reconstruct exact state by construction (the issue's own
      adversarial control: "text history cannot reconstruct a different
      state than the DAG").
    """

    CURRENT_STATE_ONLY = "current_state_only"
    STATE_PLUS_RECENT_RECEIPTS = "state_plus_recent_receipts"
    STATE_PLUS_RETRIEVED_EVENTS = "state_plus_retrieved_events"
    FULL_TRACE_PLUS_STATE = "full_trace_plus_state"
    LAST_THREE_TEXT_HISTORY = "last_three_text_history"


_OPERATION_ACTION_KIND: dict[ConversationOperation, str] = {
    ConversationOperation.UNDO: "undo",
    ConversationOperation.REDO: "redo",
    ConversationOperation.CHECKOUT_STATE: "checkout",
    ConversationOperation.FORK: "fork",
    ConversationOperation.AST_EDIT: "operator",
}


# ``action_kind_of`` moved to ``replay_preference.py`` (tenth slice), which
# now also uses it internally for ``_pick_rejected``'s same-domain
# preference; re-exported here unchanged so every existing import of
# ``slm_training.dsl.operators.replay_preference_context_views.action_kind_of``
# keeps working byte-for-byte.


@dataclass(frozen=True)
class OperatorTurnReceiptV1:
    """One opaque join-key receipt for a single recorded conversation turn.

    Never carries transcript text or a semantic label: ``action_kind`` is
    derived structurally (``_OPERATION_ACTION_KIND``), and the two state ids
    are the same content-addressed ids ``OperatorReplayPreferenceRowV1``
    itself uses.
    """

    turn_index: int
    action_kind: str
    input_state_id: str
    output_state_id: str
    schema: str = "operator_turn_receipt/v1"

    @property
    def receipt_id(self) -> str:
        return f"{self.action_kind}:{self.input_state_id}:{self.output_state_id}"

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "turn_index": self.turn_index,
            "action_kind": self.action_kind,
            "input_state_id": self.input_state_id,
            "output_state_id": self.output_state_id,
        }


def receipts_from_trace(trace: ConversationTraceV1) -> tuple[OperatorTurnReceiptV1, ...]:
    """Ordered, opaque turn receipts built directly from a trace's own recorded turns."""
    receipts = []
    for index, turn in enumerate(trace.turns):
        kind = _OPERATION_ACTION_KIND.get(turn.operation, "operator")
        receipts.append(
            OperatorTurnReceiptV1(
                turn_index=index,
                action_kind=kind,
                input_state_id=turn.input_state_id,
                output_state_id=turn.output_state_id,
            )
        )
    return tuple(receipts)


def state_lookup_from_trace(
    trace: ConversationTraceV1,
) -> dict[str, ConversationStateNodeV1]:
    """A ``state_id -> node`` map for ancestry walks (``STATE_PLUS_RETRIEVED_EVENTS``)."""
    return {node.state_id: node for node in trace.state_nodes}


@dataclass(frozen=True)
class ContextViewWindowV1:
    """One context-view materialization of visible history for one decision state.

    ``visible_receipt_ids`` are opaque tokens only: either a full receipt id
    (``"<kind>:<input_state_id>:<output_state_id>"``) or, for
    ``LAST_THREE_TEXT_HISTORY``, a state-stripped ``"kind:<action_kind>"``
    token. Neither form is transcript text or a semantic embedding.
    """

    view: ContextView
    turn_depth: int
    decision_state_id: str
    visible_receipt_ids: tuple[str, ...]
    schema: str = "operator_context_view_window/v1"

    def visible_action_kinds(self) -> tuple[str, ...]:
        """The structural ``action_kind`` token of each visible receipt, in order."""
        kinds = []
        for token in self.visible_receipt_ids:
            if token.startswith("kind:"):
                kinds.append(token.split(":", 1)[1])
            else:
                kinds.append(token.split(":", 1)[0])
        return tuple(kinds)

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "view": self.view.value,
            "turn_depth": self.turn_depth,
            "decision_state_id": self.decision_state_id,
            "visible_receipt_ids": list(self.visible_receipt_ids),
        }


def _decision_position(
    receipts: Sequence[OperatorTurnReceiptV1],
    decision_state_id: str,
    chosen_output_state_id: str,
) -> int:
    """Number of receipts chronologically strictly before the decision turn.

    A decision state can recur as the ``input_state_id`` of more than one
    receipt -- e.g. once when a chain of edits first passes through it, and
    again later when a rollback undoes back to it -- so matching on
    ``input_state_id`` alone is ambiguous. The row's own recorded
    ``chosen_output_state_id`` disambiguates: it is always the exact
    ``output_state_id`` of the one specific turn this row describes (every
    one of the seven named patterns derives ``chosen_output_state_id`` from
    a real recorded turn's own ``output_state_id``, or -- for
    ``MERGE_SUCCESS`` -- from no turn at all, handled by the fallback below).
    """
    for receipt in receipts:
        if (
            receipt.input_state_id == decision_state_id
            and receipt.output_state_id == chosen_output_state_id
        ):
            return receipt.turn_index
    # No exact (input, output) match: true for MERGE_SUCCESS rows, whose
    # decision state is a branch tip no further trace turn ever consumes as
    # an input at all -- every supplied receipt is prior history. Falls back
    # to a same-input match (kept for robustness) before that.
    for receipt in receipts:
        if receipt.input_state_id == decision_state_id:
            return receipt.turn_index
    return len(receipts)


def build_context_view_window(
    *,
    state_lookup: Mapping[str, ConversationStateNodeV1],
    receipts: Sequence[OperatorTurnReceiptV1],
    decision_state_id: str,
    chosen_output_state_id: str,
    view: ContextView,
    turn_depth: int,
) -> ContextViewWindowV1:
    """The exact, precisely-scoped set of history receipts visible under ``view``.

    Every branch is built only from already-computed state/event ids and
    structural ``action_kind`` tokens -- never transcript text, never a
    semantic embedding of any id. ``chosen_output_state_id`` (the row's own
    recorded destination) disambiguates *which* occurrence of a revisited
    decision state this call is about -- see ``_decision_position``.
    """
    if turn_depth not in TURN_DEPTHS:
        raise ValueError(f"turn_depth must be one of {TURN_DEPTHS}, got {turn_depth}")

    if view is ContextView.CURRENT_STATE_ONLY:
        # By definition: no history receipt is ever visible, at any depth.
        visible: tuple[OperatorTurnReceiptV1, ...] = ()

    elif view is ContextView.STATE_PLUS_RECENT_RECEIPTS:
        # Chronological recency in the trace's own recorded turn order --
        # may cross branch boundaries (e.g. right after a fork/checkout).
        position = _decision_position(receipts, decision_state_id, chosen_output_state_id)
        visible = tuple(receipts[max(0, position - turn_depth) : position])

    elif view is ContextView.STATE_PLUS_RETRIEVED_EVENTS:
        # Structural ancestry: only the single-branch lineage that causally
        # produced the decision state, walked via parent_state_id -- distinct
        # from recency because it excludes any chronologically-recent turn
        # that happened on a sibling branch. Keyed by the exact (input,
        # output) edge, not output alone: a state's parent_state_id is fixed
        # at node-creation time, so the edge key (parent, state) always
        # resolves to the one turn that originally *created* the state, even
        # when a later undo/redo/checkout also produces that same state as
        # some other turn's output.
        by_edge = {(r.input_state_id, r.output_state_id): r for r in receipts}
        ancestry: list[OperatorTurnReceiptV1] = []
        cursor = decision_state_id
        for _ in range(turn_depth):
            node = state_lookup.get(cursor)
            if node is None or node.parent_state_id is None:
                break
            receipt = by_edge.get((node.parent_state_id, cursor))
            if receipt is None:
                break
            ancestry.append(receipt)
            cursor = node.parent_state_id
        visible = tuple(reversed(ancestry))

    elif view is ContextView.FULL_TRACE_PLUS_STATE:
        # The complete transcript; turn_depth is recorded but always
        # saturates -- this view has no shorter-window meaning.
        position = _decision_position(receipts, decision_state_id, chosen_output_state_id)
        visible = tuple(receipts[:position])

    elif view is ContextView.LAST_THREE_TEXT_HISTORY:
        # The issue's own "existing last-three-text-history baseline" --
        # fixed at LAST_THREE_WINDOW (turn_depth is recorded but not
        # applied), and deliberately state-blind: only the action-kind
        # token survives, so this view can never reconstruct exact state,
        # by construction.
        position = _decision_position(receipts, decision_state_id, chosen_output_state_id)
        windowed = receipts[max(0, position - LAST_THREE_WINDOW) : position]
        return ContextViewWindowV1(
            view=view,
            turn_depth=turn_depth,
            decision_state_id=decision_state_id,
            visible_receipt_ids=tuple(f"kind:{r.action_kind}" for r in windowed),
        )
    else:  # pragma: no cover - exhaustive Enum
        raise ValueError(f"unknown context view {view!r}")

    return ContextViewWindowV1(
        view=view,
        turn_depth=turn_depth,
        decision_state_id=decision_state_id,
        visible_receipt_ids=tuple(r.receipt_id for r in visible),
    )


@dataclass(frozen=True)
class OperatorEventMemoryAblationCellV1:
    """One (row, context_view, turn_depth) cell of the structural ablation grid."""

    row_input_state_id: str
    semantic_relation: str
    view: ContextView
    turn_depth: int
    visible_receipt_count: int
    most_recent_action_kind: str | None
    schema: str = "operator_event_memory_ablation_cell/v1"

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "row_input_state_id": self.row_input_state_id,
            "semantic_relation": self.semantic_relation,
            "view": self.view.value,
            "turn_depth": self.turn_depth,
            "visible_receipt_count": self.visible_receipt_count,
            "most_recent_action_kind": self.most_recent_action_kind,
        }


@dataclass(frozen=True)
class OperatorEventMemoryAblationReportV1:
    """The turn-depth x context-view structural ablation grid over one report's rows.

    Structural only -- for each row and each ``(view, turn_depth)`` cell,
    records how many history receipts that combination actually exposes and
    the action-kind of the most recently visible one (or ``None``). This is
    NOT a trained-policy comparison; it is the dimension SLM-418's own
    ``OperatorEventMemoryReportV1`` docstring flagged as missing, built once
    so :mod:`slm_training.harnesses.preference.
    replay_preference_context_view_variants` can attach trained comparison
    metrics without recomputing these windows.
    """

    base_report: OperatorEventMemoryReportV1
    cells: tuple[OperatorEventMemoryAblationCellV1, ...]
    version_stamp: dict
    schema: str = "operator_event_memory_ablation_report/v1"

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "base_report": self.base_report.to_dict(),
            "cells": [cell.to_dict() for cell in self.cells],
            "version_stamp": self.version_stamp,
        }


def build_ablation_report(
    report: OperatorEventMemoryReportV1,
    *,
    state_lookup: Mapping[str, ConversationStateNodeV1],
    receipts: Sequence[OperatorTurnReceiptV1],
    views: Sequence[ContextView] = tuple(ContextView),
    turn_depths: Sequence[int] = TURN_DEPTHS,
) -> OperatorEventMemoryAblationReportV1:
    """Materialize the structural ablation grid for every row in ``report``."""
    cells: list[OperatorEventMemoryAblationCellV1] = []
    for row in report.rows:
        for view in views:
            for depth in turn_depths:
                window = build_context_view_window(
                    state_lookup=state_lookup,
                    receipts=receipts,
                    decision_state_id=row.input_state_id,
                    chosen_output_state_id=row.chosen_output_state_id,
                    view=view,
                    turn_depth=depth,
                )
                kinds = window.visible_action_kinds()
                cells.append(
                    OperatorEventMemoryAblationCellV1(
                        row_input_state_id=row.input_state_id,
                        semantic_relation=row.semantic_relation.value,
                        view=view,
                        turn_depth=depth,
                        visible_receipt_count=len(kinds),
                        most_recent_action_kind=kinds[-1] if kinds else None,
                    )
                )
    return OperatorEventMemoryAblationReportV1(
        base_report=report,
        cells=tuple(cells),
        version_stamp=build_version_stamp(
            "dsl.operators.replay_preference_context_views"
        ),
    )
