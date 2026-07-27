"""SLM-418 (DSH5-10): render replay-grounded rows into trainable preference pairs.

``extract_replay_preference_rows`` / ``extract_merge_preference_row``
(``slm_training.dsl.operators.replay_preference``) already ground every
``chosen_action``/``rejected_action`` in the exact legal set available at
one real conversation state. This module is the first slice of the
remaining SLM-418 gap those functions themselves document: nothing in the
repo converts a row into the ``PreferencePair`` shape
``scripts/train_preference.py``'s ``build-pairs``/``train`` path consumes.
It does not run any training and makes no held-out claim -- see
``docs/design/dsh5-10-replay-preference-rows.md`` for the full disposition.

Rendering never fabricates a state. ``chosen_output_state_id`` is always an
already-materialized state node by construction. The rejected side is
resolved the same way, using only information the caller's own node
resolver (or one real, pack-authorized ``OperatorLibraryV1.apply`` call)
already provides:

* ``undo`` -> the input state's own parent (an existing node).
* ``redo:<state_id>`` / ``checkout:<state_id>`` -> that state_id is already
  a materialized node -- no computation needed.
* an operator action's serialized form -> re-derived by recomputing the
  exact legal set at the input state (``enumerate_operator_legal_set``,
  the same call ``extract_replay_preference_rows`` itself makes) and
  actually applying the matching legal action through the pack-authorized
  ``library.apply`` -- the same executor every other application in this
  module goes through, not a shortcut.
* ``merge:<pair>`` -> deliberately unrendered. It is never the rejected
  side under the current single-pair extraction (only ever the *chosen*
  side of a ``MERGE_SUCCESS`` row, whose state is always resolvable via
  ``chosen_output_state_id`` like any other row), but is left honestly
  unhandled here rather than guessed at if that ever changes.

Two call shapes share this rendering, since replay-preference rows come
from two different sources with two different notions of "the trace":
``render_replay_preference_pair(row, resolve_node=trace.node, ...)`` for
rows from ``extract_replay_preference_rows`` (a real
``ConversationTraceV1``), and ``merge_node_resolver(left, right,
decision)`` to build the equivalent resolver for a ``MERGE_SUCCESS`` row
from ``extract_merge_preference_row``, whose ``left``/``right``
``BranchEditV1`` edges and ``BranchMergeDecisionV1`` are never appended to
a shared trace.

The pair's ``prompt`` is the input state's own source: this is a first,
documented modeling choice (score chosen/rejected *continuations* of the
current AST against it), not a validated training-objective decision --
whoever wires an actual training run should treat it as open, especially
against the DSH3 policy head (``typed_operator_policy.py``) the disposition
doc's remaining-scope note names, not just the generic TwoTower pair
format used here for compatibility with existing tooling.
"""

from __future__ import annotations

from collections.abc import Callable

from slm_training.dsl.operators.conversation import ConversationStateNodeV1
from slm_training.dsl.operators.legal_set import enumerate_operator_legal_set
from slm_training.dsl.operators.merge import BranchEditV1, BranchMergeDecisionV1
from slm_training.dsl.operators.registry import OperatorLibraryV1
from slm_training.dsl.operators.replay_preference import (
    OperatorReplayPreferenceRowV1,
    ProvenanceFactory,
)
from slm_training.dsl.pack import DslPack
from slm_training.harnesses.preference import PreferencePair

NodeResolver = Callable[[str], ConversationStateNodeV1]


def merge_node_resolver(
    left: BranchEditV1, right: BranchEditV1, decision: BranchMergeDecisionV1
) -> NodeResolver:
    """A ``NodeResolver`` over exactly the nodes one merge attempt exposes."""
    nodes: dict[str, ConversationStateNodeV1] = {
        left.input_node.state_id: left.input_node,
        left.output_node.state_id: left.output_node,
        right.input_node.state_id: right.input_node,
        right.output_node.state_id: right.output_node,
    }
    if decision.continuation is not None:
        merged = decision.continuation.merged_node
        nodes[merged.state_id] = merged

    def resolve(state_id: str) -> ConversationStateNodeV1:
        try:
            return nodes[state_id]
        except KeyError as exc:
            raise KeyError(f"unknown merge state ID {state_id!r}") from exc

    return resolve


def render_replay_preference_pair(
    row: OperatorReplayPreferenceRowV1,
    *,
    resolve_node: NodeResolver,
    pack: DslPack,
    library: OperatorLibraryV1,
    provenance_for: ProvenanceFactory,
) -> PreferencePair | None:
    """Render one replay-grounded row into a ``PreferencePair``, or ``None``.

    Returns ``None`` rather than fabricating a rejected state whenever the
    rejected action is ``merge:<pair>`` (out of scope; see module
    docstring), its target legal action cannot be re-derived from the exact
    legal set at ``row.input_state_id`` (would indicate a stale/inconsistent
    row), or the re-applied rejected state is identical to the chosen state
    (degenerate; not a real preference).
    """
    input_node = resolve_node(row.input_state_id)
    chosen_text = resolve_node(row.chosen_output_state_id).state.source

    action = row.rejected_action
    if action == "undo":
        if input_node.parent_state_id is None:
            return None
        rejected_text = resolve_node(input_node.parent_state_id).state.source
    elif action.startswith("redo:") or action.startswith("checkout:"):
        target_state_id = action.split(":", 1)[1]
        rejected_text = resolve_node(target_state_id).state.source
    elif action.startswith("merge:"):
        return None
    else:
        legal_set = enumerate_operator_legal_set(
            pack=pack,
            library=library,
            state=input_node.state,
            reference_table=input_node.reference_table,
            provenance=provenance_for(input_node.state),
            ordinary_nonoperator_actions=(),
        )
        match = next(
            (
                candidate
                for candidate in legal_set.operator_actions
                if candidate.serialized == action
            ),
            None,
        )
        if match is None:
            return None
        result = library.apply(
            pack,
            input_node.state,
            match.operator_id,
            match.arguments,
            provenance_for(input_node.state),
        )
        if not result.succeeded or result.state is None:
            return None
        rejected_text = result.state.source

    if rejected_text == chosen_text:
        return None

    return PreferencePair(
        prompt=input_node.state.source,
        chosen=chosen_text,
        rejected=rejected_text,
        meta={
            "pair_corpus": "replay_preference",
            "schema": "replay_preference_pair/v1",
            "semantic_relation": row.semantic_relation.value,
            "correction_reason": row.correction_reason,
            "input_state_id": row.input_state_id,
            "chosen_action": row.chosen_action,
            "rejected_action": row.rejected_action,
            "legal_set_fingerprint": row.legal_set_fingerprint,
        },
    )


def render_replay_preference_pairs(
    rows: tuple[OperatorReplayPreferenceRowV1, ...],
    *,
    resolve_node: NodeResolver,
    pack: DslPack,
    library: OperatorLibraryV1,
    provenance_for: ProvenanceFactory,
) -> list[PreferencePair]:
    """Render every row that can honestly be rendered; silently drops the rest.

    "Silently" here means no exception -- callers that need to know how
    many rows were dropped (and why) should call
    ``render_replay_preference_pair`` per-row themselves, since this
    function intentionally mirrors the shape ``write_pairs`` expects.
    """
    pairs: list[PreferencePair] = []
    for row in rows:
        pair = render_replay_preference_pair(
            row,
            resolve_node=resolve_node,
            pack=pack,
            library=library,
            provenance_for=provenance_for,
        )
        if pair is not None:
            pairs.append(pair)
    return pairs
