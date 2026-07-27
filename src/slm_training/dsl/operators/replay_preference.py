"""SLM-418 (DSH5-10): replay-grounded preference rows from undo/redo history.

Builds versioned preference rows over one exact input state from all seven
verified conversation patterns -- edit-then-undo, undo-then-redo, partial
rollback (a second or later consecutive undo, chosen over redo/checkout/edit
alternatives), checkout-another-state, fork-then-choose-one-branch,
merge-success, and pronoun-focus-followup -- where the chosen and rejected
control-or-operator actions are checked against the exact legal set
available at that state (``enumerate_operator_legal_set``), never against
transcript text.

This is an honestly partial slice of SLM-418. All seven named extraction
patterns now exist, but it does not train an SFT/preference variant,
measure held-out benefit, or produce turn-depth / context-view ablations.
See ``docs/design/dsh5-10-replay-preference-rows.md`` for the full
disposition and the remaining scope.

Merge conflict (the other half of the issue's "merge success/conflict"
pattern) is deliberately *not* modeled as a preference row here: unlike
undo/redo/checkout/fork, a merge attempt is never a recorded
``ConversationTraceV1`` turn (``merge_conversation_branches`` in
``merge.py`` operates directly on a shared base and two independently
verified ``BranchEditV1`` edges, and a successful merge starts a *fresh*
continuation trace rather than appending to either input trace). A row
would require independently replaying to a recorded chosen_output_state_id
(the issue's own acceptance criterion), and a conflicting merge has no
successor state to replay to -- fabricating a "what the user did instead"
row the trace never recorded would violate the issue's own adversarial
control that chosen/rejected rows must share exact, evidenced context. The
issue's own instruction to "mark rejected candidates as typed
illegal/conflict controls outside the ranking denominator" is honored
instead by construction: ``extract_merge_preference_row`` only ever offers
``merge:<pair>`` as a legal candidate action when
``merge_conversation_branches`` has already confirmed it succeeds, so a
conflicting merge is never mistakenly added to any ranking denominator; see
``test_merge_conflict_never_yields_a_preference_row``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from slm_training.dsl.operators.conversation import (
    ConversationOperation,
    ConversationTraceV1,
)
from slm_training.dsl.operators.contracts import (
    ApplicationProvenanceV1,
    OperatorApplicationV1,
)
from slm_training.dsl.operators.legal_set import (
    LegalOperatorActionV1,
    OperatorLegalSetV1,
    enumerate_operator_legal_set,
)
from slm_training.dsl.operators.merge import (
    BranchAuthorityResolver,
    BranchEditV1,
    BranchMergeDecisionV1,
)
from slm_training.dsl.operators.registry import OperatorLibraryV1, OperatorStateV1
from slm_training.dsl.pack import DslPack
from slm_training.harness_core.versioning import build_version_stamp

ProvenanceFactory = Callable[[OperatorStateV1], ApplicationProvenanceV1]


class ReplayPreferenceRelation(str, Enum):
    EDIT_THEN_UNDO = "edit_then_undo"
    UNDO_THEN_REDO = "undo_then_redo"
    PARTIAL_ROLLBACK = "partial_rollback"
    CHECKOUT_ANOTHER_STATE = "checkout_another_state"
    FORK_THEN_CHOOSE_ONE_BRANCH = "fork_then_choose_one_branch"
    MERGE_SUCCESS = "merge_success"
    PRONOUN_FOCUS_FOLLOWUP = "pronoun_focus_followup"


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

    Models ``undo`` (a parent exists), ``redo:<child>`` (an already-
    materialized child on the current branch), and ``checkout:<state>`` (any
    other already-materialized state in the whole trace -- ancestor,
    sibling, descendant, or cross-branch -- per
    ``checkout_conversation_state``'s actual authority, which only refuses
    checkout-to-self). ``checkout:<state>`` is listed even when its target
    coincides with the ``undo``/``redo:<child>`` destination: they are
    distinct legal tool invocations (``checkout_conversation_state`` versus
    ``undo_conversation``/``redo_conversation``) recorded as different turn
    operations, and the choice between them at a shared destination is
    itself a preference signal the issue asks for. copy availability is out
    of scope for this slice; fork itself (the act of opening a new branch)
    is likewise out of scope as a chosen/rejected candidate here -- only a
    *subsequent* checkout crossing an already-forked branch boundary is
    modeled, by ``extract_replay_preference_rows`` classifying that turn as
    ``FORK_THEN_CHOOSE_ONE_BRANCH`` rather than plain
    ``CHECKOUT_ANOTHER_STATE`` (see there).
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
    for other in trace.state_nodes:
        if other.state_id != state_id:
            actions.append(f"checkout:{other.state_id}")
    return tuple(actions)


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


def _touched_refs(application: OperatorApplicationV1) -> frozenset:
    """The opaque refs one ``AST_EDIT`` application actually bound as arguments.

    This is the module's only notion of "focus": never a transcript pronoun,
    never a semantic descriptor, just the exact ``OperatorRef`` values the
    prior turn's own verified application used -- so pronoun-focus
    classification stays grounded in the DAG, per the issue's own adversarial
    control that text history cannot reconstruct a different state than the
    DAG.
    """
    return frozenset(argument.value for argument in application.arguments)


def _action_refs(action: LegalOperatorActionV1) -> frozenset:
    return frozenset(argument.value for argument in action.arguments)


def extract_replay_preference_rows(
    trace: ConversationTraceV1,
    *,
    pack: DslPack,
    library: OperatorLibraryV1,
    provenance_for: ProvenanceFactory,
) -> OperatorEventMemoryReportV1:
    """Scan ``trace.turns`` for six replay-grounded preference patterns.

    edit-then-undo, undo-then-redo, partial-rollback (a second or later
    consecutive undo), checkout-another-state, fork-then-choose-one-branch
    (a checkout that crosses a branch boundary a prior ``FORK`` turn opened),
    and pronoun-focus-followup (a second ``AST_EDIT`` that continues
    operating on the same ref its immediate predecessor touched, over an
    equally legal same-operator action targeting a different, untouched
    ref). Each match produces one row whose chosen and rejected actions are
    both verified members of the exact legal set at the shared input state.
    A row is only emitted when an unchosen alternative actually exists in
    that legal set -- undo/redo/checkout/continued-focus is never asserted
    preferred by default.
    """
    rows: list[OperatorReplayPreferenceRowV1] = []
    turns = trace.turns
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

        if (
            current.operation is ConversationOperation.UNDO
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
                        semantic_relation=ReplayPreferenceRelation.PARTIAL_ROLLBACK,
                        correction_reason="user_continued_rollback_past_first_undo",
                        legal_set_fingerprint=legal_set.fingerprint,
                    )
                )

        if (
            current.operation is ConversationOperation.AST_EDIT
            and following.operation is ConversationOperation.AST_EDIT
            and following.input_state_id == current.output_state_id
        ):
            assert current.application is not None
            assert following.application is not None
            focus_refs = _touched_refs(current.application)
            chosen_refs = _touched_refs(following.application)
            # No focus was ever established (e.g. a zero-argument operator),
            # or the follow-up shares nothing with it: not this pattern.
            if focus_refs and (focus_refs & chosen_refs):
                decision_state_id = following.input_state_id
                legal_set = _legal_set_at(
                    trace,
                    pack=pack,
                    library=library,
                    state_id=decision_state_id,
                    provenance_for=provenance_for,
                )
                entry = next(
                    (
                        candidate
                        for candidate in legal_set.entries
                        if candidate.operator_fingerprint
                        == following.application.operator_fingerprint
                    ),
                    None,
                )
                chosen_match = (
                    next(
                        (
                            action
                            for action in entry.legal_actions
                            if action.arguments == following.application.arguments
                        ),
                        None,
                    )
                    if entry is not None
                    else None
                )
                if entry is not None and chosen_match is not None:
                    sibling_candidates = sorted(
                        (
                            action
                            for action in entry.legal_actions
                            if action.serialized != chosen_match.serialized
                            and not (_action_refs(action) & focus_refs)
                        ),
                        key=lambda action: action.serialized,
                    )
                    if sibling_candidates:
                        rows.append(
                            OperatorReplayPreferenceRowV1(
                                input_state_id=decision_state_id,
                                chosen_action=chosen_match.serialized,
                                rejected_action=sibling_candidates[0].serialized,
                                chosen_output_state_id=following.output_state_id,
                                semantic_relation=(
                                    ReplayPreferenceRelation.PRONOUN_FOCUS_FOLLOWUP
                                ),
                                correction_reason=(
                                    "user_continued_implicit_focus_over_sibling_target"
                                ),
                                legal_set_fingerprint=legal_set.fingerprint,
                            )
                        )

    fork_branch_digests = {
        trace.node(turn.output_state_id).branch_digest
        for turn in turns
        if turn.operation is ConversationOperation.FORK
    }

    for turn in turns:
        if turn.operation is not ConversationOperation.CHECKOUT_STATE:
            continue
        legal_set = _legal_set_at(
            trace,
            pack=pack,
            library=library,
            state_id=turn.input_state_id,
            provenance_for=provenance_for,
        )
        chosen = f"checkout:{turn.output_state_id}"
        rejected = _pick_rejected(legal_set, chosen)
        if chosen not in legal_set.all_serialized_actions or rejected is None:
            continue

        # A checkout that lands on -- or leaves from -- a branch a recorded
        # FORK turn opened is choosing between the two diverged branches,
        # not merely relocating within one; classify it distinctly from a
        # same-branch checkout-another-state. Branch digests are opaque
        # (never display names), so this never leaks the target beyond what
        # the trace's own recorded FORK turns already establish.
        input_branch = trace.node(turn.input_state_id).branch_digest
        output_branch = trace.node(turn.output_state_id).branch_digest
        crosses_forked_branch = output_branch != input_branch and (
            input_branch in fork_branch_digests
            or output_branch in fork_branch_digests
        )
        if crosses_forked_branch:
            relation = ReplayPreferenceRelation.FORK_THEN_CHOOSE_ONE_BRANCH
            correction_reason = "user_chose_a_different_branch_after_fork"
        else:
            relation = ReplayPreferenceRelation.CHECKOUT_ANOTHER_STATE
            correction_reason = "user_checked_out_alternate_state"

        rows.append(
            OperatorReplayPreferenceRowV1(
                input_state_id=turn.input_state_id,
                chosen_action=chosen,
                rejected_action=rejected,
                chosen_output_state_id=turn.output_state_id,
                semantic_relation=relation,
                correction_reason=correction_reason,
                legal_set_fingerprint=legal_set.fingerprint,
            )
        )

    return OperatorEventMemoryReportV1(
        rows=tuple(rows),
        version_stamp=build_version_stamp("dsl.operators.replay_preference"),
    )


def _merge_candidate_action(left: BranchEditV1, right: BranchEditV1) -> str:
    """A canonical, order-independent action name for merging two branch tips.

    Sorted so the same pair of tips always serializes identically regardless
    of which edge is passed as ``left`` versus ``right`` -- matching
    ``merge_conversation_branches``'s own order-invariant ``decision_id``.
    """
    tips = tuple(sorted((left.output_node.state_id, right.output_node.state_id)))
    return f"merge:{tips[0]}:{tips[1]}"


def extract_merge_preference_row(
    *,
    left: BranchEditV1,
    right: BranchEditV1,
    decision: BranchMergeDecisionV1,
    authority_resolver: BranchAuthorityResolver,
    provenance_for: ProvenanceFactory,
) -> OperatorReplayPreferenceRowV1 | None:
    """One ``MERGE_SUCCESS`` row for a merge attempt between two fork tips.

    Grounded from ``left``'s exact output state: at that state, the legal
    set is enumerated with ``merge:<sorted-tip-pair>`` and ``checkout:<right
    tip>`` (and ``undo``, when a parent exists) offered alongside the
    ordinary operator actions -- the same
    ``ordinary_nonoperator_actions`` mechanism ``extract_replay_preference_
    rows`` uses for ``undo``/``redo``/``checkout``. ``merge:<pair>`` is only
    ever offered when ``merge_conversation_branches`` has already confirmed
    ``decision.succeeded`` and produced a fresh continuation node; a
    conflicting merge returns ``None`` here rather than a row (see the
    module docstring for why merge conflict is not modeled as a row).

    ``authority_resolver`` is called with ``left.input_node`` -- the same
    node ``merge_conversation_branches`` itself resolves the left edge's
    authority from -- since a ``BranchEditV1`` is one verified single-
    application edge and its input-state authority governs the actions
    legal at its output tip too.

    Returns ``None`` when the merge conflicted, or when no unchosen legal
    alternative exists at the decision state (never asserting merge
    preferred by default, matching every other pattern in this module).
    """
    if not decision.succeeded or decision.continuation is None:
        return None
    decision_tip = left.output_node
    other_tip = right.output_node
    pack, library = authority_resolver(left.input_node)
    merge_action = _merge_candidate_action(left, right)
    ordinary_actions = [f"checkout:{other_tip.state_id}", merge_action]
    if decision_tip.parent_state_id is not None:
        ordinary_actions.append("undo")
    legal_set = enumerate_operator_legal_set(
        pack=pack,
        library=library,
        state=decision_tip.state,
        reference_table=decision_tip.reference_table,
        provenance=provenance_for(decision_tip.state),
        ordinary_nonoperator_actions=tuple(ordinary_actions),
    )
    rejected = _pick_rejected(legal_set, merge_action)
    if merge_action not in legal_set.all_serialized_actions or rejected is None:
        return None
    return OperatorReplayPreferenceRowV1(
        input_state_id=decision_tip.state_id,
        chosen_action=merge_action,
        rejected_action=rejected,
        chosen_output_state_id=decision.continuation.merged_node.state_id,
        semantic_relation=ReplayPreferenceRelation.MERGE_SUCCESS,
        correction_reason="user_merged_diverged_branches",
        legal_set_fingerprint=legal_set.fingerprint,
    )
