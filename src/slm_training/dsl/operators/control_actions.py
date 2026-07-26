"""Conversation control-plane action set, separate from ``AstOperatorV1`` (DSH5-08).

STOP, UNDO, REDO, CHECKOUT, FORK, and MERGE_BRANCHES navigate or
merge conversation history; they never mutate an AST. Conflating them with
``AstOperatorV1`` would let a model-generated ID cross into a namespace it
doesn't belong to (a control command could never be "the same kind of thing"
as an AST edit) and would force AST-operator legal-set/effect machinery to
reason about branch topology it has nothing to do with. This module keeps
the two candidate spaces disjoint end to end:

* ``ConversationControlKind`` is a standalone enum -- never an
  ``operator_id`` string, never registered in any ``OperatorLibraryV1``.
* :func:`enumerate_conversation_control_legal_set` builds membership
  entirely from the current ``ConversationTraceV1`` shape (parent/branch
  graph) plus, for ``MERGE_BRANCHES`` only, caller-proposed candidate
  triples resolved through the existing merge machinery (``merge.py``,
  ``sequence_merge.py``) -- never from a model's own output.
* :func:`apply_conversation_control_action` executes every legal action
  through the existing, already-replay-proven conversation/merge functions
  (``undo_conversation``, ``redo_conversation``, ``checkout_conversation_state``,
  ``fork_conversation``, ``merge_conversation_branches``,
  ``merge_branch_application_sequences``) -- this module adds no new
  mutation logic, only a typed dispatch layer and a receipt.

Unlike ``OperatorLegalSetV1``, this legal set never truncates or reports
``PARTIAL`` coverage: every candidate space here is already exactly
enumerable in bounded time from the trace alone (direct children of the
current node for REDO, every other trace node for CHECKOUT, a single fact
for UNDO/FORK), so there is no bounded-scan budget to exhaust. The one
place ``UNKNOWN`` legitimately appears is ``MERGE_BRANCHES``: a merge
candidate cannot be proven legal without an ``authority_resolver`` (the
same externally-supplied, non-model dependency ``merge_conversation_branches``
already requires), so an enumeration without one records ``UNKNOWN`` rather
than guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from slm_training.dsl.operators.contracts import (
    ApplicationProvenanceV1,
    _fingerprint,
    _require_digest,
)
from slm_training.dsl.operators.conversation import (
    ConversationOperation,
    ConversationStateNodeV1,
    ConversationTraceError,
    ConversationTraceV1,
    checkout_conversation_state,
    fork_conversation,
    redo_conversation,
    undo_conversation,
)
from slm_training.dsl.operators.merge import (
    BranchAuthorityResolver,
    BranchEditV1,
    BranchMergeDecisionV1,
    MergeReferenceTableBuilder,
    merge_conversation_branches,
)
from slm_training.dsl.operators.sequence_merge import (
    BranchSequenceMergeDecisionV1,
    merge_branch_application_sequences,
    sequence_from_branch_edits,
)
from slm_training.flow.termination import (
    brier_score,
    expected_calibration_error,
)

__all__ = [
    "ConversationControlKind",
    "ControlSupportVerdict",
    "MergeCandidateV1",
    "LegalControlActionV1",
    "ConversationControlEntryV1",
    "ConversationControlLegalSetV1",
    "ConversationControlResultV1",
    "ConversationControlDecision",
    "ConversationControlPolicyReportV1",
    "enumerate_conversation_control_legal_set",
    "apply_conversation_control_action",
    "deterministic_control_priority",
]


class ConversationControlKind(str, Enum):
    STOP = "stop"
    UNDO = "undo"
    REDO = "redo"
    CHECKOUT = "checkout"
    FORK = "fork"
    MERGE_BRANCHES = "merge_branches"


class ControlSupportVerdict(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


_TARGETED_KINDS = (ConversationControlKind.REDO, ConversationControlKind.CHECKOUT)
_BARE_KINDS = (
    ConversationControlKind.STOP,
    ConversationControlKind.UNDO,
    ConversationControlKind.FORK,
)


@dataclass(frozen=True)
class MergeCandidateV1:
    """One caller-proposed ``MERGE_BRANCHES`` triple to check for legality.

    Deliberately three state IDs, not raw ASTs or a model-chosen pair:
    membership in the trace is the only thing this type asserts. Whether
    the triple is *actually* mergeable is determined later, by replaying
    the segment through the real merge functions.
    """

    base_state_id: str
    left_state_id: str
    right_state_id: str
    schema: str = "merge_candidate/v1"

    def __post_init__(self) -> None:
        _require_digest(self.base_state_id, "base_state_id")
        _require_digest(self.left_state_id, "left_state_id")
        _require_digest(self.right_state_id, "right_state_id")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "base_state_id": self.base_state_id,
            "left_state_id": self.left_state_id,
            "right_state_id": self.right_state_id,
        }


@dataclass(frozen=True)
class LegalControlActionV1:
    """One proof-checked, legal control-plane candidate."""

    kind: ConversationControlKind
    target_state_id: str | None = None
    merge_candidate: MergeCandidateV1 | None = None
    proof_checks: tuple[str, ...] = ()
    serialized: str = ""
    schema: str = "legal_control_action/v1"

    def __post_init__(self) -> None:
        if self.kind in _TARGETED_KINDS and self.target_state_id is None:
            raise ValueError(f"{self.kind.value} requires a target_state_id")
        if self.kind is ConversationControlKind.MERGE_BRANCHES and self.merge_candidate is None:
            raise ValueError("merge_branches requires a merge_candidate")
        if self.kind in _BARE_KINDS and (
            self.target_state_id is not None or self.merge_candidate is not None
        ):
            raise ValueError(f"{self.kind.value} carries no target or merge candidate")
        if self.kind not in _TARGETED_KINDS and self.target_state_id is not None:
            raise ValueError(f"{self.kind.value} does not carry a target_state_id")
        if self.kind is not ConversationControlKind.MERGE_BRANCHES and self.merge_candidate is not None:
            raise ValueError(f"{self.kind.value} does not carry a merge_candidate")

    @property
    def action_id(self) -> str:
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind.value,
            "target_state_id": self.target_state_id,
            "merge_candidate": (
                self.merge_candidate.to_dict() if self.merge_candidate is not None else None
            ),
            "proof_checks": list(self.proof_checks),
            "serialized": self.serialized,
        }


@dataclass(frozen=True)
class ConversationControlEntryV1:
    """Per-kind rollup: every legal instance of one control-action kind."""

    kind: ConversationControlKind
    legal_actions: tuple[LegalControlActionV1, ...]
    verdict: ControlSupportVerdict
    rejection_reasons: tuple[str, ...] = ()
    schema: str = "conversation_control_entry/v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind.value,
            "legal_actions": [action.to_dict() for action in self.legal_actions],
            "verdict": self.verdict.value,
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True)
class ConversationControlLegalSetV1:
    """The whole control-plane candidate space at one trace/cursor position."""

    trace_fingerprint: str
    current_state_id: str
    entries: tuple[ConversationControlEntryV1, ...]
    schema: str = "conversation_control_legal_set/v1"

    @property
    def all_legal_actions(self) -> tuple[LegalControlActionV1, ...]:
        return tuple(
            action for entry in self.entries for action in entry.legal_actions
        )

    @property
    def legal_kinds(self) -> tuple[ConversationControlKind, ...]:
        return tuple(
            entry.kind
            for entry in self.entries
            if entry.verdict is ControlSupportVerdict.SUPPORTED
        )

    @property
    def forced_action(self) -> LegalControlActionV1 | None:
        """The COMPLETE singleton-bypass candidate, if exactly one exists."""
        actions = self.all_legal_actions
        return actions[0] if len(actions) == 1 else None

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "trace_fingerprint": self.trace_fingerprint,
            "current_state_id": self.current_state_id,
            "entries": [entry.to_dict() for entry in self.entries],
        }


def _walk_branch_segment(
    trace: ConversationTraceV1,
    *,
    base_node: ConversationStateNodeV1,
    tip_id: str,
) -> tuple[BranchEditV1, ...] | None:
    """Collapse the ``AST_EDIT`` turns from ``base_node`` to ``tip_id`` on one branch.

    Mirrors ``sequence_merge.py``'s own scoping: only a chain of ``AST_EDIT``
    turns is representable. Exactly one ``FORK`` turn establishing the
    branch (whose output state equals ``base_node.state``) is transparently
    skipped, matching how a real branch begins. Any other turn kind
    (``TRANSACTION_COMMIT``, ``COPY_STATE``, or a history-navigation turn)
    encountered along the path makes the segment unsupported -- ``None``.
    """
    node_by_id = {node.state_id: node for node in trace.state_nodes}
    turn_by_output: dict[str, Any] = {}
    for turn in trace.turns:
        turn_by_output.setdefault(turn.output_state_id, turn)

    chain: list[BranchEditV1] = []
    current_id = tip_id
    while current_id != base_node.state_id:
        node = node_by_id.get(current_id)
        turn = turn_by_output.get(current_id)
        if node is None or turn is None:
            return None
        if turn.operation is ConversationOperation.AST_EDIT:
            input_node = node_by_id.get(turn.input_state_id)
            if input_node is None or turn.application is None:
                return None
            chain.append(BranchEditV1(input_node, node, turn.application))
        elif turn.operation is ConversationOperation.FORK:
            if node.state != base_node.state:
                return None
        else:
            return None
        current_id = turn.input_state_id
    chain.reverse()
    return tuple(chain)


def _check_merge_candidate(
    trace: ConversationTraceV1,
    candidate: MergeCandidateV1,
    *,
    authority_resolver: BranchAuthorityResolver | None,
    reference_table_builder: MergeReferenceTableBuilder | None,
) -> tuple[bool, str]:
    try:
        base_node = trace.node(candidate.base_state_id)
        left_tip = trace.node(candidate.left_state_id)
        right_tip = trace.node(candidate.right_state_id)
    except ConversationTraceError:
        return False, "unknown_state_id"
    if left_tip.branch_digest == right_tip.branch_digest:
        return False, "same_branch"
    if authority_resolver is None:
        return False, "no_authority_resolver"
    left_edits = _walk_branch_segment(trace, base_node=base_node, tip_id=left_tip.state_id)
    right_edits = _walk_branch_segment(trace, base_node=base_node, tip_id=right_tip.state_id)
    if not left_edits or not right_edits:
        return False, "unsupported_segment"
    decision = _merge_segments(
        trace,
        base_node=base_node,
        left_edits=left_edits,
        right_edits=right_edits,
        authority_resolver=authority_resolver,
        reference_table_builder=reference_table_builder,
    )
    if decision.succeeded:
        return True, ""
    assert decision.conflict is not None
    return False, f"merge_conflict:{decision.conflict.kind.value}"


def _merge_segments(
    trace: ConversationTraceV1,
    *,
    base_node: ConversationStateNodeV1,
    left_edits: tuple[BranchEditV1, ...],
    right_edits: tuple[BranchEditV1, ...],
    authority_resolver: BranchAuthorityResolver,
    reference_table_builder: MergeReferenceTableBuilder | None,
) -> BranchMergeDecisionV1 | BranchSequenceMergeDecisionV1:
    base_pack, _base_library = authority_resolver(base_node)
    if len(left_edits) == 1 and len(right_edits) == 1:
        return merge_conversation_branches(
            pack=base_pack,
            base=base_node,
            left=left_edits[0],
            right=right_edits[0],
            authority_resolver=authority_resolver,
            reference_table_builder=reference_table_builder,
        )
    return merge_branch_application_sequences(
        pack=base_pack,
        base=base_node,
        left=sequence_from_branch_edits(left_edits),
        right=sequence_from_branch_edits(right_edits),
        authority_resolver=authority_resolver,
        reference_table_builder=reference_table_builder,
    )


def enumerate_conversation_control_legal_set(
    *,
    trace: ConversationTraceV1,
    merge_candidates: tuple[MergeCandidateV1, ...] = (),
    authority_resolver: BranchAuthorityResolver | None = None,
    reference_table_builder: MergeReferenceTableBuilder | None = None,
) -> ConversationControlLegalSetV1:
    """Build the whole control-plane candidate space from the trace alone.

    ``merge_candidates`` are the only externally-supplied inputs -- three
    state IDs per candidate, resolved and replayed through the real merge
    functions, never a model-generated AST or pair choice.
    """
    current = trace.current
    entries: list[ConversationControlEntryV1] = [
        ConversationControlEntryV1(
            kind=ConversationControlKind.STOP,
            legal_actions=(
                LegalControlActionV1(
                    kind=ConversationControlKind.STOP,
                    proof_checks=("always_legal",),
                    serialized="CONTROL STOP",
                ),
            ),
            verdict=ControlSupportVerdict.SUPPORTED,
        )
    ]

    if current.parent_state_id is not None:
        entries.append(
            ConversationControlEntryV1(
                kind=ConversationControlKind.UNDO,
                legal_actions=(
                    LegalControlActionV1(
                        kind=ConversationControlKind.UNDO,
                        proof_checks=("has_parent",),
                        serialized="CONTROL UNDO",
                    ),
                ),
                verdict=ControlSupportVerdict.SUPPORTED,
            )
        )
    else:
        entries.append(
            ConversationControlEntryV1(
                kind=ConversationControlKind.UNDO,
                legal_actions=(),
                verdict=ControlSupportVerdict.UNSUPPORTED,
                rejection_reasons=("nothing_to_undo",),
            )
        )

    redo_children = tuple(
        node for node in trace.state_nodes if node.parent_state_id == current.state_id
    )
    if redo_children:
        entries.append(
            ConversationControlEntryV1(
                kind=ConversationControlKind.REDO,
                legal_actions=tuple(
                    LegalControlActionV1(
                        kind=ConversationControlKind.REDO,
                        target_state_id=child.state_id,
                        proof_checks=("is_direct_child",),
                        serialized=f"CONTROL REDO target={child.state_id}",
                    )
                    for child in sorted(redo_children, key=lambda node: node.state_id)
                ),
                verdict=ControlSupportVerdict.SUPPORTED,
            )
        )
    else:
        entries.append(
            ConversationControlEntryV1(
                kind=ConversationControlKind.REDO,
                legal_actions=(),
                verdict=ControlSupportVerdict.UNSUPPORTED,
                rejection_reasons=("no_children",),
            )
        )

    checkout_targets = tuple(
        node for node in trace.state_nodes if node.state_id != current.state_id
    )
    if checkout_targets:
        entries.append(
            ConversationControlEntryV1(
                kind=ConversationControlKind.CHECKOUT,
                legal_actions=tuple(
                    LegalControlActionV1(
                        kind=ConversationControlKind.CHECKOUT,
                        target_state_id=node.state_id,
                        proof_checks=("target_exists_and_differs",),
                        serialized=f"CONTROL CHECKOUT target={node.state_id}",
                    )
                    for node in sorted(checkout_targets, key=lambda node: node.state_id)
                ),
                verdict=ControlSupportVerdict.SUPPORTED,
            )
        )
    else:
        entries.append(
            ConversationControlEntryV1(
                kind=ConversationControlKind.CHECKOUT,
                legal_actions=(),
                verdict=ControlSupportVerdict.UNSUPPORTED,
                rejection_reasons=("only_current_state_exists",),
            )
        )

    entries.append(
        ConversationControlEntryV1(
            kind=ConversationControlKind.FORK,
            legal_actions=(
                LegalControlActionV1(
                    kind=ConversationControlKind.FORK,
                    proof_checks=("always_legal_pending_nonce",),
                    serialized="CONTROL FORK",
                ),
            ),
            verdict=ControlSupportVerdict.SUPPORTED,
        )
    )

    if not merge_candidates:
        entries.append(
            ConversationControlEntryV1(
                kind=ConversationControlKind.MERGE_BRANCHES,
                legal_actions=(),
                verdict=ControlSupportVerdict.UNKNOWN,
                rejection_reasons=("no_candidates_proposed",),
            )
        )
    else:
        merge_actions: list[LegalControlActionV1] = []
        merge_rejections: list[str] = []
        for candidate in merge_candidates:
            legal, reason = _check_merge_candidate(
                trace,
                candidate,
                authority_resolver=authority_resolver,
                reference_table_builder=reference_table_builder,
            )
            if legal:
                merge_actions.append(
                    LegalControlActionV1(
                        kind=ConversationControlKind.MERGE_BRANCHES,
                        merge_candidate=candidate,
                        proof_checks=("dry_run_merge_succeeded",),
                        serialized=(
                            "CONTROL MERGE_BRANCHES "
                            f"base={candidate.base_state_id} "
                            f"left={candidate.left_state_id} "
                            f"right={candidate.right_state_id}"
                        ),
                    )
                )
            else:
                merge_rejections.append(reason)
        entries.append(
            ConversationControlEntryV1(
                kind=ConversationControlKind.MERGE_BRANCHES,
                legal_actions=tuple(merge_actions),
                verdict=(
                    ControlSupportVerdict.SUPPORTED
                    if merge_actions
                    else (
                        ControlSupportVerdict.UNKNOWN
                        if authority_resolver is None
                        else ControlSupportVerdict.UNSUPPORTED
                    )
                ),
                rejection_reasons=tuple(merge_rejections),
            )
        )

    return ConversationControlLegalSetV1(
        trace_fingerprint=trace.fingerprint,
        current_state_id=current.state_id,
        entries=tuple(entries),
    )


@dataclass(frozen=True)
class ConversationControlResultV1:
    """Execution receipt: exact input/output identity, never a bare mutation."""

    kind: ConversationControlKind
    input_trace_fingerprint: str
    output_trace: ConversationTraceV1 | None = None
    merge_decision: BranchMergeDecisionV1 | BranchSequenceMergeDecisionV1 | None = None
    stop_reason: str | None = None
    schema: str = "conversation_control_result/v1"

    def __post_init__(self) -> None:
        if self.kind is ConversationControlKind.STOP:
            if self.stop_reason is None or self.output_trace is not None or self.merge_decision is not None:
                raise ValueError("stop result must carry only a stop_reason")
        elif self.kind is ConversationControlKind.MERGE_BRANCHES:
            if self.merge_decision is None or self.output_trace is not None or self.stop_reason is not None:
                raise ValueError("merge_branches result must carry only a merge_decision")
        else:
            if self.output_trace is None or self.merge_decision is not None or self.stop_reason is not None:
                raise ValueError(f"{self.kind.value} result must carry only an output_trace")

    @property
    def succeeded(self) -> bool:
        if self.kind is ConversationControlKind.STOP:
            return True
        if self.kind is ConversationControlKind.MERGE_BRANCHES:
            assert self.merge_decision is not None
            return self.merge_decision.succeeded
        return self.output_trace is not None


def apply_conversation_control_action(
    trace: ConversationTraceV1,
    action: LegalControlActionV1,
    *,
    provenance: ApplicationProvenanceV1,
    authority_resolver: BranchAuthorityResolver | None = None,
    reference_table_builder: MergeReferenceTableBuilder | None = None,
    branch_nonce_digest: str | None = None,
    reference_seed: int | None = None,
    stop_reason: str = "policy_decided",
) -> ConversationControlResultV1:
    """Execute one legal control action through the existing, proven functions.

    This function adds no new mutation logic: every branch below is a thin,
    typed dispatch to a function ``conversation.py``/``merge.py``/
    ``sequence_merge.py`` already implements, replays, and tests.
    """
    fingerprint = trace.fingerprint
    if action.kind is ConversationControlKind.STOP:
        return ConversationControlResultV1(
            kind=action.kind,
            input_trace_fingerprint=fingerprint,
            stop_reason=stop_reason,
        )
    if action.kind is ConversationControlKind.UNDO:
        return ConversationControlResultV1(
            kind=action.kind,
            input_trace_fingerprint=fingerprint,
            output_trace=undo_conversation(trace, provenance=provenance),
        )
    if action.kind is ConversationControlKind.REDO:
        assert action.target_state_id is not None
        return ConversationControlResultV1(
            kind=action.kind,
            input_trace_fingerprint=fingerprint,
            output_trace=redo_conversation(
                trace, target_state_id=action.target_state_id, provenance=provenance
            ),
        )
    if action.kind is ConversationControlKind.CHECKOUT:
        assert action.target_state_id is not None
        return ConversationControlResultV1(
            kind=action.kind,
            input_trace_fingerprint=fingerprint,
            output_trace=checkout_conversation_state(
                trace, target_state_id=action.target_state_id, provenance=provenance
            ),
        )
    if action.kind is ConversationControlKind.FORK:
        if branch_nonce_digest is None or reference_seed is None:
            raise ValueError("fork requires branch_nonce_digest and reference_seed")
        return ConversationControlResultV1(
            kind=action.kind,
            input_trace_fingerprint=fingerprint,
            output_trace=fork_conversation(
                trace,
                branch_nonce_digest=branch_nonce_digest,
                reference_seed=reference_seed,
                provenance=provenance,
            ),
        )
    if action.kind is ConversationControlKind.MERGE_BRANCHES:
        assert action.merge_candidate is not None
        if authority_resolver is None:
            raise ValueError("merge_branches requires an authority_resolver")
        candidate = action.merge_candidate
        base_node = trace.node(candidate.base_state_id)
        left_tip = trace.node(candidate.left_state_id)
        right_tip = trace.node(candidate.right_state_id)
        left_edits = _walk_branch_segment(trace, base_node=base_node, tip_id=left_tip.state_id)
        right_edits = _walk_branch_segment(trace, base_node=base_node, tip_id=right_tip.state_id)
        if not left_edits or not right_edits:
            raise ValueError("merge_branches candidate is not a supported segment")
        decision = _merge_segments(
            trace,
            base_node=base_node,
            left_edits=left_edits,
            right_edits=right_edits,
            authority_resolver=authority_resolver,
            reference_table_builder=reference_table_builder,
        )
        return ConversationControlResultV1(
            kind=action.kind,
            input_trace_fingerprint=fingerprint,
            merge_decision=decision,
        )
    raise ValueError(f"unknown control action kind: {action.kind}")  # pragma: no cover


@dataclass(frozen=True)
class ConversationControlDecision:
    """One policy's choice over a ``ConversationControlLegalSetV1``."""

    action: LegalControlActionV1 | None
    decision_kind: str
    confidence: float = 0.0
    telemetry: dict[str, Any] = field(default_factory=dict)


_DETERMINISTIC_PRIORITY = (
    ConversationControlKind.MERGE_BRANCHES,
    ConversationControlKind.REDO,
    ConversationControlKind.UNDO,
    ConversationControlKind.CHECKOUT,
    ConversationControlKind.FORK,
    ConversationControlKind.STOP,
)


def deterministic_control_priority(
    legal_set: ConversationControlLegalSetV1,
) -> ConversationControlDecision:
    """The API/heuristic arm: COMPLETE singleton bypass, then a fixed priority order.

    No learned scoring is used here; this is the deterministic baseline the
    issue's acceptance criteria require to "remain available" regardless of
    whether a learned arm is adopted. Priority favors the most information-
    committing action (``MERGE_BRANCHES``, ``REDO``) over pure exploration
    (``FORK``) or the no-op (``STOP``), but this ordering is a documented,
    inspectable default, not a claim about optimal policy.
    """
    actions = legal_set.all_legal_actions
    if not actions:
        return ConversationControlDecision(
            action=None,
            decision_kind="abstain",
            confidence=1.0,
            telemetry={"reason": "empty_legal_set"},
        )
    if len(actions) == 1:
        return ConversationControlDecision(
            action=actions[0],
            decision_kind="forced",
            confidence=1.0,
            telemetry={"reason": "single_legal_action"},
        )
    by_kind: dict[ConversationControlKind, LegalControlActionV1] = {}
    for candidate in actions:
        by_kind.setdefault(candidate.kind, candidate)
    for kind in _DETERMINISTIC_PRIORITY:
        if kind in by_kind:
            return ConversationControlDecision(
                action=by_kind[kind],
                decision_kind="scored",
                confidence=1.0,
                telemetry={"policy": "deterministic_priority", "kind": kind.value},
            )
    raise AssertionError("unreachable: every legal action has a known kind")  # pragma: no cover


@dataclass(frozen=True)
class ConversationControlPolicyReportV1:
    """Selection metrics for one policy arm; no training loss is reported here.

    Mirrors ``models.operator_termination.OperatorTerminationReportV1``'s
    shape and reuses its calibration helpers -- STOP is scored exactly like
    an operator-policy STOP decision, kept in a distinct report type because
    its legal set and candidates are the control-plane's, not the AST
    operator policy's.
    """

    command_accuracy: float
    argument_accuracy: float
    stop_brier: float
    stop_ece: float
    premature_stop_rate: float
    late_stop_rate: float
    schema: str = "conversation_control_policy_report/v1"

    @classmethod
    def from_predictions(
        cls,
        *,
        predicted_actions: Sequence[LegalControlActionV1 | None],
        expected_actions: Sequence[LegalControlActionV1 | None],
        stop_probabilities: Sequence[float],
        stop_targets: Sequence[int],
        premature: Sequence[bool],
        late: Sequence[bool],
    ) -> "ConversationControlPolicyReportV1":
        n = max(1, len(expected_actions))
        command_matches = sum(
            1
            for predicted, expected in zip(predicted_actions, expected_actions)
            if predicted is not None and expected is not None and predicted.kind == expected.kind
        )
        argument_matches = sum(
            1
            for predicted, expected in zip(predicted_actions, expected_actions)
            if predicted is not None
            and expected is not None
            and predicted.kind == expected.kind
            and predicted.target_state_id == expected.target_state_id
            and predicted.merge_candidate == expected.merge_candidate
        )
        return cls(
            command_accuracy=command_matches / n,
            argument_accuracy=argument_matches / n,
            stop_brier=brier_score(stop_probabilities, stop_targets),
            stop_ece=expected_calibration_error(stop_probabilities, stop_targets),
            premature_stop_rate=sum(premature) / max(1, len(premature)),
            late_stop_rate=sum(late) / max(1, len(late)),
        )
