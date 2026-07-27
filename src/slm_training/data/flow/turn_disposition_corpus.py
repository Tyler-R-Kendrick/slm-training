"""Real-corpus turn-disposition rows and leakage-safe split -- VAR3-03 (SLM-433).

Extends the DSH3-24 operator-policy corpus pipeline
(``slm_training.data.flow.operator_policy_corpus``) with a parallel row
family that labels each accepted collapsed step with a gold
``TurnDispositionTargetV1`` (``slm_training.models.turn_disposition_head``),
derived through the exact same ``derive_turn_disposition``
(``slm_training.dsl.operators.turn_disposition``) the REPL variant's own
decode loop consults at inference time -- never a hand-authored or
synthetically perturbed label, per VAR3-02's own explicit next-step note.

``out_of_scope`` structurally cannot appear in this corpus: every row here
starts from a step whose recorded operator application was proven live
against a freshly re-enumerated legal set (mirroring
``operator_policy_corpus.build_operator_policy_rows``'s own
``ACCEPTED_ACTION_NOT_LIVE`` rejection), so that legal set can never be
empty. ``TurnDispositionTargetV1`` itself has no ``out_of_scope`` member, so
there is no slot in the label space to mislabel one into even by
construction -- :func:`build_turn_disposition_rows` asserts this invariant
rather than silently trusting it.

``clarify`` margin labels come from a real, corpus-wide statistic --
each live candidate action's frequency as the historically *accepted*
choice elsewhere in the same split (:func:`corpus_action_frequency`) --
never a synthetic score manufactured per example to force an outcome.
``is_state_query`` is always ``False``: this corpus generator only emits
AST-mutating operator applications, so ``answer`` never appears in its
targets -- an explicit, honestly-stated scope limit, not a fabricated claim.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from slm_training import levers
from slm_training.dsl.operators.collapse import CollapsedInstructionV1
from slm_training.dsl.operators.contracts import _fingerprint, _require_digest
from slm_training.dsl.operators.conversation import (
    ConversationTraceV1,
    OperatorAuthorityResolver,
)
from slm_training.dsl.operators.legal_set import enumerate_operator_legal_set
from slm_training.dsl.operators.turn_disposition import (
    TurnDispositionKind,
    derive_turn_disposition,
)
from slm_training.models.operator_policy_view import (
    build_operator_policy_input,
    validate_no_forbidden_fields,
)
from slm_training.models.turn_disposition_head import (
    TurnDispositionLabel,
    TurnDispositionTargetV1,
)

__all__ = [
    "TurnDispositionRowV1",
    "assert_leakage_safe_split",
    "build_turn_disposition_rows",
    "corpus_action_frequency",
]

_KIND_TO_LABEL: dict[TurnDispositionKind, TurnDispositionLabel] = {
    TurnDispositionKind.EMIT: TurnDispositionLabel.EMIT,
    TurnDispositionKind.CLARIFY: TurnDispositionLabel.CLARIFY,
    TurnDispositionKind.ANSWER: TurnDispositionLabel.ANSWER,
}


@dataclass(frozen=True)
class TurnDispositionRowV1:
    """One real-corpus turn-disposition supervision + evaluation row.

    ``trace_id`` is the leakage-safety unit (the whole collapsed
    conversation, matching ``OperatorPolicyRowV1.collapse_id`` /
    ``freeze_collapse_negative_ablation``'s own split-disjointness key) --
    never the individual row.
    """

    row_id: str
    trace_id: str
    turn_id: str
    step_index: int
    state_digest: str
    legal_set_fingerprint: str
    policy_input: dict[str, Any]
    policy_input_digest: str
    target: TurnDispositionTargetV1
    accepted_action_serialized: str
    chosen_action_serialized: str | None
    top_candidate_serialized: str
    margin: float | None
    candidate_count: int
    split: str
    schema: str = "turn_disposition_row/v1"

    def __post_init__(self) -> None:
        _require_digest(self.trace_id, "trace_id")
        _require_digest(self.turn_id, "turn_id")
        _require_digest(self.state_digest, "state_digest")
        _require_digest(self.legal_set_fingerprint, "legal_set_fingerprint")
        _require_digest(self.policy_input_digest, "policy_input_digest")
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if self.split not in {"train", "dev"}:
            raise ValueError("unsupported split")
        if self.candidate_count < 1:
            raise ValueError("a labeled row must have at least one live candidate")
        if self.policy_input_digest != _fingerprint(self.policy_input):
            raise ValueError("policy_input_digest does not match policy_input content")
        validate_no_forbidden_fields(self.policy_input)
        if self.target.label is TurnDispositionLabel.EMIT:
            if self.chosen_action_serialized is None:
                raise ValueError("emit rows require a chosen action")
        elif self.chosen_action_serialized is not None:
            raise ValueError("non-emit rows carry no chosen action")
        if not self.top_candidate_serialized:
            raise ValueError("top_candidate_serialized must be non-empty")

    @property
    def emit_correct(self) -> bool | None:
        """Whether an ``emit`` row's chosen candidate matches the historically
        accepted one; ``None`` for rows the gold disposition never emits."""
        if self.target.label is not TurnDispositionLabel.EMIT:
            return None
        return self.chosen_action_serialized == self.accepted_action_serialized

    @property
    def top_candidate_correct(self) -> bool:
        """Whether an always-emit policy's top-ranked candidate on this row
        (regardless of gold disposition) matches the historically accepted
        action -- the "would this naive/always-emit baseline be right"
        fact every row carries, unlike :attr:`emit_correct` which is only
        defined for rows the gold disposition itself emits."""
        return self.top_candidate_serialized == self.accepted_action_serialized

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "row_id": self.row_id,
            "trace_id": self.trace_id,
            "turn_id": self.turn_id,
            "step_index": self.step_index,
            "state_digest": self.state_digest,
            "legal_set_fingerprint": self.legal_set_fingerprint,
            "policy_input_digest": self.policy_input_digest,
            "policy_input": self.policy_input,
            "target": {
                "label": self.target.label.value,
                "forced_singleton": self.target.forced_singleton,
            },
            "accepted_action_serialized": self.accepted_action_serialized,
            "chosen_action_serialized": self.chosen_action_serialized,
            "top_candidate_serialized": self.top_candidate_serialized,
            "margin": self.margin,
            "candidate_count": self.candidate_count,
            "split": self.split,
        }


def corpus_action_frequency(
    accepted_actions_by_trace: Mapping[str, Sequence[str]],
) -> Counter[str]:
    """Real, corpus-wide accepted-action frequency -- the disposition score provider.

    Computed once over an entire split's historically accepted actions
    (never per example), so a candidate's "score" is a genuine corpus
    statistic, not a synthetic perturbation manufactured to force a margin
    outcome.
    """
    counts: Counter[str] = Counter()
    for actions in accepted_actions_by_trace.values():
        counts.update(actions)
    return counts


def assert_leakage_safe_split(rows: Sequence[TurnDispositionRowV1]) -> None:
    """Fail closed if any source trace appears in more than one split.

    Mirrors ``operator_policy_corpus.freeze_collapse_negative_ablation``'s
    own split-disjointness check: turns within one collapsed conversation
    trace share state history, so the leakage-safe unit is the whole trace
    (``row.trace_id``), never the individual row.
    """
    split_by_trace: dict[str, str] = {}
    for row in rows:
        previous = split_by_trace.setdefault(row.trace_id, row.split)
        if previous != row.split:
            raise ValueError("source trace appears in more than one split")


def build_turn_disposition_rows(
    *,
    trace: ConversationTraceV1,
    collapse: CollapsedInstructionV1,
    authority_resolver: OperatorAuthorityResolver,
    action_frequency: Mapping[str, int],
    split: str = "train",
    max_combinations_per_operator: int = 64,
    clarify_margin_threshold: float = levers.TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD,
    clarify_top_k: int = levers.TURN_DISPOSITION_CLARIFY_TOP_K,
) -> tuple[tuple[TurnDispositionRowV1, ...], tuple[dict[str, Any], ...]]:
    """One disposition row per accepted collapsed step. Returns ``(rows, rejected)``.

    Re-enumerates the live legal set fresh at each step exactly like
    ``operator_policy_corpus.build_operator_policy_rows``, then derives its
    gold disposition through the same ``derive_turn_disposition`` the REPL
    variant's own decode loop consults -- never a hand-labeled shortcut.
    """
    if max_combinations_per_operator <= 0:
        raise ValueError("max_combinations_per_operator must be positive")
    if collapse.turn_ids != tuple(turn.turn_id for turn in trace.turns):
        raise ValueError("collapse turn lineage differs from the source trace")

    rows: list[TurnDispositionRowV1] = []
    rejected: list[dict[str, Any]] = []
    for step_index, (operator_id, application) in enumerate(
        zip(collapse.operator_ids, collapse.applications, strict=True)
    ):
        turn = trace.turns[step_index]
        input_node = trace.node(turn.input_state_id)
        state = input_node.state
        reference_table = input_node.reference_table
        turn_pack, library = authority_resolver(input_node)

        legal_set = enumerate_operator_legal_set(
            pack=turn_pack,
            library=library,
            state=state,
            reference_table=reference_table,
            provenance=application.provenance,
            max_combinations_per_operator=max_combinations_per_operator,
        )
        accepted = next(
            (
                action.serialized
                for entry in legal_set.entries
                if entry.operator_id == operator_id
                for action in entry.legal_actions
                if action.application_id == application.application_id
            ),
            None,
        )
        if accepted is None:
            rejected.append(
                {
                    "collapse_id": collapse.collapse_id,
                    "step_index": step_index,
                    "reason": "accepted_action_not_live",
                }
            )
            continue

        actions = legal_set.all_serialized_actions

        def score_provider(actions: tuple[str, ...] = actions) -> dict[str, float]:
            return {action: float(action_frequency.get(action, 0)) for action in actions}

        disposition = derive_turn_disposition(
            legal_set=legal_set,
            is_state_query=False,
            clarify_margin_threshold=clarify_margin_threshold,
            clarify_top_k=clarify_top_k,
            score_provider=score_provider,
        )
        if disposition.kind is TurnDispositionKind.OUT_OF_SCOPE:
            # Structurally unreachable: `accepted` above already proves
            # `actions` is non-empty, so out_of_scope can never be derived
            # here. Fail loudly rather than silently mislabel or drop.
            raise AssertionError(
                "an accepted action implies a non-empty legal set; "
                "out_of_scope is unreachable in this corpus"
            )

        label = _KIND_TO_LABEL[disposition.kind]
        forced_singleton = (
            label is TurnDispositionLabel.EMIT and legal_set.forced_action is not None
        )
        target = TurnDispositionTargetV1(label=label, forced_singleton=forced_singleton)
        policy_input_dict = build_operator_policy_input(
            reference_table, legal_set, library
        ).to_dict()
        # The candidate an always-emit policy would commit to on this row,
        # independent of the gold disposition: `derive_turn_disposition`
        # only returns `chosen_action` when it itself emits, but its
        # `clarify_candidates` are already ranked top-first by the same
        # score provider, so `clarify_candidates[0]` is that same top pick.
        top_candidate = (
            disposition.chosen_action
            if disposition.kind is TurnDispositionKind.EMIT
            else disposition.clarify_candidates[0]
        )

        identity = {
            "schema": "turn_disposition_row_identity/v1",
            "collapse_id": collapse.collapse_id,
            "step_index": step_index,
            "state_digest": state.state_digest,
            "reference_table_fingerprint": reference_table.fingerprint,
        }
        rows.append(
            TurnDispositionRowV1(
                row_id=f"tdrow_{_fingerprint(identity)[:24]}",
                trace_id=collapse.collapse_id,
                turn_id=turn.turn_id,
                step_index=step_index,
                state_digest=state.state_digest,
                legal_set_fingerprint=legal_set.fingerprint,
                policy_input=policy_input_dict,
                policy_input_digest=_fingerprint(policy_input_dict),
                target=target,
                accepted_action_serialized=accepted,
                chosen_action_serialized=disposition.chosen_action,
                top_candidate_serialized=top_candidate,
                margin=disposition.margin,
                candidate_count=len(actions),
                split=split,
            )
        )
    return tuple(rows), tuple(rejected)
