"""Turn-level disposition (emit/clarify/answer/out_of_scope) -- VAR3-02 (SLM-430).

The REPL variant's decode loop makes one classification decision per turn
before it commits anything: is this turn out of scope, does it need
clarification, is it a question about existing state, or is it a single
op/transaction to emit? This module answers that question in strict I1
precedence order (docs/design/decode-invariants.md: a deterministic/singleton
bypass outranks any learned score) and never lets a learned score override a
derived fact:

* ``out_of_scope`` -- the accepted legal set (``OperatorLegalSetV1``) is
  empty. This is read directly off ``legal_set.all_serialized_actions``;
  nothing else -- no state-query flag, no score provider -- is ever
  consulted on this path. There is no candidate for a learned score to even
  apply to, so ``derive_turn_disposition`` returns before touching its
  ``score_provider`` argument at all (proven by
  ``test_out_of_scope_never_consults_score_provider`` in
  ``tests/test_dsl/test_turn_disposition.py``, which passes a provider that
  raises if called).
* ``answer`` -- the caller-supplied ``is_state_query`` flag is a corpus/
  trace-derived fact (e.g. "this turn's gold action is a read of the current
  AST, not a mutation"), never a model prediction. It consumes the replayed
  state and emits no op -- structurally represented as a self-referential
  ``COPY_STATE`` turn (see ``append_answer_turn`` and
  ``conversation.TurnArtifactV1``'s disposition validation).
* ``emit`` (forced) -- a ``COMPLETE``-coverage legal set with exactly one
  candidate (``OperatorLegalSetV1.forced_action``) is the same singleton
  bypass every other operator-policy surface in this repo honors
  (``models/operator_termination.py``'s ``forced_singleton``,
  ``control_actions.py``'s ``deterministic_control_priority`` forced path):
  committed with no learned score consulted.
* ``clarify`` / ``emit`` (scored) -- only once none of the above apply is
  ``score_provider`` consulted, for a top-1-vs-runner-up margin against the
  registered, conservative-by-default
  ``levers.TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD``. Below the threshold
  clarifies with the top ``clarify_top_k`` ranked candidates; at or above it
  emits the top-ranked one.

A ``PARTIAL``-coverage legal set that happens to enumerate only one
candidate so far is *not* treated as forced -- the bounded scan cannot rule
out more candidates beyond its budget, so it conservatively clarifies with
that one candidate rather than risk a false singleton bypass.

Natural-language generation for ``clarify``/``answer`` turns is out of
scope: both carry only structured candidates/state, never prose.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping

from slm_training.dsl.operators.contracts import ApplicationProvenanceV1
from slm_training.dsl.operators.conversation import (
    ConversationTraceV1,
    copy_conversation_state,
)
from slm_training.dsl.operators.legal_set import OperatorLegalSetV1

__all__ = [
    "ScoreProvider",
    "TurnDispositionKind",
    "TurnDispositionV1",
    "append_answer_turn",
    "append_clarify_turn",
    "derive_turn_disposition",
]


class TurnDispositionKind(str, Enum):
    OUT_OF_SCOPE = "out_of_scope"
    CLARIFY = "clarify"
    ANSWER = "answer"
    EMIT = "emit"


#: Called at most once, only when a scored (non-forced, non-derived) decision
#: is actually required -- see the module docstring's precedence order.
ScoreProvider = Callable[[], Mapping[str, float]]


@dataclass(frozen=True)
class TurnDispositionV1:
    """One turn's disposition decision; shape is validated per ``kind``."""

    kind: TurnDispositionKind
    chosen_action: str | None = None
    clarify_candidates: tuple[str, ...] = ()
    margin: float | None = None
    schema: str = "turn_disposition/v1"

    def __post_init__(self) -> None:
        if self.kind is TurnDispositionKind.OUT_OF_SCOPE:
            if (
                self.chosen_action is not None
                or self.clarify_candidates
                or self.margin is not None
            ):
                raise ValueError(
                    "out_of_scope carries no action, candidates, or margin"
                )
        elif self.kind is TurnDispositionKind.ANSWER:
            if self.chosen_action is not None or self.clarify_candidates:
                raise ValueError("answer carries no action or candidates")
        elif self.kind is TurnDispositionKind.CLARIFY:
            if self.chosen_action is not None:
                raise ValueError("clarify carries no single chosen action")
            if not self.clarify_candidates:
                raise ValueError("clarify requires at least one ranked candidate")
            if len(set(self.clarify_candidates)) != len(self.clarify_candidates):
                raise ValueError("clarify candidates must be unique")
        else:
            assert self.kind is TurnDispositionKind.EMIT
            if self.chosen_action is None:
                raise ValueError("emit requires a chosen action")
            if self.clarify_candidates:
                raise ValueError("emit carries no clarify candidates")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": self.kind.value,
            "chosen_action": self.chosen_action,
            "clarify_candidates": list(self.clarify_candidates),
            "margin": self.margin,
        }


def derive_turn_disposition(
    *,
    legal_set: OperatorLegalSetV1,
    is_state_query: bool,
    clarify_margin_threshold: float,
    clarify_top_k: int = 3,
    score_provider: ScoreProvider | None = None,
) -> TurnDispositionV1:
    """Derive one disposition in strict I1 precedence order.

    ``score_provider`` is a thunk, not a plain mapping, specifically so a
    test can prove it is never invoked on the ``out_of_scope`` / forced-
    singleton / state-query paths: pass a callable that raises and confirm
    it never fires.
    """
    if clarify_top_k < 1:
        raise ValueError("clarify_top_k must be at least 1")
    if clarify_margin_threshold < 0:
        raise ValueError("clarify_margin_threshold must be non-negative")

    actions = legal_set.all_serialized_actions
    if not actions:
        # I1: derived from the empty accepted set alone. Nothing else --
        # not is_state_query, not score_provider -- is ever consulted.
        return TurnDispositionV1(kind=TurnDispositionKind.OUT_OF_SCOPE)

    if is_state_query:
        # A corpus/trace-derived fact, never a model prediction; still
        # outranks the scored path, but only once out_of_scope is ruled out.
        return TurnDispositionV1(kind=TurnDispositionKind.ANSWER)

    forced = legal_set.forced_action
    if forced is not None:
        # I1 singleton bypass: COMPLETE coverage, exactly one candidate.
        return TurnDispositionV1(kind=TurnDispositionKind.EMIT, chosen_action=forced)

    if len(actions) == 1:
        # PARTIAL coverage with one candidate seen so far is NOT a forced
        # singleton -- the bounded scan could not rule out more candidates.
        # Conservative default: clarify rather than risk a false bypass.
        return TurnDispositionV1(
            kind=TurnDispositionKind.CLARIFY, clarify_candidates=(actions[0],)
        )

    if score_provider is None:
        raise ValueError("turn_disposition.missing_score_provider")
    scores = score_provider()
    missing = [action for action in actions if action not in scores]
    if missing:
        raise ValueError("turn_disposition.incomplete_scores")
    ranked = sorted(actions, key=lambda action: (-scores[action], action))
    margin = scores[ranked[0]] - scores[ranked[1]]
    if margin < clarify_margin_threshold:
        top_k = tuple(ranked[: min(clarify_top_k, len(ranked))])
        return TurnDispositionV1(
            kind=TurnDispositionKind.CLARIFY,
            clarify_candidates=top_k,
            margin=margin,
        )
    return TurnDispositionV1(
        kind=TurnDispositionKind.EMIT, chosen_action=ranked[0], margin=margin
    )


def append_clarify_turn(
    trace: ConversationTraceV1,
    *,
    candidates: tuple[str, ...],
    provenance: ApplicationProvenanceV1,
) -> ConversationTraceV1:
    """Record a CLARIFY turn: state-preserving, ranked candidates carried structurally.

    A self-referential ``COPY_STATE`` turn (source == current state) is the
    existing, already replay-proven representation of "one turn later, same
    AST content" (I11: "a turn may re-materialize an earlier state's
    artifact on the current branch"). No new ``ConversationOperation`` or
    replay logic is added; ``disposition``/``disposition_candidates`` are
    plain descriptive fields on ``TurnArtifactV1`` that round-trip through
    ``replay_conversation_trace`` untouched, exactly like every other turn
    field.
    """
    return copy_conversation_state(
        trace,
        source_state_id=trace.current_state_id,
        provenance=provenance,
        disposition=TurnDispositionKind.CLARIFY.value,
        disposition_candidates=candidates,
    )


def append_answer_turn(
    trace: ConversationTraceV1,
    *,
    provenance: ApplicationProvenanceV1,
) -> ConversationTraceV1:
    """Record an ANSWER turn: consumes replayed state, emits no op."""
    return copy_conversation_state(
        trace,
        source_state_id=trace.current_state_id,
        provenance=provenance,
        disposition=TurnDispositionKind.ANSWER.value,
    )
