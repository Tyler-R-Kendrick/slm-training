"""SLM-418 (DSH5-10) ninth slice: a sibling history-control action view.

The eighth slice (``replay_preference_typed_policy_adapter.py``) found that
``OperatorPolicyInputV1`` (``slm_training.models.operator_policy_view``,
DSH3-22-owned) has no row, slot, or candidate for a control action at all --
``undo``/``redo:<state>``/``checkout:<state>``/``merge:<pair>`` tokens are
collapsed into a bare ``ordinary_action_count`` scalar. It named two
follow-on levers: (A) extend ``OperatorPolicyInputV1``/
``OperatorFeatureEncoder`` with a "recently touched reference" signal, or
(B) invent a control-action representation so the other six relations
become representable.

Both levers, read literally, mean widening the frozen, DSH3-22-owned
``OperatorPolicyInputV1``/``OperatorActionViewV1``/``ReferenceModelViewV1``
dataclasses. This repo already has a dedicated design note on exactly that
question -- ``docs/design/dsh5-10-policy-scorer-adapter-gap-20260727.md``
-- and it recommends *against* doing so unilaterally: extending the
schema "needs its own version-stamped design and test pass before any
training code touches it ... should be a follow-on decision made with the
DSH3/DSH5 policy-head owners, not decided unilaterally by one autonomous
... session extending an existing frozen dataclass's invariants." DSH3-22's
own doc lists ``ordinary_action_count`` being "a bare count, not a
per-token feature" as a *deliberate* scope decision, not an oversight.

This slice therefore takes the adapter-gap doc's own recommended safer
path instead: **design B from that doc** -- "a separate scorer/head for the
history-control decision space, distinct from ``TypedOperatorPolicyScorer``"
-- built here as ``HistoryControlPolicyInputV1``, a wholly new dataclass in
this harness module. It never edits, subclasses, or monkeypatches
``operator_policy_view.py`` or ``operator_feature_encoder.py``; those
modules, and every existing ``OperatorPolicyInputV1`` producer/consumer, are
untouched by this slice. This is additive-by-construction, not merely
additive-by-intent.

What this slice builds:

* :class:`HistoryControlActionViewV1` / :class:`HistoryControlPolicyInputV1`
  -- one row per ``OperatorLegalSetV1.ordinary_nonoperator_actions`` entry,
  carrying only its ``control_kind`` (``action_kind_of``'s existing closed
  vocabulary: ``undo``/``redo``/``checkout``/``merge``). The opaque state id
  embedded in a token (e.g. ``checkout:<state>``'s ``<state>``) is never
  carried into any field -- the same allowlist-not-identity discipline
  DSH3-22 established for the operator side, reused here by construction
  rather than by copying DSH3-22's own code.
* :func:`classify_replay_row_history_control_representability` -- a
  token-only classification (no trace access) of whether a row's
  chosen/rejected pair can be posed as a *same-domain* control-action choice.
* :func:`adapt_replay_row_to_history_control_policy_input` -- for a row
  grounded in a real ``ConversationTraceV1``, reconstructs the exact
  ``OperatorLegalSetV1`` via the eighth slice's own public
  :func:`slm_training.dsl.operators.legal_set_at`, fails closed on a stale
  ``legal_set_fingerprint``, and resolves the row's chosen/rejected control
  tokens to control-row indices.
* :func:`build_history_control_representability_report` /
  :func:`synthesize_history_control_sessions` /
  :func:`evaluate_history_control_representability` -- the corpus-level
  disposition, real numbers below.

**Real result, measured against the seventh slice's own 56-row synthetic
corpus:** representability genuinely widens -- the ``chosen_action`` of
every one of the 47 non-``PRONOUN_FOCUS_FOLLOWUP`` rows classifies as a
control token, up from 0 representable before this slice (the eighth
slice's ``TypedOperatorPolicyScorer`` adapter could represent none of
them). Of these 47, the 46 grounded in a real ``ConversationTraceV1``
(every relation except ``MERGE_SUCCESS``, whose row is grounded on a
``BranchEditV1`` tip with no trace to reconstruct a legal set from --
``NOT_REPRESENTABLE_FOREIGN_STATE``, the same structural reason it stays
unpadded in the seventh slice) actually build a valid
``HistoryControlPolicyInputV1`` row when adapted. But a second,
independently verified structural fact blocks the stretch goal (a trained
pairwise scorer): **every one of those 46 rows' ``rejected_action`` is a
cross-domain operator token, never a same-domain control token.**
``_pick_rejected`` (``replay_preference.py``) draws the rejected candidate
from the *full* legal set (operator and control actions together, sorted
lexicographically); every decision state in this fixture has at least one
legal operator, and a serialized operator token always starts with the
literal ``"OPERATOR "`` -- capital ``O`` (ASCII 79) sorts before every
control-token prefix (``checkout:``, ``redo:``, ``undo``, all lowercase,
ASCII 99+). So the rejected candidate is *always* the operator action in
this corpus, never a control one: zero same-domain (control vs. control)
pairs exist to train or probe a pairwise margin over. This was verified
empirically (``same_domain_pair_count == 0`` over the real corpus), not
assumed -- see ``evaluate_history_control_representability``'s report and
the "Ninth slice" section of ``docs/design/dsh5-10-replay-preference-rows.md``
for the full disposition.

What this slice explicitly does not do: it does not train a scorer over
``HistoryControlPolicyInputV1`` (no same-domain pairs exist to train on in
this corpus); it does not change ``_pick_rejected``'s candidate-selection
policy in ``replay_preference.py`` (that is a row-extraction change, its own
separately-scoped slice, not a model-input-contract change); it does not
touch ``OperatorPolicyInputV1``, ``OperatorActionViewV1``,
``ReferenceModelViewV1``, or ``OperatorFeatureEncoder`` at all; and it does
not attempt a joint operator-and-control policy (design A from the
adapter-gap doc) -- two independent, unmerged heads remain, exactly as that
doc's recommendation described.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from slm_training.dsl.operators import (
    ConversationTraceError,
    ConversationTraceV1,
    OperatorLibraryV1,
    OperatorReplayPreferenceRowV1,
    action_kind_of,
    legal_set_at,
)
from slm_training.dsl.operators.legal_set import OperatorLegalSetV1
from slm_training.dsl.operators.replay_preference import ProvenanceFactory
from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.preference.local_decisions import split_for_group
from slm_training.harnesses.preference.replay_preference_context_view_variants import (
    ROLLBACK_CHAIN_LENGTHS,
    _checkout_and_fork_trace,  # private helper; same import convention this
    _provenance,  # repo's own test suite and the eighth-slice adapter use
    _rollback_chain_trace,
)
from slm_training.levers import MAX_HARNESS_WALL_SECONDS
from slm_training.models.operator_policy_view import (
    OperatorPolicyViewError,
    validate_no_forbidden_fields,
)

__all__ = [
    "ControlActionKind",
    "HistoryControlActionViewV1",
    "HistoryControlAdapterResultV1",
    "HistoryControlPolicyInputV1",
    "HistoryControlProbeReportV1",
    "HistoryControlRepresentabilityReportV1",
    "HistoryControlRowRepresentability",
    "HistoryControlSessionV1",
    "adapt_replay_row_to_history_control_policy_input",
    "build_history_control_policy_input",
    "build_history_control_representability_report",
    "classify_replay_row_history_control_representability",
    "evaluate_history_control_representability",
    "synthesize_history_control_sessions",
]


class ControlActionKind(str, Enum):
    """The closed vocabulary ``action_kind_of`` already classifies control tokens into."""

    UNDO = "undo"
    REDO = "redo"
    CHECKOUT = "checkout"
    MERGE = "merge"


@dataclass(frozen=True)
class HistoryControlActionViewV1:
    """One control-action candidate row: kind only, never its opaque target."""

    row: int
    control_kind: ControlActionKind
    schema: str = "history_control_action_view/v1"

    def __post_init__(self) -> None:
        if self.row < 0:
            raise ValueError("row must be non-negative")

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "row": self.row,
            "control_kind": self.control_kind.value,
        }


@dataclass(frozen=True)
class HistoryControlPolicyInputV1:
    """A sibling, additive control-action view -- not a change to ``OperatorPolicyInputV1``.

    Deliberately a wholly separate dataclass in this harness module, never a
    subclass, monkeypatch, or fork of ``operator_policy_view.py``. See the
    module docstring for why this is the adapter-gap doc's recommended
    "design B" rather than widening the DSH3-22-owned schema.
    """

    control_rows: tuple[HistoryControlActionViewV1, ...]
    legal_set_fingerprint: str
    schema: str = "history_control_policy_input/v1"

    def __post_init__(self) -> None:
        if tuple(view.row for view in self.control_rows) != tuple(
            range(len(self.control_rows))
        ):
            raise ValueError("control_rows must be densely row-indexed from 0")
        validate_no_forbidden_fields(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "control_rows": [view.to_dict() for view in self.control_rows],
            "legal_set_fingerprint": self.legal_set_fingerprint,
        }


def build_history_control_policy_input(
    legal_set: OperatorLegalSetV1,
) -> HistoryControlPolicyInputV1:
    """Build the sibling control-action view from an already-verified legal set.

    One row per ``legal_set.ordinary_nonoperator_actions`` entry -- the exact
    tuple ``OperatorPolicyInputV1`` itself collapses into a bare
    ``ordinary_action_count`` scalar (DSH3-22's own deliberate scope
    boundary). Only ``control_kind`` survives into a row; the opaque state id
    embedded in a ``checkout:<state>``/``redo:<state>``/``merge:<pair>``
    token is never carried into any field.
    """
    control_rows = tuple(
        HistoryControlActionViewV1(
            row=index, control_kind=ControlActionKind(action_kind_of(token))
        )
        for index, token in enumerate(legal_set.ordinary_nonoperator_actions)
    )
    return HistoryControlPolicyInputV1(
        control_rows=control_rows, legal_set_fingerprint=legal_set.fingerprint
    )


class HistoryControlRowRepresentability(str, Enum):
    """Whether a replay-preference row poses a same-domain control-action choice."""

    BOTH_CONTROL = "representable_both_control"
    CHOSEN_CONTROL_REJECTED_OPERATOR = "chosen_control_rejected_cross_domain_operator"
    NOT_REPRESENTABLE_OPERATOR_CHOSEN = "not_representable_operator_chosen"
    NOT_REPRESENTABLE_FOREIGN_STATE = "not_representable_foreign_state"


def classify_replay_row_history_control_representability(
    row: OperatorReplayPreferenceRowV1,
) -> HistoryControlRowRepresentability:
    """Classify a row using only its own recorded action tokens (no trace access).

    ``NOT_REPRESENTABLE_OPERATOR_CHOSEN`` when the *chosen* action itself is
    a serialized operator action (``PRONOUN_FOCUS_FOLLOWUP``'s own domain,
    already representable by the eighth slice's adapter -- out of scope
    here). Otherwise the chosen action is a control token, and the row is
    ``BOTH_CONTROL`` only if the rejected action is *also* a control token;
    when the rejected action is a cross-domain operator token, this module's
    single-domain view cannot pose a pairwise chosen-vs-rejected comparison,
    even though the chosen action alone is representable as a row -- that
    honest, narrower fact is ``CHOSEN_CONTROL_REJECTED_OPERATOR``, not a
    false positive.

    This token-only classification cannot detect a row grounded outside any
    single ``ConversationTraceV1`` (``MERGE_SUCCESS``, whose input state
    lives on a ``BranchEditV1`` tip) -- that further, trace-dependent reason
    only surfaces at :func:`adapt_replay_row_to_history_control_policy_input`
    time, as ``NOT_REPRESENTABLE_FOREIGN_STATE``.
    """
    if action_kind_of(row.chosen_action) == "operator":
        return HistoryControlRowRepresentability.NOT_REPRESENTABLE_OPERATOR_CHOSEN
    if action_kind_of(row.rejected_action) == "operator":
        return HistoryControlRowRepresentability.CHOSEN_CONTROL_REJECTED_OPERATOR
    return HistoryControlRowRepresentability.BOTH_CONTROL


@dataclass(frozen=True)
class HistoryControlAdapterResultV1:
    """One row's control-domain view, or the honest reason it has none."""

    row: OperatorReplayPreferenceRowV1
    representability: HistoryControlRowRepresentability
    policy_input: HistoryControlPolicyInputV1 | None
    accepted_control_row: int | None
    rejected_control_row: int | None


def adapt_replay_row_to_history_control_policy_input(
    row: OperatorReplayPreferenceRowV1,
    trace: ConversationTraceV1,
    *,
    library: OperatorLibraryV1,
    pack: object,
    provenance_for: ProvenanceFactory,
) -> HistoryControlAdapterResultV1:
    """Build ``HistoryControlPolicyInputV1`` for one row, or report why it cannot.

    Fails closed (``OperatorPolicyViewError``) if the reconstructed legal
    set no longer matches the row's own ``legal_set_fingerprint`` -- the same
    integrity discipline :func:`adapt_replay_row_to_typed_policy_input` uses.
    Reports ``NOT_REPRESENTABLE_FOREIGN_STATE`` (never raises) when
    ``row.input_state_id`` is not a member of ``trace``'s own state nodes,
    mirroring ``preference_pairs_from_trace``'s honest-skip convention for
    the same class of row (e.g. a ``MERGE_SUCCESS`` row grounded on a
    ``BranchEditV1`` tip rather than this trace).
    """
    representability = classify_replay_row_history_control_representability(row)
    if representability is HistoryControlRowRepresentability.NOT_REPRESENTABLE_OPERATOR_CHOSEN:
        return HistoryControlAdapterResultV1(
            row=row,
            representability=representability,
            policy_input=None,
            accepted_control_row=None,
            rejected_control_row=None,
        )

    try:
        trace.node(row.input_state_id)
    except ConversationTraceError:
        return HistoryControlAdapterResultV1(
            row=row,
            representability=HistoryControlRowRepresentability.NOT_REPRESENTABLE_FOREIGN_STATE,
            policy_input=None,
            accepted_control_row=None,
            rejected_control_row=None,
        )

    legal_set = legal_set_at(
        trace,
        pack=pack,
        library=library,
        state_id=row.input_state_id,
        provenance_for=provenance_for,
    )
    if legal_set.fingerprint != row.legal_set_fingerprint:
        raise OperatorPolicyViewError("history_control_policy_input.stale_legal_set")

    policy_input = build_history_control_policy_input(legal_set)
    tokens = legal_set.ordinary_nonoperator_actions
    accepted_control_row = next(
        (index for index, token in enumerate(tokens) if token == row.chosen_action), None
    )
    if accepted_control_row is None:
        raise OperatorPolicyViewError("history_control_policy_input.accepted_row_not_found")
    rejected_control_row = next(
        (index for index, token in enumerate(tokens) if token == row.rejected_action), None
    )

    return HistoryControlAdapterResultV1(
        row=row,
        representability=representability,
        policy_input=policy_input,
        accepted_control_row=accepted_control_row,
        rejected_control_row=rejected_control_row,
    )


@dataclass(frozen=True)
class HistoryControlRepresentabilityReportV1:
    """How much of a replay-preference corpus this sibling view can even see.

    Token-only classification (see
    :func:`classify_replay_row_history_control_representability`) -- built
    from rows alone, no trace required.
    """

    total_rows: int
    counts_by_representability: dict[str, int]
    counts_by_relation: dict[str, dict[str, int]]
    schema: str = "history_control_representability_report/v1"

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "total_rows": self.total_rows,
            "counts_by_representability": dict(self.counts_by_representability),
            "counts_by_relation": {
                relation: dict(counts) for relation, counts in self.counts_by_relation.items()
            },
        }


def build_history_control_representability_report(
    rows: tuple[OperatorReplayPreferenceRowV1, ...],
) -> HistoryControlRepresentabilityReportV1:
    """Classify every row in a corpus; representability is reported, never asserted."""
    counts_by_representability: dict[str, int] = {}
    counts_by_relation: dict[str, dict[str, int]] = {}
    for row in rows:
        representability = classify_replay_row_history_control_representability(row).value
        counts_by_representability[representability] = (
            counts_by_representability.get(representability, 0) + 1
        )
        relation = row.semantic_relation.value
        relation_counts = counts_by_relation.setdefault(relation, {})
        relation_counts[representability] = relation_counts.get(representability, 0) + 1
    return HistoryControlRepresentabilityReportV1(
        total_rows=len(rows),
        counts_by_representability=counts_by_representability,
        counts_by_relation=counts_by_relation,
    )


@dataclass(frozen=True)
class HistoryControlSessionV1:
    """One trace-grounded session's rows plus the trace context this module needs.

    Mirrors ``PronounFocusSessionV1`` (``replay_preference_typed_policy_adapter.py``):
    retains ``trace``/``pack``/``library``, which
    ``ReplayPreferenceSessionV1`` (``replay_preference_context_view_variants.py``)
    discards after extraction.
    """

    group_id: str
    split: str
    rows: tuple[OperatorReplayPreferenceRowV1, ...]
    trace: ConversationTraceV1
    pack: object
    library: OperatorLibraryV1


def synthesize_history_control_sessions() -> tuple[HistoryControlSessionV1, ...]:
    """``rollback_chain_<K>`` and ``checkout_and_fork_<K>`` sessions, same split as before.

    Reuses the seventh slice's own trace builders and ``ROLLBACK_CHAIN_LENGTHS``
    ladder (the same private-helper-import convention
    ``replay_preference_typed_policy_adapter.py`` already established for
    ``pronoun_focus_<K>``) so every group's ``split_for_group`` split is
    identical to what the full corpus already assigned -- not a re-split.

    ``merge_success`` (``_merge_session``) is excluded: its one row is
    grounded on a ``BranchEditV1`` tip, not any ``ConversationTraceV1``, so
    there is no trace this session shape can retain for it -- the same
    structural reason it stays unpadded in the seventh slice and
    unrepresentable-by-trace here (see
    :func:`adapt_replay_row_to_history_control_policy_input`'s
    ``NOT_REPRESENTABLE_FOREIGN_STATE`` path, exercised directly in this
    module's tests against a real ``_merge_session()`` row instead).
    ``pronoun_focus_<K>`` is also excluded: every one of its rows has an
    operator-serialized ``chosen_action``, classified
    ``NOT_REPRESENTABLE_OPERATOR_CHOSEN`` by construction, so it never
    contributes a control-domain candidate row.
    """
    from slm_training.dsl.operators import extract_replay_preference_rows

    sessions: list[HistoryControlSessionV1] = []
    for chain_length in ROLLBACK_CHAIN_LENGTHS:
        trace, pack, library = _rollback_chain_trace(chain_length)
        report = extract_replay_preference_rows(
            trace, pack=pack, library=library, provenance_for=_provenance
        )
        group_id = f"rollback_chain_{chain_length}"
        sessions.append(
            HistoryControlSessionV1(
                group_id=group_id,
                split=split_for_group(group_id),
                rows=report.rows,
                trace=trace,
                pack=pack,
                library=library,
            )
        )
    for pad_length in ROLLBACK_CHAIN_LENGTHS:
        trace, pack, library = _checkout_and_fork_trace(pad_length=pad_length)
        report = extract_replay_preference_rows(
            trace, pack=pack, library=library, provenance_for=_provenance
        )
        group_id = f"checkout_and_fork_{pad_length}"
        sessions.append(
            HistoryControlSessionV1(
                group_id=group_id,
                split=split_for_group(group_id),
                rows=report.rows,
                trace=trace,
                pack=pack,
                library=library,
            )
        )
    return tuple(sessions)


@dataclass(frozen=True)
class HistoryControlProbeReportV1:
    """The real ninth-slice disposition: representability widened, pairing did not."""

    representability: HistoryControlRepresentabilityReportV1
    same_domain_pair_count: int
    cross_domain_pair_count: int
    verdict: str
    note: str
    version_stamp: dict
    schema: str = "history_control_probe_report/v1"

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "representability": self.representability.to_dict(),
            "same_domain_pair_count": self.same_domain_pair_count,
            "cross_domain_pair_count": self.cross_domain_pair_count,
            "verdict": self.verdict,
            "note": self.note,
            "version_stamp": self.version_stamp,
        }


_NO_SAME_DOMAIN_PAIRS_NOTE = (
    "no same-domain (control-vs-control) chosen/rejected pair exists in this "
    "corpus to train or probe a pairwise margin over: _pick_rejected "
    "(dsl/operators/replay_preference.py) draws the rejected candidate from "
    "the full legal set (operator and control actions together, sorted "
    "lexicographically), and every decision state in this fixture has at "
    "least one legal operator, whose serialized token always starts with "
    "the literal 'OPERATOR ' -- capital 'O' (ASCII 79) sorts before every "
    "control-token prefix ('checkout:'/'redo:'/'undo', lowercase, ASCII "
    "99+). So the rejected candidate is always the operator action in this "
    "corpus, never a control one. Verified empirically over the real "
    "corpus (same_domain_pair_count == 0), not assumed. Changing this would "
    "require changing _pick_rejected's candidate-selection policy in "
    "replay_preference.py -- a row-extraction change, its own separately "
    "scoped slice, not attempted here."
)

_SAME_DOMAIN_PAIRS_NOTE = (
    "same-domain (control-vs-control) pairs exist in this corpus; a trained "
    "pairwise scorer over HistoryControlPolicyInputV1.control_rows would be "
    "the next real step -- not attempted this slice."
)


def evaluate_history_control_representability(
    *, max_wall_seconds: float = MAX_HARNESS_WALL_SECONDS
) -> HistoryControlProbeReportV1:
    """The real corpus-level disposition: representability report plus pairing counts.

    Enforces the repo's hard run cap, matching every prior slice's harness
    convention, even though a real run completes in a small fraction of a
    second.
    """
    start = time.monotonic()
    sessions = synthesize_history_control_sessions()
    all_rows = tuple(row for session in sessions for row in session.rows)
    representability = build_history_control_representability_report(all_rows)

    same_domain = 0
    cross_domain = 0
    for session in sessions:
        for row in session.rows:
            result = adapt_replay_row_to_history_control_policy_input(
                row,
                session.trace,
                library=session.library,
                pack=session.pack,
                provenance_for=_provenance,
            )
            if result.representability is HistoryControlRowRepresentability.BOTH_CONTROL:
                same_domain += 1
            elif (
                result.representability
                is HistoryControlRowRepresentability.CHOSEN_CONTROL_REJECTED_OPERATOR
            ):
                cross_domain += 1
            if time.monotonic() - start > max_wall_seconds:
                raise TimeoutError(
                    "history-control representability probe exceeded "
                    f"MAX_HARNESS_WALL_SECONDS ({max_wall_seconds}s); a "
                    "timed-out run is never evidence (AGENTS.md hard run cap)"
                )

    verdict = (
        "same_domain_pairs_found_fixture_scale"
        if same_domain > 0
        else "no_same_domain_pairs_fixture_scale"
    )
    return HistoryControlProbeReportV1(
        representability=representability,
        same_domain_pair_count=same_domain,
        cross_domain_pair_count=cross_domain,
        verdict=verdict,
        note=_SAME_DOMAIN_PAIRS_NOTE if same_domain > 0 else _NO_SAME_DOMAIN_PAIRS_NOTE,
        version_stamp=build_version_stamp(
            "harness.preference.replay_preference_history_control_policy"
        ),
    )
