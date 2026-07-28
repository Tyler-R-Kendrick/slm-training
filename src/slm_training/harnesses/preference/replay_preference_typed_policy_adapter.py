"""SLM-418 (DSH5-10) eighth slice: TypedOperatorPolicyScorer adapter.

PR #1202 (seventh slice) named this the next lever: wire a real DSH3
``TypedOperatorPolicyScorer`` (``src/slm_training/harnesses/experiments/
typed_operator_policy.py``) against the replay-preference corpus in place of
the sixth/seventh slices' two-feature linear scorer, to see whether a richer
scorer breaks the flat held-out accuracy that both slices observed.

This slice is an honestly bounded increment, not the full comparison the
seventh slice's next-lever note gestured at. Building it surfaced a real
structural constraint that itself answers part of the question:

``TypedOperatorPolicyScorer`` only consumes ``OperatorPolicyInputV1``
(``slm_training.models.operator_policy_view``), whose ``action_rows`` are
built **only from ``OperatorLegalSetV1.entries``** -- one row per *operator
declaration*. Control actions (``"undo"``, ``"redo:<state>"``,
``"checkout:<state>"``, ``"merge:<pair>"``) live in
``OperatorLegalSetV1.ordinary_nonoperator_actions``, a bare tuple of
strings collapsed into a single scalar (``ordinary_action_count``) --
``OperatorPolicyInputV1`` has **no row, slot, or candidate representing any
control action at all**. Six of the seven replay-preference relations
(``EDIT_THEN_UNDO``, ``UNDO_THEN_REDO``, ``PARTIAL_ROLLBACK``,
``CHECKOUT_ANOTHER_STATE``, ``FORK_THEN_CHOOSE_ONE_BRANCH``,
``MERGE_SUCCESS``) have a chosen and/or rejected action that is a control
token -- these rows are **structurally unrepresentable** as
``TypedOperatorPolicyScorer`` input, not merely unwired. Only
``PRONOUN_FOCUS_FOLLOWUP`` rows are representable: both the chosen and
rejected action are serialized operator actions (``"OPERATOR <id> ..."``)
for the *same* operator (``extract_replay_preference_rows`` draws both from
one ``OperatorLegalEntryV1.legal_actions``), differing only in which
argument is bound to which slot -- exactly the ``TypedOperatorPolicyScorer``
argument head's job.

A second structural fact this slice's own reading of
``operator_policy_objective.route_operator_policy_inference`` and
``TypedOperatorPolicyScorer.forward``/``decide_typed_operator_policy``
surfaces: every representable row in the fixture corpus has exactly one
legal *operator* at its decision state (this module's pronoun fixture
registers a single operator), so ``route_operator_policy_inference`` always
returns ``COMPLETE_SINGLETON`` for these rows, and
``decide_typed_operator_policy`` -- the repo's actual DSH3 decision entry
point -- returns ``selected_argument_rows=()`` and **never calls the model**
for a ``COMPLETE_SINGLETON`` route. This is the repository's own
non-negotiable "deterministic/singleton bypass outranks any learned score"
invariant (AGENTS.md), applied correctly: the argument head is not gated by
this repo's actual inference path here. What this module measures with
:func:`argument_logit_margin` is therefore an explicit **diagnostic probe**
of ``scorer.forward()``'s raw argument logits only -- never a claim about
what ``decide_typed_operator_policy`` would output, and never routed through
the actual decision function.

What this slice builds:

* :func:`classify_replay_row_representability` -- classifies a row as
  representable (both actions are same-operator argument choices) or not,
  using ``replay_preference_context_views.action_kind_of``'s existing
  closed action-kind vocabulary, never a new one.
* :func:`adapt_replay_row_to_typed_policy_input` -- for a representable row,
  reconstructs the exact ``OperatorLegalSetV1`` the row was extracted
  against (:func:`slm_training.dsl.operators.replay_preference.legal_set_at`)
  and fails closed (``OperatorPolicyViewError``) if its fingerprint no
  longer matches the row's own recorded ``legal_set_fingerprint`` -- an
  integrity check, not a convenience default. Builds
  ``OperatorPolicyInputV1`` via the DSH3-owned
  ``build_operator_policy_input`` (no shadow input builder), then resolves
  the row's chosen/rejected bound arguments to the exact reference rows
  ``OperatorPolicyInputV1`` assigned them.
* :func:`argument_logit_margin` -- the diagnostic-only forward-logit probe
  described above.
* :func:`build_representability_report` -- classifies every row in a
  session corpus (representable vs not, by relation), so the real scope
  limit is reported as data, not asserted in prose alone.
* :func:`evaluate_typed_policy_argument_probe` -- trains
  ``TypedOperatorPolicyScorer`` on the *representable* subset's train split
  and reports the probe's pairwise margin accuracy on the *representable*
  subset's held-out split, using the exact same ``split_for_group`` session
  split the seventh slice's corpus already assigned to
  ``pronoun_focus_<pad_length>`` sessions (no re-split).

What this slice explicitly does not do: it does not attempt to invent a
control-action representation for the six unrepresentable relations (that
would require extending ``OperatorPolicyInputV1`` itself -- a DSH3-owned
contract change, genuinely out of scope here); it does not call
``decide_typed_operator_policy`` or claim any certified decision-path
result; and it does not compare against or replace the sixth/seventh
slices' two-feature linear scorer, which stays as-is.

Ninth slice (v2): the eighth slice's own named lever (1). The flatness above
was diagnosed as a *representational* ceiling -- ``OperatorFeatureEncoder``
had no field capable of expressing "which reference did the immediately
preceding turn touch," which is the entire content of
``PRONOUN_FOCUS_FOLLOWUP``'s preference signal. That field now exists
(``ReferenceModelViewV1.recently_touched``, tagged post hoc via
``tag_recently_touched_rows``), and this adapter populates it from
``dsl.operators.replay_preference.refs_touched_by_preceding_turn`` -- the
module-owned ``_touched_refs`` focus notion, addressed by decision state.

Two honesty constraints this slice is bound by, both reported as data rather
than asserted in prose:

* The flag is a strictly **pre-decision history fact**: it reads the already
  recorded ``OperatorApplicationV1.arguments`` of the turn that *produced*
  the decision state. Nothing about the choice being scored enters it.
* On *this fixture* it is nevertheless **definitionally aligned with the
  label**: ``extract_replay_preference_rows``'s pronoun-focus branch emits a
  row only when the chosen action's refs intersect the focus set and the
  rejected action's refs do not. So a non-zero margin here demonstrates that
  the boundary and encoder *can represent and learn* this relation --
  it is not, and must never be reported as, evidence that the feature
  generalizes to unseen preference structure. :func:`evaluate_typed_policy_argument_probe`
  emits the exact chosen/rejected tag counts (``recently_touched_diagnostics``)
  and a verbatim ``recently_touched_note`` so a reader cannot lose this.

See ``docs/design/dsh5-10-replay-preference-rows.md``'s "Eighth slice" and
"Ninth slice" for the real measured numbers.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from slm_training.dsl.operators import (
    ConversationTraceV1,
    OperatorLibraryV1,
    OperatorReplayPreferenceRowV1,
    action_kind_of,
    deserialize_operator_action,
    legal_set_at,
    refs_touched_by_preceding_turn,
)
from slm_training.dsl.operators.replay_preference import ProvenanceFactory
from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.experiments.typed_operator_policy import (
    TypedOperatorPolicyExampleV1,
    TypedOperatorPolicyScorer,
    train_typed_operator_policy,
)
from slm_training.harnesses.preference.local_decisions import split_for_group
from slm_training.harnesses.preference.replay_preference_context_view_variants import (
    ROLLBACK_CHAIN_LENGTHS,
    _pronoun_trace,  # private helper; same import convention this repo's own test suite uses
    _provenance,
)
from slm_training.levers import MAX_HARNESS_WALL_SECONDS
from slm_training.models.operator_policy_view import (
    OperatorPolicyInputV1,
    OperatorPolicyViewError,
    build_operator_policy_input,
    tag_recently_touched_rows,
)

__all__ = [
    "PronounFocusSessionV1",
    "ReplayRowRepresentability",
    "TypedPolicyAdapterResultV1",
    "TypedPolicyArgumentProbeReportV1",
    "TypedPolicyRepresentabilityReportV1",
    "adapt_replay_row_to_typed_policy_input",
    "argument_logit_margin",
    "build_representability_report",
    "classify_replay_row_representability",
    "evaluate_typed_policy_argument_probe",
    "synthesize_pronoun_focus_sessions",
]


class ReplayRowRepresentability(str, Enum):
    """Whether a replay-preference row can be posed as TypedOperatorPolicyScorer input."""

    ARGUMENT_CHOICE = "representable_argument_choice"
    NOT_REPRESENTABLE_CONTROL_ACTION = "not_representable_control_action"
    NOT_REPRESENTABLE_DIFFERENT_OPERATOR = "not_representable_different_operator"


def classify_replay_row_representability(
    row: OperatorReplayPreferenceRowV1,
) -> ReplayRowRepresentability:
    """Classify a row using only its own recorded action tokens.

    A row is representable only when both ``chosen_action`` and
    ``rejected_action`` are serialized operator actions
    (``action_kind_of(...) == "operator"``) for the *same* operator --
    ``TypedOperatorPolicyScorer.action_rows`` carries no row for a control
    action, and this module builds no substitute representation for one.
    """
    if (
        action_kind_of(row.chosen_action) != "operator"
        or action_kind_of(row.rejected_action) != "operator"
    ):
        return ReplayRowRepresentability.NOT_REPRESENTABLE_CONTROL_ACTION
    chosen_operator_id, _ = deserialize_operator_action(row.chosen_action)
    rejected_operator_id, _ = deserialize_operator_action(row.rejected_action)
    if chosen_operator_id != rejected_operator_id:
        return ReplayRowRepresentability.NOT_REPRESENTABLE_DIFFERENT_OPERATOR
    return ReplayRowRepresentability.ARGUMENT_CHOICE


@dataclass(frozen=True)
class TypedPolicyAdapterResultV1:
    """One row's typed-policy view, or the honest reason it has none."""

    row: OperatorReplayPreferenceRowV1
    representability: ReplayRowRepresentability
    policy_input: OperatorPolicyInputV1 | None
    accepted_action_row: int | None
    accepted_argument_rows: tuple[tuple[str, int], ...]
    rejected_argument_rows: tuple[tuple[str, int], ...]
    recently_touched_rows: tuple[int, ...] = ()

    def to_typed_example(self, row_id: str) -> TypedOperatorPolicyExampleV1:
        if self.policy_input is None or self.accepted_action_row is None:
            raise OperatorPolicyViewError("replay_preference_policy_input.not_representable")
        return TypedOperatorPolicyExampleV1(
            row_id=row_id,
            view=self.policy_input,
            accepted_action_row=self.accepted_action_row,
            accepted_argument_rows=self.accepted_argument_rows,
        )


def adapt_replay_row_to_typed_policy_input(
    row: OperatorReplayPreferenceRowV1,
    trace: ConversationTraceV1,
    *,
    library: OperatorLibraryV1,
    pack: object,
    provenance_for: ProvenanceFactory,
) -> TypedPolicyAdapterResultV1:
    """Build ``OperatorPolicyInputV1`` for one row, or report why it cannot.

    Fails closed (``OperatorPolicyViewError``) if the reconstructed legal
    set no longer matches the row's own ``legal_set_fingerprint`` -- a
    stale trace/library can never silently produce a mismatched view.
    """
    representability = classify_replay_row_representability(row)
    if representability is not ReplayRowRepresentability.ARGUMENT_CHOICE:
        return TypedPolicyAdapterResultV1(
            row=row,
            representability=representability,
            policy_input=None,
            accepted_action_row=None,
            accepted_argument_rows=(),
            rejected_argument_rows=(),
        )

    legal_set = legal_set_at(
        trace,
        pack=pack,
        library=library,
        state_id=row.input_state_id,
        provenance_for=provenance_for,
    )
    if legal_set.fingerprint != row.legal_set_fingerprint:
        raise OperatorPolicyViewError("replay_preference_policy_input.stale_legal_set")

    node = trace.node(row.input_state_id)
    policy_input = build_operator_policy_input(node.reference_table, legal_set, library)

    table_entries = (*node.reference_table.entries, *node.reference_table.selectors)
    row_by_ref = {
        (entry.ref.KIND, entry.ref.opaque_id): reference_row
        for reference_row, entry in enumerate(table_entries)
    }

    # Ninth slice: attach the bounded history bit the eighth slice found
    # missing. ``refs_touched_by_preceding_turn`` reads the exact
    # ``OperatorApplicationV1.arguments`` of the already-recorded turn that
    # *produced* this decision state -- strictly pre-decision, never the
    # outcome of the choice being scored. The opaque refs are resolved
    # through the same ``row_by_ref`` map the accepted/rejected arguments use
    # and then discarded: only row numbers cross into the sanitized view, and
    # only as a boolean. A touched ref that is no longer in this state's
    # reference table is simply not tagged (no row exists to carry it).
    recently_touched_rows = tuple(
        sorted(
            row_by_ref[(ref.KIND, ref.opaque_id)]
            for ref in refs_touched_by_preceding_turn(trace, row.input_state_id)
            if (ref.KIND, ref.opaque_id) in row_by_ref
        )
    )
    policy_input = tag_recently_touched_rows(policy_input, recently_touched_rows)

    chosen_operator_id, chosen_arguments = deserialize_operator_action(row.chosen_action)
    _, rejected_arguments = deserialize_operator_action(row.rejected_action)

    action_row = next(
        (
            action.row
            for action in policy_input.action_rows
            if action.operator_id == chosen_operator_id
        ),
        None,
    )
    if action_row is None:
        raise OperatorPolicyViewError("replay_preference_policy_input.action_row_not_found")

    accepted_argument_rows = tuple(
        (argument.slot_id, row_by_ref[(argument.value.KIND, argument.value.opaque_id)])
        for argument in chosen_arguments
    )
    rejected_argument_rows = tuple(
        (argument.slot_id, row_by_ref[(argument.value.KIND, argument.value.opaque_id)])
        for argument in rejected_arguments
    )

    return TypedPolicyAdapterResultV1(
        row=row,
        representability=representability,
        policy_input=policy_input,
        accepted_action_row=action_row,
        accepted_argument_rows=accepted_argument_rows,
        rejected_argument_rows=rejected_argument_rows,
        recently_touched_rows=recently_touched_rows,
    )


def argument_logit_margin(
    scorer: TypedOperatorPolicyScorer, result: TypedPolicyAdapterResultV1
) -> float:
    """Diagnostic-only chosen-minus-rejected argument logit margin.

    Calls ``scorer.forward()`` directly, deliberately bypassing
    ``decide_typed_operator_policy``'s route gate. Every representable row
    in this module's fixture corpus has exactly one legal operator, so
    ``route_operator_policy_inference`` always resolves ``COMPLETE_
    SINGLETON`` and the repo's real decision entry point never calls the
    model for these rows (the deterministic/singleton-bypass invariant,
    correctly applied). This function measures only whether the trained
    argument head *could* discriminate chosen from rejected -- never what
    the gated inference path would actually decide.
    """
    if result.policy_input is None or result.accepted_action_row is None:
        raise OperatorPolicyViewError("replay_preference_policy_input.not_representable")
    _, argument_logits = scorer(result.policy_input)
    margins: list[float] = []
    for (chosen_slot, chosen_row), (rejected_slot, rejected_row) in zip(
        result.accepted_argument_rows, result.rejected_argument_rows, strict=True
    ):
        if chosen_slot != rejected_slot:
            raise ValueError("chosen/rejected argument slots must align")
        logits, candidate_rows = argument_logits[(result.accepted_action_row, chosen_slot)]
        chosen_logit = logits[candidate_rows.index(chosen_row)]
        rejected_logit = logits[candidate_rows.index(rejected_row)]
        margins.append(float(chosen_logit - rejected_logit))
    if not margins:
        raise ValueError("representable row produced no aligned argument slots")
    return sum(margins) / len(margins)


@dataclass(frozen=True)
class TypedPolicyRepresentabilityReportV1:
    """How much of a replay-preference corpus TypedOperatorPolicyScorer can even see."""

    total_rows: int
    counts_by_representability: dict[str, int]
    counts_by_relation: dict[str, dict[str, int]]
    schema: str = "typed_policy_representability_report/v1"

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "total_rows": self.total_rows,
            "counts_by_representability": dict(self.counts_by_representability),
            "counts_by_relation": {
                relation: dict(counts) for relation, counts in self.counts_by_relation.items()
            },
        }


def build_representability_report(
    rows: tuple[OperatorReplayPreferenceRowV1, ...],
) -> TypedPolicyRepresentabilityReportV1:
    """Classify every row in a corpus; representability is reported, never asserted."""
    counts_by_representability: dict[str, int] = {}
    counts_by_relation: dict[str, dict[str, int]] = {}
    for row in rows:
        representability = classify_replay_row_representability(row).value
        counts_by_representability[representability] = (
            counts_by_representability.get(representability, 0) + 1
        )
        relation = row.semantic_relation.value
        relation_counts = counts_by_relation.setdefault(relation, {})
        relation_counts[representability] = relation_counts.get(representability, 0) + 1
    return TypedPolicyRepresentabilityReportV1(
        total_rows=len(rows),
        counts_by_representability=counts_by_representability,
        counts_by_relation=counts_by_relation,
    )


@dataclass(frozen=True)
class PronounFocusSessionV1:
    """One ``pronoun_focus_<pad_length>`` session's rows plus replay context.

    Mirrors ``ReplayPreferenceSessionV1``
    (``replay_preference_context_view_variants.py``) but additionally
    retains ``trace``/``pack``/``library`` -- required to reconstruct
    ``OperatorLegalSetV1`` per row via :func:`adapt_replay_row_to_typed_policy_input`,
    which that dataclass discards after extraction.
    """

    group_id: str
    split: str
    rows: tuple[OperatorReplayPreferenceRowV1, ...]
    trace: ConversationTraceV1
    pack: object
    library: OperatorLibraryV1


def synthesize_pronoun_focus_sessions() -> tuple[PronounFocusSessionV1, ...]:
    """The seventh slice's own ``pronoun_focus_<pad_length>`` session ladder.

    Reuses ``replay_preference_context_view_variants._pronoun_trace`` and
    ``ROLLBACK_CHAIN_LENGTHS`` directly (the same private-helper-import
    convention this repo's own
    ``tests/test_harnesses/preference/test_operator_history_pairs.py``
    already uses) so the train/held-out split assigned to each
    ``pronoun_focus_<pad_length>`` group by ``split_for_group`` is
    identical to the one the seventh slice's full 16-session corpus used --
    this is not a re-split, it is the same split over the same group ids.
    """
    from slm_training.dsl.operators import extract_replay_preference_rows

    sessions = []
    for pad_length in ROLLBACK_CHAIN_LENGTHS:
        trace, pack, library = _pronoun_trace(pad_length=pad_length)
        report = extract_replay_preference_rows(
            trace, pack=pack, library=library, provenance_for=_provenance
        )
        group_id = f"pronoun_focus_{pad_length}"
        sessions.append(
            PronounFocusSessionV1(
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
class TypedPolicyArgumentProbeReportV1:
    """Real train/held-out result for the representable (PRONOUN_FOCUS_FOLLOWUP) subset.

    Diagnostic-probe evidence only -- see the module docstring and
    :func:`argument_logit_margin` for exactly what this does and does not
    claim about the repo's actual gated decision path.
    """

    representability: TypedPolicyRepresentabilityReportV1
    train_example_count: int
    held_out_pair_count: int
    pairwise_margin_accuracy: float
    mean_margin: float
    recently_touched_diagnostics: dict
    recently_touched_note: str
    gating_note: str
    version_stamp: dict
    schema: str = "typed_policy_argument_probe_report/v1"

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "representability": self.representability.to_dict(),
            "train_example_count": self.train_example_count,
            "held_out_pair_count": self.held_out_pair_count,
            "pairwise_margin_accuracy": self.pairwise_margin_accuracy,
            "mean_margin": self.mean_margin,
            "recently_touched_diagnostics": dict(self.recently_touched_diagnostics),
            "recently_touched_note": self.recently_touched_note,
            "gating_note": self.gating_note,
            "version_stamp": self.version_stamp,
        }


_GATING_NOTE = (
    "diagnostic probe only: every representable row has exactly one legal "
    "operator, so route_operator_policy_inference always resolves "
    "COMPLETE_SINGLETON and decide_typed_operator_policy -- the repo's real "
    "decision entry point -- never calls this model for these rows "
    "(the deterministic/singleton-bypass invariant, applied correctly). "
    "This report measures only whether the trained argument head could "
    "discriminate chosen from rejected in a raw forward pass, never what "
    "the gated inference path would output."
)

_RECENTLY_TOUCHED_NOTE = (
    "ReferenceModelViewV1.recently_touched is a strictly pre-decision "
    "history bit (the refs the already-recorded turn that produced this "
    "decision state bound as arguments), never an outcome or an identity. "
    "On this fixture it is nevertheless definitionally aligned with the "
    "label: extract_replay_preference_rows emits a PRONOUN_FOCUS_FOLLOWUP "
    "row only when the chosen action's refs intersect that focus set and "
    "the rejected action's refs do not. A non-zero margin therefore shows "
    "this boundary and encoder can represent and learn the relation; it is "
    "never evidence that the feature generalizes to unseen preference "
    "structure, and no such claim is made."
)


def _recently_touched_diagnostics(
    results: Sequence[TypedPolicyAdapterResultV1],
) -> dict[str, int]:
    """Count how the history tag actually landed, per representable row.

    Reported rather than asserted: ``chosen_rows_tagged`` vs
    ``rejected_rows_tagged`` is the exact measurement of the
    label-alignment caveat in :data:`_RECENTLY_TOUCHED_NOTE`.
    """
    chosen_tagged = 0
    rejected_tagged = 0
    tagged_rows = 0
    for result in results:
        tagged = set(result.recently_touched_rows)
        tagged_rows += len(tagged)
        chosen_tagged += sum(
            1 for _, reference_row in result.accepted_argument_rows
            if reference_row in tagged
        )
        rejected_tagged += sum(
            1 for _, reference_row in result.rejected_argument_rows
            if reference_row in tagged
        )
    return {
        "representable_rows": len(results),
        "tagged_reference_rows": tagged_rows,
        "chosen_rows_tagged": chosen_tagged,
        "rejected_rows_tagged": rejected_tagged,
    }


def evaluate_typed_policy_argument_probe(
    *,
    dim: int = 16,
    steps: int = 200,
    learning_rate: float = 0.05,
    seed: int = 0,
    max_wall_seconds: float = MAX_HARNESS_WALL_SECONDS,
) -> TypedPolicyArgumentProbeReportV1:
    """Train ``TypedOperatorPolicyScorer`` on representable rows; probe held-out.

    Fails closed (raises) on an empty train or held-out split, exactly like
    the seventh slice's own comparison harness, and enforces the repo's
    hard run cap.
    """
    import torch

    start = time.monotonic()
    sessions = synthesize_pronoun_focus_sessions()
    all_rows = tuple(row for session in sessions for row in session.rows)
    representability = build_representability_report(all_rows)

    train_results: list[TypedPolicyAdapterResultV1] = []
    held_out_results: list[TypedPolicyAdapterResultV1] = []
    for session in sessions:
        for row in session.rows:
            if classify_replay_row_representability(row) is not ReplayRowRepresentability.ARGUMENT_CHOICE:
                continue
            result = adapt_replay_row_to_typed_policy_input(
                row,
                session.trace,
                library=session.library,
                pack=session.pack,
                provenance_for=_provenance,
            )
            if session.split == "train":
                train_results.append(result)
            else:
                held_out_results.append(result)

    if not train_results or not held_out_results:
        raise ValueError(
            "evaluate_typed_policy_argument_probe requires at least one "
            "representable row in each split; conversation variants stay in "
            "one split, so the corpus cannot be re-split to force this"
        )
    if time.monotonic() - start > max_wall_seconds:
        raise TimeoutError(
            "typed-policy argument probe exceeded MAX_HARNESS_WALL_SECONDS "
            f"({max_wall_seconds}s); a timed-out run is never evidence "
            "(AGENTS.md hard run cap)"
        )

    torch.manual_seed(seed)
    examples = tuple(
        result.to_typed_example(row_id=f"{result.row.input_state_id}:{index}")
        for index, result in enumerate(train_results)
    )
    scorer = TypedOperatorPolicyScorer.from_examples(examples, dim=dim)
    train_typed_operator_policy(scorer, examples, steps=steps, learning_rate=learning_rate)

    if time.monotonic() - start > max_wall_seconds:
        raise TimeoutError(
            "typed-policy argument probe exceeded MAX_HARNESS_WALL_SECONDS "
            f"({max_wall_seconds}s); a timed-out run is never evidence "
            "(AGENTS.md hard run cap)"
        )

    correct = 0.0
    margin_sum = 0.0
    with torch.no_grad():
        for result in held_out_results:
            margin = argument_logit_margin(scorer, result)
            margin_sum += margin
            correct += 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)

    return TypedPolicyArgumentProbeReportV1(
        representability=representability,
        train_example_count=len(train_results),
        held_out_pair_count=len(held_out_results),
        pairwise_margin_accuracy=correct / len(held_out_results),
        mean_margin=margin_sum / len(held_out_results),
        recently_touched_diagnostics=_recently_touched_diagnostics(
            (*train_results, *held_out_results)
        ),
        recently_touched_note=_RECENTLY_TOUCHED_NOTE,
        gating_note=_GATING_NOTE,
        version_stamp=build_version_stamp(
            "harness.preference.replay_preference_typed_policy_adapter"
        ),
    )
