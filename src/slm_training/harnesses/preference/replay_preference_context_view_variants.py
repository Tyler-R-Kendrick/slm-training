"""SLM-418 (DSH5-10): bounded, fixture-scale matched context-view variants.

Wires ``slm_training.dsl.operators.replay_preference_context_views``'s five
context-view input representations into the "train matched SFT/preference
variants using the DSH3-selected policy/control heads ... event and state IDs
are join/evidence keys, never semantic embeddings" bullet of SLM-418. This is
an honestly reduced slice of that bullet, not the full thing:

* The "matched variant" trained per ``(context_view, turn_depth)`` cell is a
  tiny **two-feature linear pairwise scorer** (``history_repeat_score``,
  ``is_history_control``), never a DSH3 head / checkpoint / neural policy.
  Both features are computed purely from opaque action-kind tokens and
  visible-receipt counts -- never transcript text, never a semantic
  embedding of any state/event id. Wiring a real DSH3 policy/control head
  here is out of scope for this slice (see the disposition doc).
* The corpus is a small, deterministic, **synthetic** set of fixture-scale
  conversation sessions (``synthesize_bounded_session_corpus``) built with
  the same public compiler/operator contracts every other operator test in
  this repo uses -- not real user telemetry. Numbers this module reports are
  fixture-scale wiring evidence, never a certified held-out-benefit or
  ship-readiness claim.
* Sessions are split train/held-out by ``split_for_group`` (the same
  stable-hash group split ``local_decisions.py`` already uses), and every row
  from one session stays in that session's split -- "conversation variants
  stay in one split," per the issue's own adversarial control.
* Measured: pairwise chosen>rejected preference accuracy (overall and by
  semantic relation), a narrow pairwise calibration proxy, and corpus-
  composition rates (undo/redo/rollback family share). NOT measured here,
  and explicitly out of scope: full action/operator/argument accuracy
  against a real policy, unintended-mutation telemetry (structurally
  impossible -- this module only scores existing legal candidates and never
  calls ``OperatorLibraryV1.apply``), and CAP0/CAP1/CAP2 retention (requires
  the full CAP-gated eval suite integrated with a trained checkpoint).

See ``docs/design/dsh5-10-replay-preference-rows.md`` for the full
disposition.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import product

from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BindingPhase,
    BoundArgumentV1,
    BranchEditV1,
    CompilerCoverage,
    ConversationStateNodeV1,
    ConversationTraceV1,
    EffectDeltaKind,
    EffectDeltaV1,
    OperatorArgumentSlotV1,
    OperatorEventMemoryReportV1,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorRejectedError,
    OperatorStateV1,
    RefKind,
    ReferenceDescriptorV1,
    RegisteredOperatorV1,
    ReplayPreferenceRelation,
    append_operator_turn,
    branch_fingerprint,
    build_reference_table,
    checkout_conversation_state,
    clone_reference_table_for_branch,
    create_conversation_trace,
    extract_merge_preference_row,
    extract_replay_preference_rows,
    fork_conversation,
    merge_conversation_branches,
    redo_conversation,
    undo_conversation,
)
from slm_training.dsl.operators.replay_preference_context_views import (
    TURN_DEPTHS,
    ContextView,
    ContextViewWindowV1,
    OperatorTurnReceiptV1,
    action_kind_of,
    build_context_view_window,
    receipts_from_trace,
    state_lookup_from_trace,
)
from slm_training.dsl.pack import get_pack
from slm_training.harness_core.versioning import build_version_stamp
from slm_training.harnesses.preference.local_decisions import Split, split_for_group
from slm_training.levers import MAX_HARNESS_WALL_SECONDS

# --------------------------------------------------------------------------- #
# Synthetic, fixture-scale conversation sessions.
#
# Every session below is built from the same public compiler/operator
# contracts (`OperatorStateV1`, `OperatorLibraryV1`, `create_conversation_trace`
# and friends) that every other operator test in this repo uses -- no
# transcript text, no shortcuts around the legal-set/replay machinery.
# --------------------------------------------------------------------------- #


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _provenance(state: OperatorStateV1, *, request_id: str = "ablation-request") -> ApplicationProvenanceV1:
    return ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.replay_preference_ablation_fixture",
        compiler_version="v1",
        source_artifact_digest=_sha(state.source),
        request_id=request_id,
    )


def _counter_source(step: int) -> str:
    return f'root = TextContent(":hero.step_{step}")'


_COUNTER_OPERATOR_ID = "openui.replay_pref_ablation_counter"


def _counter_pack_and_root() -> tuple[object, OperatorLibraryV1, OperatorStateV1]:
    """A zero-argument operator whose output is strictly novel every step.

    Unlike a small-cycle fixture (e.g. title -> body -> caption -> title),
    this never revisits an already-materialized state's content, so a chain
    of any length (needed for the 1/2/4/8/16 turn-depth matrix) never
    collides with an earlier state's digest.
    """
    base_pack = get_pack("openui")
    root_state = OperatorStateV1.from_source(base_pack, _counter_source(0))

    def execute(state: OperatorStateV1, _arguments: tuple) -> OperatorMutationV1:
        prefix = ':hero.step_'
        if prefix not in state.source:
            raise OperatorRejectedError("ablation_fixture.no_counter")
        try:
            current = int(state.source.split(prefix, 1)[1].split('"', 1)[0])
        except ValueError as exc:
            raise OperatorRejectedError("ablation_fixture.bad_counter") from exc
        return OperatorMutationV1(
            source=_counter_source(current + 1),
            effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
        )

    declaration = AstOperatorV1(
        operator_id=_COUNTER_OPERATOR_ID,
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(),
        preconditions=(),
        effect_signature=(),
        locality="node",
        cost=1.0,
    )
    library = OperatorLibraryV1((RegisteredOperatorV1(declaration, execute),))
    pack = replace(base_pack, operator_library=library)
    return pack, library, root_state


def _table(state: OperatorStateV1, branch: str, *, seed: int, request_id: str = "ablation-request"):
    return build_reference_table(
        request_id=request_id,
        state_digest=state.state_digest,
        branch_digest=branch,
        descriptors=(
            ReferenceDescriptorV1(
                ref_kind=RefKind.VALUE,
                semantic_fingerprint=_sha("ablation-visible-value"),
                value_type="openui.string",
            ),
        ),
        seed=seed,
    )


def _bump(pack, library, trace, *, seed: int):
    """Apply the counter operator once; returns the extended trace."""
    result = library.apply(pack, trace.current.state, _COUNTER_OPERATOR_ID, (), _provenance(trace.current.state))
    assert result.succeeded and result.state is not None
    output_table = _table(result.state, trace.current.branch_digest, seed=seed)
    return append_operator_turn(
        trace,
        pack=pack,
        library=library,
        application=result.application,
        output_reference_table=output_table,
    )


def _rollback_chain_trace(
    chain_length: int,
) -> tuple[ConversationTraceV1, object, OperatorLibraryV1]:
    """``chain_length`` novel edits, then a full rollback, then one redo.

    Produces one ``EDIT_THEN_UNDO`` row, ``chain_length - 1`` ``PARTIAL_
    ROLLBACK`` rows at successively deeper positions in the rollback (later
    rows see more consecutive prior "undo" receipts -- the natural turn-
    depth ladder this ablation studies), and one ``UNDO_THEN_REDO`` row.
    """
    pack, library, root_state = _counter_pack_and_root()
    branch = branch_fingerprint(root_state.state_digest, _sha(f"rollback-{chain_length}"))
    root_table = _table(root_state, branch, seed=1)
    trace = create_conversation_trace(
        pack=pack,
        root_state=root_state,
        root_reference_table=root_table,
        provenance=_provenance(root_state),
    )
    for step in range(chain_length):
        trace = _bump(pack, library, trace, seed=step + 2)
    for _ in range(chain_length):
        trace = undo_conversation(trace, provenance=_provenance(trace.current.state))
    # One redo back up, to also exercise UNDO_THEN_REDO.
    child_id = next(
        node.state_id
        for node in trace.state_nodes
        if node.parent_state_id == trace.current_state_id
        and node.branch_digest == trace.current.branch_digest
    )
    trace = redo_conversation(trace, target_state_id=child_id, provenance=_provenance(trace.current.state))
    return trace, pack, library


def _checkout_and_fork_trace(
    pad_length: int = 3,
) -> tuple[ConversationTraceV1, object, OperatorLibraryV1]:
    """One ``CHECKOUT_ANOTHER_STATE`` row and one ``FORK_THEN_CHOOSE_ONE_BRANCH`` row.

    ``pad_length`` novel edits precede the checkout/fork pair, giving the
    ``STATE_PLUS_RECENT_RECEIPTS``/``STATE_PLUS_RETRIEVED_EVENTS`` windows
    real turn-depth variation to expose at the decision point -- the padding
    is plain zero-argument counter-operator edits (the same fixture the
    rollback-chain sessions use), so it never changes which two relations
    this trace produces, only how much prior history is visible before them.
    """
    pack, library, root_state = _counter_pack_and_root()
    branch = branch_fingerprint(root_state.state_digest, _sha(f"checkout-fork-{pad_length}"))
    root_table = _table(root_state, branch, seed=1)
    trace = create_conversation_trace(
        pack=pack,
        root_state=root_state,
        root_reference_table=root_table,
        provenance=_provenance(root_state),
    )
    for step in range(pad_length):
        trace = _bump(pack, library, trace, seed=step + 2)
    tip_id = trace.current_state_id
    trace = checkout_conversation_state(
        trace, target_state_id=trace.root_state_id, provenance=_provenance(trace.current.state)
    )
    trace = fork_conversation(
        trace,
        branch_nonce_digest=_sha("checkout-fork-branch"),
        reference_seed=41,
        provenance=_provenance(trace.current.state),
    )
    trace = checkout_conversation_state(
        trace, target_state_id=tip_id, provenance=_provenance(trace.current.state)
    )
    return trace, pack, library


_PRONOUN_OPERATOR_ID = "openui.replay_pref_ablation_pronoun"
_PRONOUN_SOURCE = 'root = TextContent(":hero.title")'
_PRONOUN_TARGETS = {0: ":hero.target_a", 1: ":hero.target_b"}


def _pronoun_trace(
    pad_length: int = 0,
) -> tuple[ConversationTraceV1, object, OperatorLibraryV1]:
    """One ``PRONOUN_FOCUS_FOLLOWUP`` row: two edits continuing the same ref.

    ``pad_length`` warm-up edits precede the focus-continuing pair, alternating
    between the two refs (starting with ref 1) so consecutive warm-up turns
    never share a touched ref and cannot themselves spuriously satisfy the
    focus-overlap condition -- only the final ``(ref 0, ref 0)`` pair this
    trace always ends on is guaranteed to classify as
    ``PRONOUN_FOCUS_FOLLOWUP``. This gives the turn-depth ablation genuine
    prior history to expose at that decision point, mirroring the
    rollback-chain sessions' own padding.
    """
    base_pack = get_pack("openui")
    root_state = OperatorStateV1.from_source(base_pack, _PRONOUN_SOURCE)
    descriptors = tuple(
        ReferenceDescriptorV1(
            ref_kind=RefKind.VALUE,
            semantic_fingerprint=_sha(f"ablation-pronoun-target-{index}"),
            value_type="openui.string",
        )
        for index in (0, 1)
    )
    branch = branch_fingerprint(root_state.state_digest, _sha(f"pronoun-branch-{pad_length}"))

    def table_for(state: OperatorStateV1):
        return build_reference_table(
            request_id="ablation-request",
            state_digest=state.state_digest,
            branch_digest=branch,
            descriptors=descriptors,
            seed=9,
        )

    root_table = table_for(root_state)
    semantic_by_ref = {entry.ref: entry.descriptor.semantic_fingerprint for entry in root_table.entries}
    index_by_semantic = {descriptor.semantic_fingerprint: index for index, descriptor in enumerate(descriptors)}
    ref_by_index = {
        index_by_semantic[entry.descriptor.semantic_fingerprint]: entry.ref for entry in root_table.entries
    }

    def execute(_state: OperatorStateV1, arguments: tuple[BoundArgumentV1, ...]) -> OperatorMutationV1:
        index = index_by_semantic[semantic_by_ref[arguments[0].value]]
        return OperatorMutationV1(
            source=f'root = TextContent("{_PRONOUN_TARGETS[index]}")',
            effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
        )

    declaration = AstOperatorV1(
        operator_id=_PRONOUN_OPERATOR_ID,
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(OperatorArgumentSlotV1("value", RefKind.VALUE, BindingPhase.APPLICATION),),
        preconditions=(),
        effect_signature=(),
        locality="node",
        cost=1.0,
    )
    library = OperatorLibraryV1((RegisteredOperatorV1(declaration, execute),))
    pack = replace(base_pack, operator_library=library)
    trace = create_conversation_trace(
        pack=pack, root_state=root_state, root_reference_table=root_table, provenance=_provenance(root_state)
    )

    def apply_edit(current_trace: ConversationTraceV1, ref) -> ConversationTraceV1:
        result = library.apply(pack, current_trace.current.state, _PRONOUN_OPERATOR_ID, (BoundArgumentV1("value", ref),), _provenance(current_trace.current.state))
        assert result.succeeded and result.state is not None
        return append_operator_turn(
            current_trace,
            pack=pack,
            library=library,
            application=result.application,
            output_reference_table=table_for(result.state),
        )

    for step in range(pad_length):
        trace = apply_edit(trace, ref_by_index[1 - (step % 2)])
    trace = apply_edit(trace, ref_by_index[0])
    trace = apply_edit(trace, ref_by_index[0])
    return trace, pack, library


def _merge_session():
    """One ``MERGE_SUCCESS`` row from two disjoint-target branches off a shared base."""
    base_pack = get_pack("openui")
    source = 'root = Card([TextContent(":hero.title"), TextContent(":hero.body")], "clear")'
    state = OperatorStateV1.from_source(base_pack, source)
    root_branch = branch_fingerprint(state.state_digest, _sha("ablation-merge-root"))
    descriptors = tuple(
        ReferenceDescriptorV1(ref_kind=RefKind.NODE, semantic_fingerprint=_sha(name), value_type=f"openui.{name}")
        for name in ("title", "body")
    )
    root_table = build_reference_table(
        request_id="ablation-merge-request",
        state_digest=state.state_digest,
        branch_digest=root_branch,
        descriptors=descriptors,
        seed=1,
    )
    base = ConversationStateNodeV1(
        parent_state_id=None, branch_digest=root_branch, state=state, reference_table=root_table
    )
    authorities: dict[str, tuple] = {}

    def make_branch(name: str, target_name: str, replacement: str):
        branch = branch_fingerprint(state.state_digest, _sha(f"ablation-merge-{name}"))
        table = clone_reference_table_for_branch(root_table, branch_digest=branch, seed=len(authorities) + 3)
        input_node = ConversationStateNodeV1(
            parent_state_id=base.state_id, branch_digest=branch, state=state, reference_table=table
        )
        target = next(
            entry.ref for entry in table.entries if entry.descriptor.value_type == f"openui.{target_name}"
        )
        before = f":hero.{target_name}"

        def execute(inner_state: OperatorStateV1, _arguments: tuple) -> OperatorMutationV1:
            delta = EffectDeltaV1(
                kind=EffectDeltaKind.PROPERTY, target=target, before="before", after="after"
            )
            return OperatorMutationV1(
                source=inner_state.source.replace(before, replacement),
                effect=ActionEffectV1(
                    property_deltas=(delta,), compiler_coverage=CompilerCoverage.EXACT
                ),
            )

        declaration = AstOperatorV1(
            operator_id=f"openui.ablation_merge_{name}",
            version="v1",
            domain="openui.ast",
            codomain="openui.ast",
            argument_slots=(),
            preconditions=(),
            effect_signature=(EffectDeltaKind.PROPERTY,),
            locality="node",
            cost=1.0,
        )
        library = OperatorLibraryV1((RegisteredOperatorV1(declaration, execute),))
        branch_pack = replace(base_pack, operator_library=library)
        applied = library.apply(
            branch_pack,
            input_node.state,
            declaration.operator_id,
            (),
            _provenance(input_node.state, request_id="ablation-merge-request"),
        )
        assert applied.succeeded and applied.state is not None
        output_table = build_reference_table(
            request_id=table.request_id,
            state_digest=applied.state.state_digest,
            branch_digest=branch,
            descriptors=tuple(entry.descriptor for entry in table.entries),
            seed=len(authorities) + 13,
        )
        output_node = ConversationStateNodeV1(
            parent_state_id=input_node.state_id, branch_digest=branch, state=applied.state, reference_table=output_table
        )
        authorities[input_node.state_id] = (branch_pack, library)
        return BranchEditV1(input_node, output_node, applied.application)

    left = make_branch("left", "title", ":hero.heading")
    right = make_branch("right", "body", ":hero.copy")

    def resolve(node: ConversationStateNodeV1):
        return authorities[node.state_id]

    def rebuild_merged_table(_pack: object, merged_state: OperatorStateV1, branch_digest: str, seed: int):
        merged_descriptors = tuple(
            ReferenceDescriptorV1(
                ref_kind=RefKind.NODE,
                semantic_fingerprint=_sha(f"ablation-canonical:{marker}"),
                value_type=f"openui.{marker.removeprefix(':hero.')}",
            )
            for marker in (":hero.heading", ":hero.copy")
            if marker in merged_state.source
        )
        return build_reference_table(
            request_id="ablation-merge-request",
            state_digest=merged_state.state_digest,
            branch_digest=branch_digest,
            descriptors=merged_descriptors,
            seed=seed,
        )

    decision = merge_conversation_branches(
        pack=base_pack,
        base=base,
        left=left,
        right=right,
        authority_resolver=resolve,
        reference_table_builder=rebuild_merged_table,
    )
    assert decision.succeeded
    row = extract_merge_preference_row(
        left=left,
        right=right,
        decision=decision,
        authority_resolver=resolve,
        provenance_for=lambda merge_state: _provenance(merge_state, request_id="ablation-merge-request"),
    )
    assert row is not None
    return row, left, right


# --------------------------------------------------------------------------- #
# Session corpus.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ReplayPreferenceSessionV1:
    """One conversation "session" -- rows plus the history receipts behind them.

    ``group_id`` is the unit ``split_for_group`` partitions on: every row in
    one session shares one split (the issue's adversarial control,
    "conversation variants stay in one split").
    """

    group_id: str
    split: Split
    report: OperatorEventMemoryReportV1
    state_lookup: dict
    receipts: tuple[OperatorTurnReceiptV1, ...]


#: Chain lengths exercised by the rollback-chain sessions -- directly the
#: issue's own matrix ("undo all/partial ... 1/2/4/8/16 turn depth").
ROLLBACK_CHAIN_LENGTHS: tuple[int, ...] = TURN_DEPTHS


def synthesize_bounded_session_corpus() -> tuple[ReplayPreferenceSessionV1, ...]:
    """A small, deterministic, synthetic corpus covering all 7 named relations.

    Fixture-scale wiring evidence only -- never real user telemetry. Turn-
    depth variation is real (and load-bearing) for the rollback-chain
    sessions (``EDIT_THEN_UNDO``/``PARTIAL_ROLLBACK``/``UNDO_THEN_REDO``) and,
    as of this slice, for ``CHECKOUT_ANOTHER_STATE``/``FORK_THEN_CHOOSE_ONE_
    BRANCH`` (one padded session per ``ROLLBACK_CHAIN_LENGTHS`` entry, via
    ``_checkout_and_fork_trace(pad_length=...)``) and ``PRONOUN_FOCUS_
    FOLLOWUP`` (same ladder, via ``_pronoun_trace(pad_length=...)``).
    ``MERGE_SUCCESS`` still gets exactly one unpadded session -- honestly
    documented as carrying no turn-depth variation, since no trace object
    exists for a merge decision at all (see ``_merge_session`` and
    ``replay_preference.py``'s own module docstring), so there is no receipt
    sequence to pad in the first place.
    """
    sessions: list[ReplayPreferenceSessionV1] = []

    for chain_length in ROLLBACK_CHAIN_LENGTHS:
        trace, pack, library = _rollback_chain_trace(chain_length)
        report = extract_replay_preference_rows(trace, pack=pack, library=library, provenance_for=_provenance)
        group_id = f"rollback_chain_{chain_length}"
        sessions.append(
            ReplayPreferenceSessionV1(
                group_id=group_id,
                split=split_for_group(group_id),
                report=report,
                state_lookup=state_lookup_from_trace(trace),
                receipts=receipts_from_trace(trace),
            )
        )

    for pad_length in ROLLBACK_CHAIN_LENGTHS:
        checkout_fork_trace, cf_pack, cf_library = _checkout_and_fork_trace(pad_length=pad_length)
        checkout_fork_report = extract_replay_preference_rows(
            checkout_fork_trace, pack=cf_pack, library=cf_library, provenance_for=_provenance
        )
        group_id = f"checkout_and_fork_{pad_length}"
        sessions.append(
            ReplayPreferenceSessionV1(
                group_id=group_id,
                split=split_for_group(group_id),
                report=checkout_fork_report,
                state_lookup=state_lookup_from_trace(checkout_fork_trace),
                receipts=receipts_from_trace(checkout_fork_trace),
            )
        )

    for pad_length in ROLLBACK_CHAIN_LENGTHS:
        pronoun_trace, pn_pack, pn_library = _pronoun_trace(pad_length=pad_length)
        pronoun_report = extract_replay_preference_rows(
            pronoun_trace, pack=pn_pack, library=pn_library, provenance_for=_provenance
        )
        group_id = f"pronoun_focus_{pad_length}"
        sessions.append(
            ReplayPreferenceSessionV1(
                group_id=group_id,
                split=split_for_group(group_id),
                report=pronoun_report,
                state_lookup=state_lookup_from_trace(pronoun_trace),
                receipts=receipts_from_trace(pronoun_trace),
            )
        )

    merge_row, left, right = _merge_session()
    merge_report = OperatorEventMemoryReportV1(
        rows=(merge_row,), version_stamp=build_version_stamp("dsl.operators.replay_preference")
    )
    # No trace object exists for a merge decision (see replay_preference.py's
    # own module docstring); the only prior receipts available are the two
    # branch edits themselves, so ancestry walks saturate at depth 1.
    merge_receipts = (
        OperatorTurnReceiptV1(
            turn_index=0,
            action_kind="operator",
            input_state_id=left.input_node.state_id,
            output_state_id=left.output_node.state_id,
        ),
    )
    merge_state_lookup = {
        left.input_node.state_id: left.input_node,
        left.output_node.state_id: left.output_node,
        right.output_node.state_id: right.output_node,
    }
    sessions.append(
        ReplayPreferenceSessionV1(
            group_id="merge_success",
            split=split_for_group("merge_success"),
            report=merge_report,
            state_lookup=merge_state_lookup,
            receipts=merge_receipts,
        )
    )

    return tuple(sessions)


# --------------------------------------------------------------------------- #
# Tiny two-feature pairwise scorer (deliberately NOT a DSH3 head/checkpoint).
# --------------------------------------------------------------------------- #
def _features(action: str, window: ContextViewWindowV1) -> tuple[float, float]:
    """``(history_repeat_score, is_history_control)`` -- both structural, both bounded."""
    kinds = window.visible_action_kinds()
    kind = action_kind_of(action)
    repeat = (kinds.count(kind) / len(kinds)) if kinds else 0.0
    is_control = 0.0 if kind == "operator" else 1.0
    return (repeat, is_control)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _train_pairwise_linear_scorer(
    pairs: Sequence[tuple[tuple[float, float], tuple[float, float]]], *, steps: int, lr: float
) -> tuple[float, float]:
    """Full-batch gradient descent on pairwise logistic loss; two weights, no bias.

    Bias cancels in a chosen-minus-rejected margin computed from the same
    context window, so it carries no information here and is omitted.
    """
    w0 = w1 = 0.0
    n = len(pairs)
    if n == 0:
        return (0.0, 0.0)
    for _ in range(steps):
        g0 = g1 = 0.0
        for chosen, rejected in pairs:
            d0 = chosen[0] - rejected[0]
            d1 = chosen[1] - rejected[1]
            margin = w0 * d0 + w1 * d1
            grad_coeff = _sigmoid(margin) - 1.0  # d(-log sigmoid(margin))/d(margin)
            g0 += grad_coeff * d0
            g1 += grad_coeff * d1
        w0 -= lr * g0 / n
        w1 -= lr * g1 / n
    return (w0, w1)


def _score(action: str, window: ContextViewWindowV1, weights: tuple[float, float]) -> float:
    f0, f1 = _features(action, window)
    return weights[0] * f0 + weights[1] * f1


def _view_depth_pairs(sessions: Sequence[ReplayPreferenceSessionV1], *, view: ContextView, depth: int):
    """Chosen/rejected feature pairs and their (row, window) pairs for one (view, depth) cell.

    A plain module-level function, not a closure re-defined per loop
    iteration, so ``view``/``depth`` are always explicit arguments rather
    than captured loop variables (Ruff B023).
    """
    pairs = []
    windows = []
    for session in sessions:
        for row in session.report.rows:
            window = build_context_view_window(
                state_lookup=session.state_lookup,
                receipts=session.receipts,
                decision_state_id=row.input_state_id,
                chosen_output_state_id=row.chosen_output_state_id,
                view=view,
                turn_depth=depth,
            )
            pairs.append((_features(row.chosen_action, window), _features(row.rejected_action, window)))
            windows.append((row, window))
    return pairs, windows


# --------------------------------------------------------------------------- #
# Comparison report.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ContextViewCellMetricsV1:
    view: ContextView
    turn_depth: int
    train_pair_count: int
    held_out_pair_count: int
    pairwise_preference_accuracy: float
    mean_calibration_error: float
    accuracy_by_relation: dict
    weights: tuple[float, float]
    schema: str = "replay_preference_context_view_cell_metrics/v1"

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "view": self.view.value,
            "turn_depth": self.turn_depth,
            "train_pair_count": self.train_pair_count,
            "held_out_pair_count": self.held_out_pair_count,
            "pairwise_preference_accuracy": self.pairwise_preference_accuracy,
            "mean_calibration_error": self.mean_calibration_error,
            "accuracy_by_relation": dict(self.accuracy_by_relation),
            "weights": list(self.weights),
        }


@dataclass(frozen=True)
class ReplayPreferenceContextViewComparisonReportV1:
    """Fixture-scale matched context-view comparison. Wiring evidence only.

    See the module docstring for exactly what is, and is not, measured.
    """

    cells: tuple[ContextViewCellMetricsV1, ...]
    session_count: int
    train_session_count: int
    held_out_session_count: int
    row_count: int
    undo_family_rate: float
    held_out_benefit: dict
    unintended_mutation_note: str
    trace_replay_note: str
    cap_retention_note: str
    version_stamp: dict
    schema: str = "replay_preference_context_view_comparison_report/v1"

    def to_dict(self) -> dict:
        return {
            "schema": self.schema,
            "cells": [cell.to_dict() for cell in self.cells],
            "session_count": self.session_count,
            "train_session_count": self.train_session_count,
            "held_out_session_count": self.held_out_session_count,
            "row_count": self.row_count,
            "undo_family_rate": self.undo_family_rate,
            "held_out_benefit": dict(self.held_out_benefit),
            "unintended_mutation_note": self.unintended_mutation_note,
            "trace_replay_note": self.trace_replay_note,
            "cap_retention_note": self.cap_retention_note,
            "version_stamp": self.version_stamp,
        }


_UNDO_FAMILY = frozenset(
    {
        ReplayPreferenceRelation.EDIT_THEN_UNDO.value,
        ReplayPreferenceRelation.UNDO_THEN_REDO.value,
        ReplayPreferenceRelation.PARTIAL_ROLLBACK.value,
    }
)

_UNINTENDED_MUTATION_NOTE = (
    "structurally inapplicable: this harness only scores existing legal "
    "candidates via _score() and never calls OperatorLibraryV1.apply, so no "
    "mutation is possible by construction; zero is not a measured rate."
)
_TRACE_REPLAY_NOTE = (
    "inherited, not independently re-measured: extract_replay_preference_rows/"
    "extract_merge_preference_row are unchanged by this slice, and their own "
    "acceptance-criterion tests in tests/test_dsl/test_replay_preference.py "
    "already prove every relation this synthetic corpus produces "
    "independently replays to its recorded chosen_output_state_id."
)
_CAP_RETENTION_NOTE = (
    "out of scope for this slice: CAP0/CAP1/CAP2 retention requires the full "
    "CAP-gated eval suite (src/slm_training/evals/cap2_operator.py and "
    "friends) integrated with a trained policy checkpoint; this module trains "
    "a tiny two-feature diagnostic linear scorer, not a certified checkpoint."
)


def train_replay_preference_context_view_variants(
    sessions: Sequence[ReplayPreferenceSessionV1],
    *,
    views: Sequence[ContextView] = tuple(ContextView),
    turn_depths: Sequence[int] = TURN_DEPTHS,
    steps: int = 200,
    lr: float = 0.5,
    max_wall_seconds: float = MAX_HARNESS_WALL_SECONDS,
) -> ReplayPreferenceContextViewComparisonReportV1:
    """Train and compare one tiny pairwise scorer per ``(view, turn_depth)`` cell.

    Fails closed (raises) rather than silently reporting on an empty split,
    and enforces the repository's hard run cap (``slm_training.levers.
    MAX_HARNESS_WALL_SECONDS``) even though this harness completes in a small
    fraction of it.
    """
    start = time.monotonic()
    train_sessions = [session for session in sessions if session.split == "train"]
    held_out_sessions = [session for session in sessions if session.split == "held_out"]
    if not train_sessions or not held_out_sessions:
        raise ValueError(
            "train_replay_preference_context_view_variants requires at least "
            "one session in each split; conversation variants must stay in "
            "one split, so the corpus cannot be re-split to force this"
        )

    all_rows = [
        (session, row) for session in sessions for row in session.report.rows
    ]
    undo_family_rate = (
        sum(1 for _, row in all_rows if row.semantic_relation.value in _UNDO_FAMILY) / len(all_rows)
        if all_rows
        else 0.0
    )

    cells: list[ContextViewCellMetricsV1] = []
    baseline_key = (ContextView.CURRENT_STATE_ONLY, turn_depths[0])
    baseline_accuracy: float | None = None
    best_non_baseline: tuple[float, ContextView, int] | None = None

    for view, depth in product(views, turn_depths):
        if time.monotonic() - start > max_wall_seconds:
            raise TimeoutError(
                "replay-preference context-view ablation exceeded "
                f"MAX_HARNESS_WALL_SECONDS ({max_wall_seconds}s); a timed-out "
                "run is never evidence (AGENTS.md hard run cap)"
            )

        train_pairs, _train_windows = _view_depth_pairs(train_sessions, view=view, depth=depth)
        held_out_pairs, held_out_windows = _view_depth_pairs(held_out_sessions, view=view, depth=depth)

        weights = _train_pairwise_linear_scorer(train_pairs, steps=steps, lr=lr)

        correct = 0.0
        brier_sum = 0.0
        relation_correct: dict = {}
        relation_total: dict = {}
        for row, window in held_out_windows:
            chosen_score = _score(row.chosen_action, window, weights)
            rejected_score = _score(row.rejected_action, window, weights)
            margin = chosen_score - rejected_score
            outcome = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
            correct += outcome
            brier_sum += (1.0 - _sigmoid(margin)) ** 2
            relation = row.semantic_relation.value
            relation_correct[relation] = relation_correct.get(relation, 0.0) + outcome
            relation_total[relation] = relation_total.get(relation, 0) + 1

        held_out_count = len(held_out_windows)
        accuracy = correct / held_out_count if held_out_count else float("nan")
        calibration = brier_sum / held_out_count if held_out_count else float("nan")
        accuracy_by_relation = {
            relation: relation_correct[relation] / relation_total[relation] for relation in relation_total
        }

        cells.append(
            ContextViewCellMetricsV1(
                view=view,
                turn_depth=depth,
                train_pair_count=len(train_pairs),
                held_out_pair_count=held_out_count,
                pairwise_preference_accuracy=accuracy,
                mean_calibration_error=calibration,
                accuracy_by_relation=accuracy_by_relation,
                weights=weights,
            )
        )

        if (view, depth) == baseline_key:
            baseline_accuracy = accuracy
        elif view is not ContextView.CURRENT_STATE_ONLY and held_out_count:
            if best_non_baseline is None or accuracy > best_non_baseline[0]:
                best_non_baseline = (accuracy, view, depth)

    if baseline_accuracy is None or math.isnan(baseline_accuracy):
        raise ValueError(
            "baseline (current_state_only) cell was not computed or had no "
            "held-out pairs; a NaN baseline can never support a benefit verdict"
        )

    if best_non_baseline is None:
        held_out_benefit = {
            "verdict": "no_non_baseline_held_out_data",
            "baseline_accuracy": baseline_accuracy,
            "notes": (
                "no non-CURRENT_STATE_ONLY cell had any held-out pair; cannot "
                "compare, not a benefit claim of any kind."
            ),
        }
    else:
        best_accuracy, best_view, best_depth = best_non_baseline
        verdict = "benefit_observed_fixture_scale" if best_accuracy > baseline_accuracy else "no_benefit_fixture_scale"
        held_out_benefit = {
            "verdict": verdict,
            "baseline_accuracy": baseline_accuracy,
            "best_view": best_view.value,
            "best_turn_depth": best_depth,
            "best_accuracy": best_accuracy,
            "notes": (
                "fixture-scale synthetic corpus (a few dozen rows across "
                f"{len(sessions)} sessions); not a powered or certified "
                "held-out-benefit claim either way -- see the falsification/"
                "stop rule in the SLM-418 issue text and "
                "docs/design/dsh5-10-replay-preference-rows.md."
            ),
        }

    return ReplayPreferenceContextViewComparisonReportV1(
        cells=tuple(cells),
        session_count=len(sessions),
        train_session_count=len(train_sessions),
        held_out_session_count=len(held_out_sessions),
        row_count=len(all_rows),
        undo_family_rate=undo_family_rate,
        held_out_benefit=held_out_benefit,
        unintended_mutation_note=_UNINTENDED_MUTATION_NOTE,
        trace_replay_note=_TRACE_REPLAY_NOTE,
        cap_retention_note=_CAP_RETENTION_NOTE,
        version_stamp=build_version_stamp(
            "harness.preference.replay_preference_context_view_variants"
        ),
    )
