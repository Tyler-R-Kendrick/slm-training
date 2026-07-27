"""Tests for VAR3-03 (SLM-433) real-corpus turn-disposition rows."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from slm_training.data.flow.turn_disposition_corpus import (
    assert_leakage_safe_split,
    build_turn_disposition_rows,
    corpus_action_frequency,
)
from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    CompilerCoverage,
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorRejectedError,
    OperatorStateV1,
    RegisteredOperatorV1,
    append_operator_turn,
    branch_fingerprint,
    build_reference_table,
    collapse_conversation_trace,
    create_conversation_trace,
)
from slm_training.dsl.pack import get_pack
from slm_training.models.turn_disposition_head import TurnDispositionLabel

SOURCE = 'root = TextContent(":hero.title")'


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _provenance(state: OperatorStateV1) -> ApplicationProvenanceV1:
    return ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.disposition_corpus_fixture",
        compiler_version="v1",
        source_artifact_digest=_sha(state.source),
        request_id="request-1",
    )


def _table(state: OperatorStateV1, branch: str, *, seed: int):
    return build_reference_table(
        request_id="request-1",
        state_digest=state.state_digest,
        branch_digest=branch,
        descriptors=(),
        seed=seed,
    )


def _declaration(operator_id: str) -> AstOperatorV1:
    return AstOperatorV1(
        operator_id=operator_id,
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(),
        preconditions=(),
        effect_signature=(),
        locality="node",
        cost=1.0,
    )


def _trace_and_collapse(library: OperatorLibraryV1, first_operator_id: str):
    """Build a 2-turn trace applying ``first_operator_id`` then whatever
    remains live, and collapse it -- mirrors
    ``test_data/flow/test_operator_policy_corpus.py``'s own fixture shape."""
    pack = replace(get_pack("openui"), operator_library=library)
    root_state = OperatorStateV1.from_source(pack, SOURCE)
    branch = branch_fingerprint(root_state.state_digest, _sha("root-branch"))
    root_table = _table(root_state, branch, seed=1)
    trace = create_conversation_trace(
        pack=pack,
        root_state=root_state,
        root_reference_table=root_table,
        provenance=_provenance(root_state),
    )

    def append(trace, operator_id, *, seed):
        state = trace.current.state
        result = library.apply(pack, state, operator_id, (), _provenance(state))
        assert result.succeeded, result.application.rejection
        output_table = _table(result.state, trace.current.branch_digest, seed=seed)
        return append_operator_turn(
            trace,
            pack=pack,
            library=library,
            application=result.application,
            output_reference_table=output_table,
        )

    other_operator_id = next(
        d.operator_id for d in library.declarations if d.operator_id != first_operator_id
    )
    trace = append(trace, first_operator_id, seed=2)
    trace = append(trace, other_operator_id, seed=3)

    def authority_resolver(_node):
        return pack, library

    decision = collapse_conversation_trace(
        pack=pack, authority_resolver=authority_resolver, trace=trace
    )
    assert decision.rejection is None, decision.rejection
    return trace, decision.collapse, authority_resolver


def _ambiguous_fixture():
    """Two operators, both unconditionally legal at every state -- every
    step has >= 2 real candidates, exercising the scored clarify/emit path."""

    def execute_set_x(_state, _arguments):
        return OperatorMutationV1(
            source='root = TextContent(":set.x")',
            effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
        )

    def execute_set_y(_state, _arguments):
        return OperatorMutationV1(
            source='root = TextContent(":set.y")',
            effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
        )

    library = OperatorLibraryV1(
        (
            RegisteredOperatorV1(_declaration("openui.set_x"), execute_set_x),
            RegisteredOperatorV1(_declaration("openui.set_y"), execute_set_y),
        )
    )
    return _trace_and_collapse(library, "openui.set_x")


def _forced_singleton_fixture():
    """B requires A's effect, so exactly one operator is ever live at each
    step -- every row is a forced-singleton emit."""

    def execute_a(state, _arguments):
        if ":hero.title" not in state.source:
            raise OperatorRejectedError("fixture.a_precondition_failed")
        return OperatorMutationV1(
            source=state.source.replace(":hero.title", ":hero.body"),
            effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
        )

    def execute_b(state, _arguments):
        if ":hero.body" not in state.source:
            raise OperatorRejectedError("fixture.b_precondition_failed")
        return OperatorMutationV1(
            source=state.source.replace(":hero.body", ":hero.footer"),
            effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
        )

    library = OperatorLibraryV1(
        (
            RegisteredOperatorV1(_declaration("openui.set_a"), execute_a),
            RegisteredOperatorV1(_declaration("openui.set_b"), execute_b),
        )
    )
    return _trace_and_collapse(library, "openui.set_a")


def test_forced_singleton_rows_are_labeled_emit_and_derived_from_the_bypass() -> None:
    trace, collapse, authority_resolver = _forced_singleton_fixture()
    rows, rejected = build_turn_disposition_rows(
        trace=trace,
        collapse=collapse,
        authority_resolver=authority_resolver,
        action_frequency={},
    )
    assert rejected == ()
    assert len(rows) == 2
    for row in rows:
        assert row.target.label is TurnDispositionLabel.EMIT
        assert row.target.forced_singleton is True
        assert row.candidate_count == 1
        assert row.emit_correct is True
        assert row.top_candidate_correct is True


def test_ambiguous_rows_use_real_corpus_frequency_not_synthetic_scores() -> None:
    trace, collapse, authority_resolver = _ambiguous_fixture()
    # Step 0 has two live candidates (set_x, set_y); the accepted one is
    # set_x. A real, corpus-derived frequency table that favors the actually
    # accepted action's serialized form should push the margin to EMIT
    # (agreeing with history) rather than CLARIFY, without ever injecting a
    # synthetic per-example score.
    rows, rejected = build_turn_disposition_rows(
        trace=trace,
        collapse=collapse,
        authority_resolver=authority_resolver,
        action_frequency={},
    )
    assert rejected == ()
    assert len(rows) == 2
    step_zero = rows[0]
    assert step_zero.candidate_count == 2
    # Tied (zero) real frequency for both candidates -> zero margin -> below
    # the registered clarify threshold -> the honest, expected CLARIFY.
    assert step_zero.target.label is TurnDispositionLabel.CLARIFY
    assert step_zero.target.forced_singleton is False
    assert step_zero.emit_correct is None

    # A real corpus statistic built from several *other* traces that
    # historically preferred step zero's accepted action: not a synthetic
    # per-example score, just more accumulated real evidence.
    frequency = corpus_action_frequency(
        {
            "other-trace-1": (step_zero.accepted_action_serialized,),
            "other-trace-2": (step_zero.accepted_action_serialized,),
            "other-trace-3": (step_zero.accepted_action_serialized,),
        }
    )
    skewed_rows, _ = build_turn_disposition_rows(
        trace=trace,
        collapse=collapse,
        authority_resolver=authority_resolver,
        action_frequency=frequency,
    )
    assert skewed_rows[0].target.label is TurnDispositionLabel.EMIT
    assert skewed_rows[0].emit_correct is True


def test_accepted_action_absent_from_a_re_enumerated_library_is_rejected_not_mislabeled() -> None:
    """A row whose recorded action cannot be re-proven live is rejected
    outright -- never coerced into a disposition target. This is the only
    way a fresh legal set could ever disagree with history in this corpus,
    and it is handled by rejection, so ``out_of_scope`` (which would require
    an *empty* legal set) can never be silently substituted for it either."""
    trace, collapse, _authority_resolver = _forced_singleton_fixture()

    def execute_a(_state, _arguments):
        raise OperatorRejectedError("fixture.now_unconditionally_rejected")

    def execute_b(state, _arguments):
        if ":hero.body" not in state.source:
            raise OperatorRejectedError("fixture.b_precondition_failed")
        return OperatorMutationV1(
            source=state.source.replace(":hero.body", ":hero.footer"),
            effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
        )

    stub_library = OperatorLibraryV1(
        (
            RegisteredOperatorV1(_declaration("openui.set_a"), execute_a),
            RegisteredOperatorV1(_declaration("openui.set_b"), execute_b),
        )
    )
    stub_pack = replace(get_pack("openui"), operator_library=stub_library)

    def authority_resolver(_node):
        return stub_pack, stub_library

    rows, rejected = build_turn_disposition_rows(
        trace=trace,
        collapse=collapse,
        authority_resolver=authority_resolver,
        action_frequency={},
    )
    # Step 0's recorded "openui.set_a" can never be re-proven live against
    # this stub (it now unconditionally rejects) -- rejected outright, not
    # coerced into any disposition target. Step 1's "openui.set_b" is
    # unaffected (its precondition still holds at that historical state) and
    # is still built normally.
    assert len(rows) == 1
    assert rows[0].step_index == 1
    assert len(rejected) == 1
    assert rejected[0]["reason"] == "accepted_action_not_live"
    assert rejected[0]["step_index"] == 0


def test_leakage_safe_split_rejects_a_trace_split_across_train_and_dev() -> None:
    trace, collapse, authority_resolver = _forced_singleton_fixture()
    rows, _rejected = build_turn_disposition_rows(
        trace=trace,
        collapse=collapse,
        authority_resolver=authority_resolver,
        action_frequency={},
        split="train",
    )
    assert_leakage_safe_split(rows)  # single split: fine

    mixed = (*rows, replace(rows[0], split="dev"))
    with pytest.raises(ValueError, match="more than one split"):
        assert_leakage_safe_split(mixed)


def test_corpus_action_frequency_aggregates_across_traces() -> None:
    frequency = corpus_action_frequency(
        {
            "trace-a": ("OPERATOR foo", "OPERATOR bar"),
            "trace-b": ("OPERATOR foo",),
        }
    )
    assert frequency["OPERATOR foo"] == 2
    assert frequency["OPERATOR bar"] == 1
    assert frequency["OPERATOR absent"] == 0


def test_rows_carry_only_allowlisted_policy_input_fields() -> None:
    trace, collapse, authority_resolver = _ambiguous_fixture()
    rows, _rejected = build_turn_disposition_rows(
        trace=trace,
        collapse=collapse,
        authority_resolver=authority_resolver,
        action_frequency={},
    )
    for row in rows:
        assert row.to_dict()["policy_input_digest"] == row.policy_input_digest
