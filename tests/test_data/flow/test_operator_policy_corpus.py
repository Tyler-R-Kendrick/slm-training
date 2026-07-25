"""Tests for DSH3-24 operator-policy corpus rows built from collapse negatives."""

from __future__ import annotations

import hashlib
from dataclasses import replace

from slm_training.data.flow.operator_policy_corpus import (
    RowRejectionKind,
    build_operator_policy_corpus,
    build_operator_policy_rows,
)
from slm_training.dsl.operators import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    CompilerCoverage,
    HardNegativeOutcome,
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
from slm_training.models.operator_policy_view import validate_no_forbidden_fields

SOURCE = 'root = TextContent(":hero.title")'


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _provenance(state: OperatorStateV1) -> ApplicationProvenanceV1:
    return ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.policy_corpus_fixture",
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


def _trace_and_collapse(library: OperatorLibraryV1, base_pack):
    """Build a 2-turn trace over ``library``'s two operators and collapse it."""
    pack = replace(base_pack, operator_library=library)
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

    declarations = library.declarations
    trace = append(trace, declarations[0].operator_id, seed=2)
    trace = append(trace, declarations[1].operator_id, seed=3)

    def authority_resolver(_node):
        return pack, library

    decision = collapse_conversation_trace(
        pack=pack, authority_resolver=authority_resolver, trace=trace
    )
    assert decision.rejection is None, decision.rejection
    return pack, library, trace, decision.collapse, authority_resolver


def _different_result_fixture():
    """Two unconditional overwrites: both orders succeed but disagree, so the
    swap is a genuine DIFFERENT_RESULT hard negative."""

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
    return _trace_and_collapse(library, get_pack("openui"))


def _conflict_fixture():
    """B requires A's effect; swapping the order makes B fail on the root
    state, a genuine CONFLICT hard negative."""

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
    return _trace_and_collapse(library, get_pack("openui"))


def test_different_result_hard_negative_is_reprojected_onto_step_zero() -> None:
    pack, library, trace, collapse, authority_resolver = _different_result_fixture()
    assert len(collapse.hard_negatives) == 1
    assert collapse.hard_negatives[0].outcome is HardNegativeOutcome.DIFFERENT_RESULT

    rows, rejected = build_operator_policy_rows(
        trace=trace, collapse=collapse, authority_resolver=authority_resolver
    )
    assert rejected == ()
    assert len(rows) == 2
    assert rows[0].accepted_operator_id == "openui.set_x"
    assert rows[1].accepted_operator_id == "openui.set_y"
    assert rows[0].hard_negative is not None
    assert rows[0].hard_negative.outcome is HardNegativeOutcome.DIFFERENT_RESULT
    assert rows[0].hard_negative.alternate_operator_id == "openui.set_y"
    assert rows[0].hard_negative.alternate_action_row is not None
    assert rows[1].hard_negative is None


def test_conflict_hard_negative_is_reprojected_onto_step_zero() -> None:
    pack, library, trace, collapse, authority_resolver = _conflict_fixture()
    assert len(collapse.hard_negatives) == 1
    assert collapse.hard_negatives[0].outcome is HardNegativeOutcome.CONFLICT

    rows, rejected = build_operator_policy_rows(
        trace=trace, collapse=collapse, authority_resolver=authority_resolver
    )
    assert rejected == ()
    assert rows[0].hard_negative is not None
    assert rows[0].hard_negative.outcome is HardNegativeOutcome.CONFLICT
    assert rows[0].hard_negative.conflict_code is not None
    assert rows[0].hard_negative.observed_final_state_digest is None
    assert rows[1].hard_negative is None


def test_rows_carry_only_allowlisted_policy_input_fields() -> None:
    _pack, _library, trace, collapse, authority_resolver = _different_result_fixture()
    rows, _rejected = build_operator_policy_rows(
        trace=trace, collapse=collapse, authority_resolver=authority_resolver
    )
    for row in rows:
        validate_no_forbidden_fields(row.policy_input)
        assert row.to_dict()["policy_input_digest"] == row.policy_input_digest


def test_accepted_action_absent_from_a_re_enumerated_library_is_rejected() -> None:
    _pack, _library, trace, collapse, _authority_resolver = _different_result_fixture()

    def execute_set_x(_state, _arguments):
        return OperatorMutationV1(
            source='root = TextContent(":set.x")',
            effect=ActionEffectV1(compiler_coverage=CompilerCoverage.EXACT),
        )

    def execute_set_y_always_rejects(_state, _arguments):
        raise OperatorRejectedError("fixture.now_unconditionally_rejected")

    # Same registry/declarations as the original (so step 0's re-enumeration
    # reproduces byte-identical evidence) but "openui.set_y" now always
    # rejects: re-enumeration at step 1 can never find the recorded
    # application live, so it must be rejected rather than forced in from
    # the collapse's own recorded membership.
    stub_library = OperatorLibraryV1(
        (
            RegisteredOperatorV1(_declaration("openui.set_x"), execute_set_x),
            RegisteredOperatorV1(
                _declaration("openui.set_y"), execute_set_y_always_rejects
            ),
        )
    )
    stub_pack = replace(get_pack("openui"), operator_library=stub_library)

    def authority_resolver(_node):
        return stub_pack, stub_library

    rows, rejected = build_operator_policy_rows(
        trace=trace, collapse=collapse, authority_resolver=authority_resolver
    )
    assert len(rows) == 1
    assert rows[0].accepted_operator_id == "openui.set_x"
    assert len(rejected) == 1
    assert rejected[0]["reason"] == RowRejectionKind.ACCEPTED_ACTION_NOT_LIVE
    assert rejected[0]["step_index"] == 1


def test_corpus_quality_report_aggregates_hard_negatives_and_rejections() -> None:
    _pack, _library, trace, collapse, authority_resolver = _different_result_fixture()
    rows, report = build_operator_policy_corpus(
        trace=trace, collapse=collapse, authority_resolver=authority_resolver
    )
    assert report.total_steps == 2
    assert report.accepted_rows == len(rows) == 2
    assert report.rejected_rows == 0
    assert report.hard_negative_counts == {"different_result": 1}
    assert report.positive_only_rows == 1
    payload = report.to_dict()
    assert payload["schema"] == "operator_policy_corpus_quality_report/v1"
