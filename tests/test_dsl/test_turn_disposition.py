from __future__ import annotations

import hashlib

import pytest

from slm_training.dsl.operators.contracts import ApplicationProvenanceV1, _fingerprint
from slm_training.dsl.operators.conversation import (
    ConversationOperation,
    ConversationTraceError,
    OperatorLibraryV1,
    OperatorStateV1,
    TurnArtifactV1,
    branch_fingerprint,
    build_reference_table,
    copy_conversation_state,
    create_conversation_trace,
    replay_conversation_trace,
)
from slm_training.dsl.operators.legal_set import (
    LegalOperatorActionV1,
    LegalSetCoverage,
    OperatorLegalEntryV1,
    OperatorLegalSetV1,
    OperatorSupportVerdict,
    serialize_operator_action,
)
from slm_training.dsl.operators.turn_disposition import (
    TurnDispositionKind,
    TurnDispositionV1,
    append_answer_turn,
    append_clarify_turn,
    derive_turn_disposition,
)
from slm_training.dsl.pack import get_pack
from slm_training.levers import (
    TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD,
    TURN_DISPOSITION_CLARIFY_TOP_K,
)

SOURCE = 'root = TextContent(":hero.title")'


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest(label: str) -> str:
    return _fingerprint({"schema": "test_turn_disposition/v1", "label": label})


def _legal_set(
    *,
    ordinary: tuple[str, ...] = (),
    entries: tuple[OperatorLegalEntryV1, ...] = (),
    coverage: LegalSetCoverage = LegalSetCoverage.COMPLETE,
) -> OperatorLegalSetV1:
    return OperatorLegalSetV1(
        state_fingerprint=_digest("state"),
        reference_table_fingerprint=_digest("refs"),
        registry_fingerprint=_digest("registry"),
        entries=entries,
        ordinary_nonoperator_actions=ordinary,
        coverage=coverage,
        max_combinations_per_operator=10,
        max_rejection_samples_per_operator=0,
    )


def _partial_single_entry() -> OperatorLegalEntryV1:
    operator_id = "openui.turn_disposition_fixture"
    operator_fingerprint = _digest("operator")
    serialized = serialize_operator_action(operator_id, ())
    action = LegalOperatorActionV1(
        operator_id=operator_id,
        operator_fingerprint=operator_fingerprint,
        arguments=(),
        semantic_id=_digest("semantic"),
        application_id=_digest("application"),
        proof_fingerprint=_digest("proof"),
        proof_checks=("check",),
        serialized=serialized,
    )
    return OperatorLegalEntryV1(
        operator_id=operator_id,
        operator_fingerprint=operator_fingerprint,
        argument_domains=(),
        legal_actions=(action,),
        verdict=OperatorSupportVerdict.SUPPORTED,
        coverage=LegalSetCoverage.PARTIAL,
        evaluated_combinations=1,
        total_combinations=5,
        rejection_counts=(),
    )


def _boom() -> dict[str, float]:
    raise AssertionError("score_provider must not be called on this path")


# ---------------------------------------------------------------------------
# Pure precedence logic
# ---------------------------------------------------------------------------


def test_out_of_scope_never_consults_score_provider() -> None:
    """I1: derived from the empty legal set alone; nothing else is consulted."""
    result = derive_turn_disposition(
        legal_set=_legal_set(),
        is_state_query=True,  # even a "query" flag must not override this
        clarify_margin_threshold=TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD,
        score_provider=_boom,
    )
    assert result.kind is TurnDispositionKind.OUT_OF_SCOPE
    assert result.chosen_action is None
    assert result.clarify_candidates == ()
    assert result.margin is None


def test_answer_outranks_scoring_but_not_out_of_scope() -> None:
    result = derive_turn_disposition(
        legal_set=_legal_set(ordinary=("OPERATOR a", "OPERATOR b")),
        is_state_query=True,
        clarify_margin_threshold=TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD,
        score_provider=_boom,
    )
    assert result.kind is TurnDispositionKind.ANSWER


def test_forced_singleton_bypasses_scoring() -> None:
    legal_set = _legal_set(ordinary=("OPERATOR only",))
    result = derive_turn_disposition(
        legal_set=legal_set,
        is_state_query=False,
        clarify_margin_threshold=TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD,
        score_provider=_boom,
    )
    assert result.kind is TurnDispositionKind.EMIT
    assert result.chosen_action == "OPERATOR only"
    assert result.margin is None


def test_partial_coverage_single_candidate_is_conservatively_clarified() -> None:
    """A single PARTIAL-coverage candidate is not a proven singleton."""
    legal_set = _legal_set(entries=(_partial_single_entry(),), coverage=LegalSetCoverage.PARTIAL)
    assert legal_set.forced_action is None
    result = derive_turn_disposition(
        legal_set=legal_set,
        is_state_query=False,
        clarify_margin_threshold=TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD,
        score_provider=_boom,
    )
    assert result.kind is TurnDispositionKind.CLARIFY
    assert len(result.clarify_candidates) == 1


def test_ambiguous_margin_clarifies_with_top_k() -> None:
    legal_set = _legal_set(ordinary=("OPERATOR a", "OPERATOR b", "OPERATOR c"))
    result = derive_turn_disposition(
        legal_set=legal_set,
        is_state_query=False,
        clarify_margin_threshold=TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD,
        clarify_top_k=TURN_DISPOSITION_CLARIFY_TOP_K,
        score_provider=lambda: {
            "OPERATOR a": 0.9,
            "OPERATOR b": 0.85,
            "OPERATOR c": 0.1,
        },
    )
    assert result.kind is TurnDispositionKind.CLARIFY
    assert result.clarify_candidates[0] == "OPERATOR a"
    assert result.margin == pytest.approx(0.05)
    assert len(result.clarify_candidates) == 3


def test_confident_margin_emits_top_candidate() -> None:
    legal_set = _legal_set(ordinary=("OPERATOR a", "OPERATOR b"))
    result = derive_turn_disposition(
        legal_set=legal_set,
        is_state_query=False,
        clarify_margin_threshold=TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD,
        score_provider=lambda: {"OPERATOR a": 0.95, "OPERATOR b": 0.1},
    )
    assert result.kind is TurnDispositionKind.EMIT
    assert result.chosen_action == "OPERATOR a"
    assert result.margin == pytest.approx(0.85)


def test_missing_score_provider_raises_for_ambiguous_legal_set() -> None:
    legal_set = _legal_set(ordinary=("OPERATOR a", "OPERATOR b"))
    with pytest.raises(ValueError, match="missing_score_provider"):
        derive_turn_disposition(
            legal_set=legal_set,
            is_state_query=False,
            clarify_margin_threshold=TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD,
        )


def test_incomplete_scores_raise() -> None:
    legal_set = _legal_set(ordinary=("OPERATOR a", "OPERATOR b"))
    with pytest.raises(ValueError, match="incomplete_scores"):
        derive_turn_disposition(
            legal_set=legal_set,
            is_state_query=False,
            clarify_margin_threshold=TURN_DISPOSITION_CLARIFY_MARGIN_THRESHOLD,
            score_provider=lambda: {"OPERATOR a": 0.9},
        )


def test_invalid_thresholds_are_rejected() -> None:
    legal_set = _legal_set(ordinary=("OPERATOR a",))
    with pytest.raises(ValueError, match="clarify_top_k"):
        derive_turn_disposition(
            legal_set=legal_set,
            is_state_query=False,
            clarify_margin_threshold=0.1,
            clarify_top_k=0,
        )
    with pytest.raises(ValueError, match="clarify_margin_threshold"):
        derive_turn_disposition(
            legal_set=legal_set,
            is_state_query=False,
            clarify_margin_threshold=-0.1,
        )


# ---------------------------------------------------------------------------
# TurnDispositionV1 shape validation
# ---------------------------------------------------------------------------


def test_turn_disposition_shape_is_validated_per_kind() -> None:
    with pytest.raises(ValueError, match="out_of_scope carries no"):
        TurnDispositionV1(kind=TurnDispositionKind.OUT_OF_SCOPE, chosen_action="x")
    with pytest.raises(ValueError, match="answer carries no"):
        TurnDispositionV1(kind=TurnDispositionKind.ANSWER, chosen_action="x")
    with pytest.raises(ValueError, match="emit requires a chosen action"):
        TurnDispositionV1(kind=TurnDispositionKind.EMIT)
    with pytest.raises(ValueError, match="clarify requires at least one"):
        TurnDispositionV1(kind=TurnDispositionKind.CLARIFY)
    with pytest.raises(ValueError, match="unique"):
        TurnDispositionV1(
            kind=TurnDispositionKind.CLARIFY, clarify_candidates=("a", "a")
        )


# ---------------------------------------------------------------------------
# Conversation-trace round trip (acceptance: clarify/answer survive replay)
# ---------------------------------------------------------------------------


def _provenance(state: OperatorStateV1) -> ApplicationProvenanceV1:
    return ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.turn_disposition_fixture",
        compiler_version="v1",
        source_artifact_digest=_sha(state.source),
        request_id="request-1",
    )


def _trace_fixture():
    pack = get_pack("openui")
    state = OperatorStateV1.from_source(pack, SOURCE)
    branch = branch_fingerprint(state.state_digest, _sha("root-branch"))
    table = build_reference_table(
        request_id="request-1",
        state_digest=state.state_digest,
        branch_digest=branch,
        descriptors=(),
        seed=1,
    )
    trace = create_conversation_trace(
        pack=pack,
        root_state=state,
        root_reference_table=table,
        provenance=_provenance(state),
    )
    return pack, trace


def test_clarify_turn_is_recorded_and_survives_replay() -> None:
    pack, trace = _trace_fixture()
    provenance = _provenance(trace.current.state)
    candidates = ("OPERATOR a", "OPERATOR b")
    clarified = append_clarify_turn(trace, candidates=candidates, provenance=provenance)

    turn = clarified.turns[-1]
    assert turn.operation is ConversationOperation.COPY_STATE
    assert turn.disposition == "clarify"
    assert turn.disposition_candidates == candidates
    assert clarified.current.state == trace.current.state
    assert clarified.current_state_id != trace.current_state_id

    library = OperatorLibraryV1(())
    replayed = replay_conversation_trace(pack=pack, library=library, trace=clarified)
    assert replayed.state_id == clarified.current_state_id

    # The disposition and its candidates survive the round trip verbatim.
    payload = clarified.to_dict()["turns"][-1]
    assert payload["disposition"] == "clarify"
    assert payload["disposition_candidates"] == list(candidates)


def test_answer_turn_is_recorded_and_survives_replay() -> None:
    pack, trace = _trace_fixture()
    provenance = _provenance(trace.current.state)
    answered = append_answer_turn(trace, provenance=provenance)

    turn = answered.turns[-1]
    assert turn.operation is ConversationOperation.COPY_STATE
    assert turn.disposition == "answer"
    assert turn.disposition_candidates == ()
    assert answered.current.state == trace.current.state

    library = OperatorLibraryV1(())
    replayed = replay_conversation_trace(pack=pack, library=library, trace=answered)
    assert replayed.state_id == answered.current_state_id
    assert answered.to_dict()["turns"][-1]["disposition"] == "answer"


def test_clarify_and_answer_dispositions_require_a_self_referential_copy() -> None:
    pack, trace = _trace_fixture()
    provenance = _provenance(trace.current.state)
    with pytest.raises(ConversationTraceError, match="unrecordable turn disposition"):
        TurnArtifactV1(
            operation=ConversationOperation.COPY_STATE,
            input_state_id=trace.root_state_id,
            output_state_id=trace.root_state_id,
            provenance=provenance,
            copy_source_state_id=trace.root_state_id,
            disposition="out_of_scope",
        )
    with pytest.raises(ConversationTraceError, match="requires disposition candidates"):
        TurnArtifactV1(
            operation=ConversationOperation.COPY_STATE,
            input_state_id=trace.root_state_id,
            output_state_id=trace.root_state_id,
            provenance=provenance,
            copy_source_state_id=trace.root_state_id,
            disposition="clarify",
        )
    with pytest.raises(ConversationTraceError, match="mutating turn"):
        TurnArtifactV1(
            operation=ConversationOperation.COPY_STATE,
            input_state_id=trace.root_state_id,
            output_state_id=trace.root_state_id,
            provenance=provenance,
            copy_source_state_id=trace.root_state_id,
            disposition="emit",
        )
    with pytest.raises(ConversationTraceError, match="candidates require a clarify"):
        TurnArtifactV1(
            operation=ConversationOperation.COPY_STATE,
            input_state_id=trace.root_state_id,
            output_state_id=trace.root_state_id,
            provenance=provenance,
            copy_source_state_id=trace.root_state_id,
            disposition_candidates=("stray",),
        )


def test_copy_conversation_state_still_defaults_to_no_disposition() -> None:
    """Existing COPY_STATE callers with no disposition are unaffected."""
    pack, trace = _trace_fixture()
    copied = copy_conversation_state(
        trace,
        source_state_id=trace.root_state_id,
        provenance=_provenance(trace.current.state),
    )
    assert copied.turns[-1].disposition is None
    assert copied.turns[-1].disposition_candidates == ()
