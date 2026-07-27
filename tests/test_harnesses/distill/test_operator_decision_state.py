"""Tests for DSH4-01 operator decision-state export (SLM-386).

Exercises the acceptance bar from the Linear issue directly:

* traces capture state/action evidence before repair or fallback mutation;
* every accepted trajectory replays to a verifier-equivalent canonical
  result, and a genuinely diverged trajectory reports its first divergent
  action instead of silently succeeding;
* exact versus approximate evidence and coverage stay explicit;
* gold-state and on-policy states are both supported;
* the stop rule blocks any teacher-query export over unverified replay.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from slm_training.dsl.operators.contracts import (
    ActionEffectV1,
    ApplicationProvenanceV1,
    AstOperatorV1,
    BindingPhase,
    BoundArgumentV1,
    CompilerCoverage,
    EffectDeltaKind,
    EffectDeltaV1,
    OperatorArgumentSlotV1,
    RefKind,
)
from slm_training.dsl.operators.legal_set import (
    LegalSetCoverage,
    OperatorSupportVerdict,
)
from slm_training.dsl.operators.references import (
    ReferenceDescriptorV1,
    build_reference_table,
)
from slm_training.dsl.operators.registry import (
    OperatorLibraryV1,
    OperatorMutationV1,
    OperatorRejectedError,
    OperatorStateV1,
    RegisteredOperatorV1,
)
from slm_training.dsl.pack import get_pack
from slm_training.evals.solver_state_supervision import SupervisionSource
from slm_training.harnesses.distill.legal_set_teacher_trace import TeacherTraceManifest
from slm_training.harnesses.distill.operator_decision_state import (
    CaptureStage,
    OperatorDecisionStateError,
    build_operator_trajectory,
    capture_operator_decision_state,
    export_for_teacher_query,
    replay_operator_trajectory,
    write_operator_decision_state_traces,
)

SOURCE0 = 'root = TextContent(":hero.title")'
SOURCE1 = 'root = TextContent(":hero.body")'
OPERATOR_ID = "openui.dsh4_01_fixture"


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _manifest() -> TeacherTraceManifest:
    return TeacherTraceManifest(
        manifest_id="manifest-dsh4-01-fixture-0",
        teacher_model_id="fixture/teacher",
        teacher_revision="fixture",
        prompt_template_hash=_sha("fixture-prompt-template"),
        pack_id="openui",
        compiler_version="openui-fixture",
        state_schema_version="dsh4-01.v1",
        timestamp="2026-07-25T00:00:00+00:00",
        provenance={"source": "fixture"},
    )


def _descriptor(index: int) -> ReferenceDescriptorV1:
    return ReferenceDescriptorV1(
        ref_kind=RefKind.VALUE,
        semantic_fingerprint=_sha(f"dsh4-01-value-{index}"),
        value_type="openui.string",
    )


def _declaration() -> AstOperatorV1:
    return AstOperatorV1(
        operator_id=OPERATOR_ID,
        version="v1",
        domain="openui.ast",
        codomain="openui.ast",
        argument_slots=(
            OperatorArgumentSlotV1("value", RefKind.VALUE, BindingPhase.APPLICATION),
        ),
        preconditions=(),
        effect_signature=(EffectDeltaKind.TOPOLOGY,),
        locality="node",
        cost=1.0,
    )


def _make_executor(semantic_by_ref, sem0: str, sem1: str, *, strict: bool = False):
    """Two-value fixture executor: sem0 rewrites hero.title->body, sem1 body->subtitle.

    ``strict=True`` makes the sem1 branch always reject -- used to simulate a
    genuine replay divergence (the registry changed between recording and
    replay-time verification of an already-accepted trajectory step).
    """

    def execute(state, arguments):
        ref = arguments[0].value
        semantic = semantic_by_ref[ref]
        if semantic == sem0:
            if ":hero.title" not in state.source:
                raise OperatorRejectedError("fixture.title_not_present")
            return OperatorMutationV1(
                source=state.source.replace(":hero.title", ":hero.body"),
                effect=ActionEffectV1(
                    topology_deltas=(
                        EffectDeltaV1(EffectDeltaKind.TOPOLOGY, ref, "hero.title", "hero.body"),
                    ),
                    compiler_coverage=CompilerCoverage.EXACT,
                ),
            )
        if semantic == sem1:
            if strict:
                raise OperatorRejectedError("fixture.now_rejected")
            if ":hero.body" not in state.source:
                raise OperatorRejectedError("fixture.body_not_present")
            return OperatorMutationV1(
                source=state.source.replace(":hero.body", ":hero.subtitle"),
                effect=ActionEffectV1(
                    topology_deltas=(
                        EffectDeltaV1(EffectDeltaKind.TOPOLOGY, ref, "hero.body", "hero.subtitle"),
                    ),
                    compiler_coverage=CompilerCoverage.EXACT,
                ),
            )
        raise OperatorRejectedError("fixture.unknown_value")

    return execute


def _fixture(*, strict: bool = False):
    base_pack = get_pack("openui")
    state0 = OperatorStateV1.from_source(base_pack, SOURCE0)
    descriptors = (_descriptor(0), _descriptor(1))
    table0 = build_reference_table(
        request_id="request-1",
        state_digest=state0.state_digest,
        branch_digest=_sha("branch"),
        descriptors=descriptors,
        seed=1,
    )
    semantic_by_ref = {
        entry.ref: entry.descriptor.semantic_fingerprint for entry in table0.entries
    }
    sem0, sem1 = _descriptor(0).semantic_fingerprint, _descriptor(1).semantic_fingerprint
    declaration = _declaration()
    library = OperatorLibraryV1(
        (RegisteredOperatorV1(declaration, _make_executor(semantic_by_ref, sem0, sem1, strict=strict)),)
    )
    pack = replace(base_pack, operator_library=library)
    provenance0 = ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.fixture",
        compiler_version="v1",
        source_artifact_digest=_sha(SOURCE0),
        request_id="request-1",
    )
    return pack, library, state0, table0, provenance0, semantic_by_ref, sem0, sem1


def _table_for(state, semantic_by_ref_source):
    return build_reference_table(
        request_id="request-1",
        state_digest=state.state_digest,
        branch_digest=_sha("branch"),
        descriptors=(_descriptor(0), _descriptor(1)),
        seed=1,
    )


def _ref_for(table, sem_target):
    return next(
        entry.ref for entry in table.entries if entry.descriptor.semantic_fingerprint == sem_target
    )


def _build_two_step_trajectory(pack, library, state0, table0, provenance0, sem0, sem1):
    """Apply the fixture operator twice to build a genuine 2-step trajectory."""
    ref0 = _ref_for(table0, sem0)
    result1 = library.apply(
        pack, state0, OPERATOR_ID, (BoundArgumentV1("value", ref0),), provenance0
    )
    assert result1.succeeded
    state1 = result1.state
    assert state1 is not None

    table1 = _table_for(state1, None)
    provenance1 = ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.fixture",
        compiler_version="v1",
        source_artifact_digest=_sha(state1.source),
        request_id="request-1",
    )
    ref1 = _ref_for(table1, sem1)
    result2 = library.apply(
        pack, state1, OPERATOR_ID, (BoundArgumentV1("value", ref1),), provenance1
    )
    assert result2.succeeded
    state2 = result2.state
    assert state2 is not None

    trajectory = build_operator_trajectory(
        "traj-dsh4-01",
        (
            (state0, result1.application),
            (state1, result2.application),
        ),
    )
    return trajectory, state1, state2


def test_capture_binds_exact_state_and_complete_legal_set() -> None:
    pack, library, state0, table0, provenance0, semantic_by_ref, sem0, sem1 = _fixture()
    manifest = _manifest()
    ref0 = _ref_for(table0, sem0)
    dry = library.dry_run(pack, state0, OPERATOR_ID, (BoundArgumentV1("value", ref0),), provenance0)
    assert dry.succeeded

    trace = capture_operator_decision_state(
        pack=pack,
        library=library,
        state=state0,
        reference_table=table0,
        provenance=provenance0,
        manifest=manifest,
        supervision_source=SupervisionSource.GOLD,
        trace_id="trace-dsh4-01-000",
        accepted_application_ids=(dry.application_id,),
        current_scores={dry.application_id: 0.87},
    )

    assert trace.capture_stage is CaptureStage.PRE_REPAIR
    assert trace.manifest_id == manifest.manifest_id
    assert trace.supervision_source is SupervisionSource.GOLD
    # Exactly one candidate is legal at state0 (hero.title -> hero.body);
    # the sem1 candidate is rejected because hero.body is not yet present.
    assert trace.legal_set.coverage is LegalSetCoverage.COMPLETE
    assert trace.coverage is LegalSetCoverage.COMPLETE
    assert trace.legal_set.legal_operator_ids == (OPERATOR_ID,)
    assert len(trace.legal_set.operator_actions) == 1
    assert trace.legal_set.operator_actions[0].application_id == dry.application_id
    assert trace.accepted_application_ids == (dry.application_id,)
    assert trace.current_scores == {dry.application_id: 0.87}
    assert trace.approximate is False

    payload = trace.to_dict()
    assert payload["schema"] == "dsh4-01.v1"
    assert payload["legal_set"]["schema"] == "operator_legal_set/v2"
    # No free-form reasoning/chain-of-thought field exists on the export.
    assert "reasoning" not in payload
    assert "chain_of_thought" not in payload
    assert set(payload) == {
        "schema", "trace_id", "manifest_id", "supervision_source", "capture_stage",
        "approximate", "coverage", "timestamp", "state", "reference_table",
        "provenance", "legal_set", "accepted_application_ids", "current_scores",
    }


def test_capture_rejects_accepted_id_absent_from_legal_set() -> None:
    pack, library, state0, table0, provenance0, semantic_by_ref, sem0, sem1 = _fixture()
    with pytest.raises(OperatorDecisionStateError, match="absent from the exported legal set"):
        capture_operator_decision_state(
            pack=pack,
            library=library,
            state=state0,
            reference_table=table0,
            provenance=provenance0,
            manifest=_manifest(),
            supervision_source=SupervisionSource.GOLD,
            trace_id="trace-dsh4-01-bad-accept",
            accepted_application_ids=("not-a-real-application-id",),
        )


def test_capture_rejects_scored_id_absent_from_legal_set() -> None:
    pack, library, state0, table0, provenance0, semantic_by_ref, sem0, sem1 = _fixture()
    with pytest.raises(OperatorDecisionStateError, match="absent from the exported legal set"):
        capture_operator_decision_state(
            pack=pack,
            library=library,
            state=state0,
            reference_table=table0,
            provenance=provenance0,
            manifest=_manifest(),
            supervision_source=SupervisionSource.GOLD,
            trace_id="trace-dsh4-01-bad-score",
            current_scores={"not-a-real-application-id": 1.0},
        )


def test_gold_and_on_policy_supervision_sources_both_supported() -> None:
    pack, library, state0, table0, provenance0, semantic_by_ref, sem0, sem1 = _fixture()
    manifest = _manifest()
    for source in (SupervisionSource.GOLD, SupervisionSource.ON_POLICY):
        trace = capture_operator_decision_state(
            pack=pack,
            library=library,
            state=state0,
            reference_table=table0,
            provenance=provenance0,
            manifest=manifest,
            supervision_source=source,
            trace_id=f"trace-dsh4-01-{source.value}",
        )
        assert trace.supervision_source is source
        assert trace.to_dict()["supervision_source"] == source.value


def test_partial_coverage_and_approximate_flags_stay_explicit() -> None:
    pack, library, state0, table0, provenance0, semantic_by_ref, sem0, sem1 = _fixture()
    trace = capture_operator_decision_state(
        pack=pack,
        library=library,
        state=state0,
        reference_table=table0,
        provenance=provenance0,
        manifest=_manifest(),
        supervision_source=SupervisionSource.ON_POLICY,
        trace_id="trace-dsh4-01-partial",
        approximate=True,
        max_combinations_per_operator=1,
    )
    assert trace.coverage is LegalSetCoverage.PARTIAL
    entry = trace.legal_set.entries[0]
    assert entry.coverage is LegalSetCoverage.PARTIAL
    assert entry.evaluated_combinations == 1
    assert entry.total_combinations == 2
    # Budget truncation is lazy and never hard-prunes (DSH3-06): whichever
    # single candidate was evaluated first, the verdict stays SUPPORTED or
    # UNKNOWN -- never UNSUPPORTED with partial coverage.
    assert entry.verdict in (OperatorSupportVerdict.SUPPORTED, OperatorSupportVerdict.UNKNOWN)
    assert trace.approximate is True


def test_accepted_trajectory_replays_to_canonical_result() -> None:
    pack, library, state0, table0, provenance0, semantic_by_ref, sem0, sem1 = _fixture()
    trajectory, state1, state2 = _build_two_step_trajectory(
        pack, library, state0, table0, provenance0, sem0, sem1
    )
    result = replay_operator_trajectory(pack=pack, library=library, trajectory=trajectory)
    assert result.verified is True
    assert result.steps_replayed == 2
    assert result.first_divergence is None
    assert result.canonical_final_state is not None
    assert result.canonical_final_state.state_digest == state2.state_digest
    assert result.canonical_final_state.source == pack.canonicalize(SOURCE1.replace(":hero.body", ":hero.subtitle"))


def test_diverged_trajectory_reports_first_divergent_action() -> None:
    pack, library, state0, table0, provenance0, semantic_by_ref, sem0, sem1 = _fixture()
    trajectory, state1, state2 = _build_two_step_trajectory(
        pack, library, state0, table0, provenance0, sem0, sem1
    )

    # A strict registry that now rejects the sem1 branch outright -- the
    # registry changed between recording and replay-time re-verification.
    strict_pack, strict_library, _, _, _, _, _, _ = _fixture(strict=True)

    result = replay_operator_trajectory(
        pack=strict_pack, library=strict_library, trajectory=trajectory
    )
    assert result.verified is False
    assert result.steps_replayed == 1
    assert result.canonical_final_state is None
    assert result.first_divergence is not None
    assert result.first_divergence.step_index == 1
    assert result.first_divergence.code == "operator.replay.application_identity_mismatch"
    assert result.first_divergence.recorded_application_id == trajectory.steps[1].application.application_id


def test_export_for_teacher_query_blocks_on_unverified_trajectory() -> None:
    pack, library, state0, table0, provenance0, semantic_by_ref, sem0, sem1 = _fixture()
    manifest = _manifest()
    ref0 = _ref_for(table0, sem0)
    dry = library.dry_run(pack, state0, OPERATOR_ID, (BoundArgumentV1("value", ref0),), provenance0)
    trace = capture_operator_decision_state(
        pack=pack,
        library=library,
        state=state0,
        reference_table=table0,
        provenance=provenance0,
        manifest=manifest,
        supervision_source=SupervisionSource.GOLD,
        trace_id="trace-dsh4-01-gate",
        accepted_application_ids=(dry.application_id,),
    )

    trajectory, _, _ = _build_two_step_trajectory(
        pack, library, state0, table0, provenance0, sem0, sem1
    )
    strict_pack, strict_library, _, _, _, _, _, _ = _fixture(strict=True)

    with pytest.raises(OperatorDecisionStateError, match="repair the exporter before querying any teacher"):
        export_for_teacher_query(
            trace, trajectory, pack=strict_pack, library=strict_library
        )

    # No trajectory to verify -> exports the decision-state trace alone.
    exported = export_for_teacher_query(trace, None, pack=pack, library=library)
    assert exported == trace.to_dict()

    # A verified trajectory against the original (unchanged) registry exports cleanly.
    exported_verified = export_for_teacher_query(trace, trajectory, pack=pack, library=library)
    assert exported_verified == trace.to_dict()


def test_reference_table_stale_for_state_is_rejected() -> None:
    from slm_training.harnesses.distill.operator_decision_state import (
        OperatorDecisionStateTraceV1,
    )

    pack, library, state0, table0, provenance0, semantic_by_ref, sem0, sem1 = _fixture()
    manifest = _manifest()
    trace = capture_operator_decision_state(
        pack=pack,
        library=library,
        state=state0,
        reference_table=table0,
        provenance=provenance0,
        manifest=manifest,
        supervision_source=SupervisionSource.GOLD,
        trace_id="trace-dsh4-01-stale-ref",
    )

    other_state = OperatorStateV1.from_source(pack, SOURCE1)
    other_table = _table_for(other_state, None)

    with pytest.raises(OperatorDecisionStateError, match="reference table is stale"):
        OperatorDecisionStateTraceV1(
            trace_id=trace.trace_id,
            manifest_id=trace.manifest_id,
            supervision_source=trace.supervision_source,
            state=state0,
            reference_table=other_table,
            provenance=provenance0,
            legal_set=trace.legal_set,
            accepted_application_ids=trace.accepted_application_ids,
            current_scores=trace.current_scores,
            approximate=trace.approximate,
            capture_stage=trace.capture_stage,
            timestamp=trace.timestamp,
        )


def test_legal_set_not_bound_to_reference_table_is_rejected() -> None:
    from slm_training.harnesses.distill.operator_decision_state import (
        OperatorDecisionStateTraceV1,
    )

    pack, library, state0, table0, provenance0, semantic_by_ref, sem0, sem1 = _fixture()
    manifest = _manifest()
    trace0 = capture_operator_decision_state(
        pack=pack,
        library=library,
        state=state0,
        reference_table=table0,
        provenance=provenance0,
        manifest=manifest,
        supervision_source=SupervisionSource.GOLD,
        trace_id="trace-dsh4-01-mismatch-a",
    )

    _, state1, _ = _build_two_step_trajectory(
        pack, library, state0, table0, provenance0, sem0, sem1
    )
    table1 = _table_for(state1, None)
    provenance1 = ApplicationProvenanceV1(
        pack_id="openui",
        compiler_id="openui.fixture",
        compiler_version="v1",
        source_artifact_digest=_sha(state1.source),
        request_id="request-1",
    )
    trace1 = capture_operator_decision_state(
        pack=pack,
        library=library,
        state=state1,
        reference_table=table1,
        provenance=provenance1,
        manifest=manifest,
        supervision_source=SupervisionSource.ON_POLICY,
        trace_id="trace-dsh4-01-mismatch-b",
    )

    with pytest.raises(OperatorDecisionStateError, match="not bound to the exported reference table"):
        OperatorDecisionStateTraceV1(
            trace_id=trace0.trace_id,
            manifest_id=trace0.manifest_id,
            supervision_source=trace0.supervision_source,
            state=state0,
            reference_table=table0,
            provenance=provenance0,
            legal_set=trace1.legal_set,
            accepted_application_ids=(),
            current_scores=None,
            approximate=False,
            capture_stage=trace0.capture_stage,
            timestamp=trace0.timestamp,
        )


def test_write_operator_decision_state_traces_writes_jsonl(tmp_path) -> None:
    pack, library, state0, table0, provenance0, semantic_by_ref, sem0, sem1 = _fixture()
    manifest = _manifest()
    trace_a = capture_operator_decision_state(
        pack=pack,
        library=library,
        state=state0,
        reference_table=table0,
        provenance=provenance0,
        manifest=manifest,
        supervision_source=SupervisionSource.GOLD,
        trace_id="trace-dsh4-01-write-a",
    )
    trace_b = capture_operator_decision_state(
        pack=pack,
        library=library,
        state=state0,
        reference_table=table0,
        provenance=provenance0,
        manifest=manifest,
        supervision_source=SupervisionSource.ON_POLICY,
        trace_id="trace-dsh4-01-write-b",
    )

    path = tmp_path / "traces.jsonl"
    write_operator_decision_state_traces(path, [trace_a, trace_b])

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    loaded = [json.loads(line) for line in lines]
    assert {row["trace_id"] for row in loaded} == {trace_a.trace_id, trace_b.trace_id}
    assert loaded[0]["legal_set"]["schema"] == "operator_legal_set/v2"


def test_trajectory_requires_accepted_applications() -> None:
    from slm_training.harnesses.distill.operator_decision_state import (
        OperatorTrajectoryStepV1,
    )

    pack, library, state0, table0, provenance0, semantic_by_ref, sem0, sem1 = _fixture()
    rejected = library.apply(
        pack, state0, OPERATOR_ID, (BoundArgumentV1("value", _ref_for(table0, sem1)),), provenance0
    )
    assert not rejected.succeeded
    with pytest.raises(OperatorDecisionStateError, match="accepted"):
        OperatorTrajectoryStepV1(state_before=state0, application=rejected.application)
