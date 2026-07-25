import json
from dataclasses import replace

import pytest

from slm_training.harnesses.capability_artifacts import (
    SCHEMA_VERSION,
    AnswerArtifactV1,
    CapabilityCertificateV1,
    CompilerCoverageV1,
    ComplexityVectorV1,
    DerivationActivityV1,
    LLMProvenanceV1,
    ProcessProvenanceV1,
    QAPairArtifactV1,
    QuestionArtifactV1,
    TeacherTraceMode,
    TeacherTraceV1,
    ValidationReportV1,
    artifact_from_json,
    record_json,
    require_publishable,
)
from slm_training.harnesses.staged import Capability

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _complexity() -> ComplexityVectorV1:
    return ComplexityVectorV1(
        ast_nodes=4,
        max_depth=2,
        decision_count=3,
        marker_count=1,
    )


def _process(name: str = "compiler") -> ProcessProvenanceV1:
    return ProcessProvenanceV1(
        process_id=name,
        process_version="v1",
        config_sha256=SHA_A,
        code_sha256=SHA_B,
    )


def _question(*, parent_ids: tuple[str, ...] = (SHA_A,)) -> QuestionArtifactV1:
    return QuestionArtifactV1(
        family_id="family-1",
        split_id="train",
        parent_ids=parent_ids,
        question_sha256=SHA_B,
        marker_table_id=SHA_C,
        grammar_start="document",
        category="generation",
        complexity=_complexity(),
    )


def _answer(
    *, surface_sha256: str = SHA_D, parent_ids: tuple[str, ...] = (SHA_A,)
) -> AnswerArtifactV1:
    return AnswerArtifactV1(
        family_id="family-1",
        split_id="train",
        parent_ids=parent_ids,
        canonical_ast_sha256=SHA_B,
        surface_sha256=surface_sha256,
        marker_table_id=SHA_C,
        grammar_start="document",
        category="generation",
        complexity=_complexity(),
    )


def test_equivalent_canonical_answers_share_semantic_id_not_record_bytes() -> None:
    first = _answer(surface_sha256=SHA_A, parent_ids=(SHA_C,))
    second = _answer(surface_sha256=SHA_D, parent_ids=(SHA_D,))

    assert first.semantic_id == second.semantic_id
    assert record_json(first) != record_json(second)
    assert "created_at" not in first.to_dict()
    assert "invocation_id" not in first.to_dict()


def test_separate_runs_retain_distinct_activity_ids_and_full_process_provenance() -> (
    None
):
    first = DerivationActivityV1(
        invocation_id="run-1",
        created_at="2026-07-23T20:00:00Z",
        source_ids=(SHA_A,),
        output_ids=(SHA_B,),
        process=_process(),
        seed=7,
        llm=LLMProvenanceV1(
            provider="provider",
            model="model-v1",
            prompt_sha256=SHA_C,
            response_sha256=SHA_D,
        ),
        teacher_trace=TeacherTraceV1(
            mode=TeacherTraceMode.EXACT,
            teacher_id="compiler",
            teacher_version="v3",
            trace_sha256=SHA_A,
        ),
    )
    second = replace(first, invocation_id="run-2")

    assert first.activity_id != second.activity_id
    assert first.to_dict()["process"]["process_version"] == "v1"
    assert first.to_dict()["teacher_trace"]["mode"] == "exact"


def test_accepted_set_identity_is_independent_of_canonical_preference() -> None:
    first = QAPairArtifactV1(
        question_id=SHA_A,
        accepted_answer_ids=(SHA_B, SHA_C),
        canonical_preference_answer_id=SHA_B,
        equivalence_relation_sha256=SHA_D,
    )
    second = replace(first, canonical_preference_answer_id=SHA_C)
    reordered = replace(first, accepted_answer_ids=(SHA_C, SHA_B))

    assert first.accepted_set_id == second.accepted_set_id == reordered.accepted_set_id
    assert record_json(first) != record_json(second)
    with pytest.raises(ValueError, match="must belong"):
        replace(first, canonical_preference_answer_id=SHA_D)


def test_validation_and_certificate_fail_closed_on_incomplete_evidence() -> None:
    incomplete = CompilerCoverageV1(
        required_paths=("parse", "serialize"),
        covered_paths=("parse",),
    )
    with pytest.raises(ValueError, match="complete compiler coverage"):
        ValidationReportV1(
            invocation_id="validation-1",
            created_at="2026-07-23T20:00:00Z",
            source_ids=(SHA_A,),
            process=_process("validator"),
            accepted=True,
            rejection_codes=(),
            compiler_coverage=incomplete,
        )
    with pytest.raises(ValueError, match="requires rejection codes"):
        CapabilityCertificateV1(
            capability=Capability.CAP0_GRAMMAR,
            plan_id="plan",
            plan_sha256=SHA_A,
            qa_pair_ids=(SHA_B,),
            validation_report_ids=(SHA_C,),
            gate_process=_process("gate"),
            passed=False,
        )


def test_publication_resolves_every_descendant_and_exact_process_version() -> None:
    question = _question()
    answer = _answer()
    pair = QAPairArtifactV1(
        question_id=question.semantic_id,
        accepted_answer_ids=(answer.semantic_id,),
        canonical_preference_answer_id=answer.semantic_id,
        equivalence_relation_sha256=SHA_D,
    )
    validation = ValidationReportV1(
        invocation_id="validation-1",
        created_at="2026-07-23T20:00:00Z",
        source_ids=(pair.semantic_id,),
        process=_process("validator"),
        accepted=True,
        rejection_codes=(),
        compiler_coverage=CompilerCoverageV1(
            required_paths=("parse",),
            covered_paths=("parse",),
        ),
    )
    certificate = CapabilityCertificateV1(
        capability=Capability.CAP0_GRAMMAR,
        plan_id="plan",
        plan_sha256=SHA_A,
        qa_pair_ids=(pair.semantic_id,),
        validation_report_ids=(validation.report_id,),
        gate_process=_process("gate"),
        passed=True,
    )
    derivation = DerivationActivityV1(
        invocation_id="derivation-1",
        created_at="2026-07-23T20:00:00Z",
        source_ids=(question.semantic_id,),
        output_ids=(answer.semantic_id,),
        process=_process(),
        seed=0,
    )
    rows = (question, answer, pair, derivation, validation, certificate)

    with pytest.raises(ValueError, match="unresolved source IDs"):
        require_publishable(rows)
    require_publishable(rows, external_ids=(SHA_A,))


def test_missing_required_provenance_and_schema_migrations_are_rejected() -> None:
    with pytest.raises(ValueError, match="process_version"):
        ProcessProvenanceV1(
            process_id="compiler",
            process_version="",
            config_sha256=SHA_A,
            code_sha256=SHA_B,
        )
    with pytest.raises(ValueError, match="schema mismatch"):
        replace(_answer(), schema_version="capability_artifacts/v2")
    assert _answer().schema_version == SCHEMA_VERSION


def test_json_round_trip_rejects_recorded_identity_and_migration_drift() -> None:
    answer = _answer()
    assert artifact_from_json(record_json(answer)) == answer

    payload = answer.to_dict()
    payload["semantic_id"] = SHA_A
    with pytest.raises(ValueError, match="not canonical"):
        artifact_from_json(json.dumps(payload))

    payload = answer.to_dict()
    payload["schema_version"] = "capability_artifacts/v2"
    with pytest.raises(ValueError, match="schema mismatch"):
        artifact_from_json(json.dumps(payload))


def test_serialization_and_hashes_are_deterministic_under_relation_order() -> None:
    first = _answer()
    second = replace(first, parent_ids=tuple(reversed(first.parent_ids)))
    assert record_json(first) == record_json(second)
    assert first.semantic_id == second.semantic_id
