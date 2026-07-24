import hashlib

import pytest

from slm_training.autoresearch.storage import CampaignStore
from slm_training.data.leakage import fingerprint_prompt
from slm_training.data.store import DataStore
from slm_training.dsl.schema import load_jsonl
from slm_training.harnesses.experiments.teacher_programs import (
    AdmissionMode,
    TeacherBudgetExceeded,
    TeacherProgramCandidate,
    TeacherProgramExecutionRequestV1,
    TeacherProgramExecutor,
    TeacherProgramGenerationManifestV1,
    TeacherProgramRequestV1,
    TeacherRawArchiveRefV1,
    admit_teacher_programs,
    materialize_teacher_admission,
)


def _judge_evidence(
    candidate_id: str,
    prompt: str,
    program: str,
    *,
    approved: bool,
    provider: str = "judge-provider",
    model_family: str = "judge",
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "provider": provider,
        "model_family": model_family,
        "approved": approved,
        "raw_artifact_sha256": "a" * 64,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "program_sha256": hashlib.sha256(program.encode("utf-8")).hexdigest(),
    }


def _candidate(
    candidate_id: str,
    *,
    program: str = 'root = Card(":x")',
    prompt: str = "Create a card",
    audit: bool | None = True,
    judge: bool | None = True,
    judge_evidence: dict[str, object] | None = None,
):
    if judge_evidence is None and judge is not None:
        judge_evidence = _judge_evidence(candidate_id, prompt, program, approved=judge)
    return TeacherProgramCandidate(
        candidate_id=candidate_id,
        request_id="request-1",
        response_id=f"response-{candidate_id}",
        prompt=prompt,
        program_payloads=(program,),
        provider="provider",
        model="model",
        revision="rev",
        generator_family="generator",
        judge_family="judge",
        coverage_gap_ids=("rico_held:far",),
        program_family_id="teacher",
        lineage_id=candidate_id,
        split_group_id=candidate_id,
        provenance_complete=True,
        independent_judge_passed=judge,
        judge_evidence=judge_evidence,
        human_audit_passed=audit,
        required_facts=("component:Card",),
    )


_EMPTY = {
    key: set()
    for key in (
        "ids",
        "split_group_ids",
        "prompts",
        "openuis",
        "structures",
        "pairs",
        "design_mds",
    )
}


def test_request_and_generation_manifest_require_frozen_inputs():
    request = TeacherProgramRequestV1(
        "r",
        "coverage",
        ("rico_held:far",),
        ("Card",),
        "document",
        "protected",
        "template",
        0,
    )
    manifest = TeacherProgramGenerationManifestV1(
        "m",
        "provider",
        "model",
        "revision",
        (request.request_id,),
        1.0,
        1,
        1,
        "protected",
    )
    assert manifest.manifest_hash == manifest.manifest_hash


def test_deep_mode_is_hands_off_but_requires_independent_judge():
    result = admit_teacher_programs(
        [
            _candidate("automatic", audit=None),
            _candidate("missing-judge", audit=None, judge=None),
        ],
        mode=AdmissionMode.DEEP_VERIFIED,
        protected_fingerprints=_EMPTY,
    )
    assert [row["candidate_id"] for row in result.accepted] == ["automatic"]
    assert result.accepted[0]["admission_tier"] == "Silver"
    assert result.rejected[0]["reason"] == "independent_judge_evidence_missing"


def test_deep_mode_rejects_asserted_or_tampered_judge_evidence():
    bare = _candidate("bare", judge=True, judge_evidence={})
    same_provider = _candidate(
        "same-provider",
        judge_evidence=_judge_evidence(
            "same-provider",
            "Create a card",
            'root = Card(":x")',
            approved=True,
            provider="provider",
        ),
    )
    tampered = _candidate(
        "tampered",
        judge_evidence={
            **_judge_evidence(
                "tampered", "Create a card", 'root = Card(":x")', approved=True
            ),
            "program_sha256": "b" * 64,
        },
    )
    result = admit_teacher_programs(
        [bare, same_provider, tampered],
        mode=AdmissionMode.DEEP_VERIFIED,
        protected_fingerprints=_EMPTY,
    )
    assert [row["reason"] for row in result.rejected] == [
        "independent_judge_evidence_missing:candidate_id,provider,model_family,approved,raw_artifact_sha256,prompt_sha256,program_sha256",
        "independent_judge_provider_not_independent",
        "independent_judge_program_hash_mismatch",
    ]


def test_exactly_one_payload_and_protected_leakage_fail_closed():
    invalid = _candidate("invalid")
    invalid = TeacherProgramCandidate(**{**invalid.__dict__, "program_payloads": ()})
    protected = {**_EMPTY, "prompts": {fingerprint_prompt("Create a card")}}
    result = admit_teacher_programs(
        [invalid, _candidate("leaked")],
        mode="deep_verified",
        protected_fingerprints=protected,
    )
    assert {row["reason"] for row in result.rejected} == {
        "program_payload_count",
        "protected_split_leakage",
    }


def test_deep_canonical_dedup_differs_from_no_canonical_dedup():
    rows = [
        _candidate("a", prompt="Create a card"),
        _candidate("b", prompt="Build a card"),
    ]
    deep = admit_teacher_programs(
        rows, mode="deep_verified", protected_fingerprints=_EMPTY
    )
    no_dedup = admit_teacher_programs(
        rows, mode="no_canonical_dedup", protected_fingerprints=_EMPTY
    )
    assert len(deep.accepted) == 1
    assert len(no_dedup.accepted) == 2


def test_parse_only_is_not_promoted_by_missing_judges():
    result = admit_teacher_programs(
        [_candidate("parse", audit=None, judge=None)],
        mode="parse_only",
        protected_fingerprints=_EMPTY,
    )
    assert len(result.accepted) == 1
    assert result.accepted[0]["verification"]["tier"] == "Bronze"


def test_materialize_deep_admission_is_train_snapshot_with_raw_pointer(tmp_path):
    result = admit_teacher_programs(
        [_candidate("deep", audit=None)],
        mode="deep_verified",
        protected_fingerprints=_EMPTY,
    )
    directory = tmp_path / "outputs" / "data" / "train" / "teacher_v1"
    manifest = materialize_teacher_admission(
        result,
        output_dir=directory,
        dataset_id="teacher_v1",
        raw_archive=TeacherRawArchiveRefV1("hf://datasets/example/raw", "a" * 64),
    )
    record = load_jsonl(directory / "records.jsonl")[0]
    assert record.meta["verification_tier"] == "Silver"
    assert record.meta["teacher_program_admission"]["raw_archive"]["uri"].startswith(
        "hf://"
    )
    assert (
        DataStore(tmp_path).resolve("train", "teacher_v1").fingerprint
        == manifest["content_fingerprint"]
    )


def test_materialize_rejects_controls(tmp_path):
    result = admit_teacher_programs(
        [_candidate("parse", audit=None, judge=None)],
        mode="parse_only",
        protected_fingerprints=_EMPTY,
    )
    with pytest.raises(ValueError, match="deep_verified"):
        materialize_teacher_admission(
            result,
            output_dir=tmp_path / "teacher",
            dataset_id="teacher_v1",
            raw_archive=TeacherRawArchiveRefV1("hf://datasets/example/raw", "a" * 64),
        )


def _executable_manifest():
    return TeacherProgramGenerationManifestV1(
        "manifest",
        "provider",
        "model",
        "revision",
        ("request-1",),
        1.0,
        10,
        10,
        "protected",
        input_cost_per_1k_usd=1.0,
        output_cost_per_1k_usd=1.0,
        campaign_manifest_sha256="b" * 64,
    )


def test_executor_blocks_budget_before_provider_call(tmp_path):
    calls = 0

    def transport(_request):
        nonlocal calls
        calls += 1
        return {"usage": {"input_tokens": 1, "output_tokens": 1}}

    executor = TeacherProgramExecutor(
        _executable_manifest(), CampaignStore("teacher", tmp_path)
    )
    with pytest.raises(TeacherBudgetExceeded):
        executor.execute(
            [TeacherProgramExecutionRequestV1("request-1", "intent", 11, 1)],
            transport,
        )
    assert calls == 0
    assert executor.archive.verify_event_chain() == []


def test_executor_archives_usage_and_resume_skips_completed_request(tmp_path):
    calls = 0

    def transport(_request):
        nonlocal calls
        calls += 1
        return {
            "raw": {"response_id": "response-1"},
            "usage": {"input_tokens": 2, "output_tokens": 3},
        }

    executor = TeacherProgramExecutor(
        _executable_manifest(), CampaignStore("teacher", tmp_path)
    )
    request = TeacherProgramExecutionRequestV1("request-1", "intent", 4, 5)
    first = executor.execute([request], transport)
    second = executor.execute([request], transport)
    assert first.completed_request_ids == ("request-1",)
    assert first.spent_dollars == pytest.approx(0.005)
    assert second.skipped_request_ids == ("request-1",)
    assert calls == 1
    assert [event["event_type"] for event in executor.archive.verify_event_chain()] == [
        "teacher_program_attempted",
        "teacher_program_completed",
    ]
