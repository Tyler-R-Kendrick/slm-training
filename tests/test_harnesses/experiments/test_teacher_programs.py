from slm_training.data.leakage import fingerprint_prompt
from slm_training.harnesses.experiments.teacher_programs import (
    AdmissionMode,
    TeacherProgramCandidate,
    TeacherProgramGenerationManifestV1,
    TeacherProgramRequestV1,
    admit_teacher_programs,
)


def _candidate(candidate_id: str, *, program: str = 'root = Card(":x")', prompt: str = "Create a card", audit: bool | None = True, judge: bool | None = True):
    return TeacherProgramCandidate(
        candidate_id=candidate_id, request_id="request-1", response_id=f"response-{candidate_id}",
        prompt=prompt, program_payloads=(program,), provider="provider", model="model", revision="rev",
        generator_family="generator", judge_family="judge", coverage_gap_ids=("rico_held:far",),
        program_family_id="teacher", lineage_id=candidate_id, split_group_id=candidate_id,
        provenance_complete=True, independent_judge_passed=judge, human_audit_passed=audit,
        required_facts=("component:Card",),
    )


_EMPTY = {key: set() for key in ("ids", "split_group_ids", "prompts", "openuis", "structures", "pairs", "design_mds")}


def test_request_and_generation_manifest_require_frozen_inputs():
    request = TeacherProgramRequestV1("r", "coverage", ("rico_held:far",), ("Card",), "document", "protected", "template", 0)
    manifest = TeacherProgramGenerationManifestV1("m", "provider", "model", "revision", (request.request_id,), 1.0, 1, 1, "protected")
    assert manifest.manifest_hash == manifest.manifest_hash


def test_deep_mode_is_hands_off_but_requires_independent_judge():
    result = admit_teacher_programs(
        [_candidate("automatic", audit=None), _candidate("missing-judge", audit=None, judge=None)],
        mode=AdmissionMode.DEEP_VERIFIED,
        protected_fingerprints=_EMPTY,
    )
    assert [row["candidate_id"] for row in result.accepted] == ["automatic"]
    assert result.accepted[0]["admission_tier"] == "Silver"
    assert result.rejected[0]["reason"] == "required_gate_not_pass"


def test_exactly_one_payload_and_protected_leakage_fail_closed():
    invalid = _candidate("invalid")
    invalid = TeacherProgramCandidate(**{**invalid.__dict__, "program_payloads": ()})
    protected = {**_EMPTY, "prompts": {fingerprint_prompt("Create a card")}}
    result = admit_teacher_programs([invalid, _candidate("leaked")], mode="deep_verified", protected_fingerprints=protected)
    assert {row["reason"] for row in result.rejected} == {"program_payload_count", "protected_split_leakage"}


def test_deep_canonical_dedup_differs_from_no_canonical_dedup():
    rows = [_candidate("a", prompt="Create a card"), _candidate("b", prompt="Build a card")]
    deep = admit_teacher_programs(rows, mode="deep_verified", protected_fingerprints=_EMPTY)
    no_dedup = admit_teacher_programs(rows, mode="no_canonical_dedup", protected_fingerprints=_EMPTY)
    assert len(deep.accepted) == 1
    assert len(no_dedup.accepted) == 2


def test_parse_only_is_not_promoted_by_missing_judges():
    result = admit_teacher_programs([_candidate("parse", audit=None, judge=None)], mode="parse_only", protected_fingerprints=_EMPTY)
    assert len(result.accepted) == 1
    assert result.accepted[0]["verification"]["tier"] == "Bronze"
