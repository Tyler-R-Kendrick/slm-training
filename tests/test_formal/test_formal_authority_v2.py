"""EVID-06 / SLM-526: formal_authority/v2 adapters and invariants."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.autoresearch.formal import (
    formal_preflight_payload,
    formal_preflight_sha256,
    migrate_formal_preflight_v1,
)
from slm_training.formal.authority import (
    FORMAL_AUTHORITY_SCHEMA,
    CheckerResultRefV1,
    FormalAuthorityV2,
    build_formal_authority_v2,
    checker_result_from_backend,
    detect_payload_schema,
    from_formal_object_v1,
    from_formal_preflight_v1,
    load_formal_authority,
    negotiate_schema,
    prefer_write_payload,
    to_formal_object_v1,
    to_formal_preflight_v1,
)
from slm_training.formal.judgment import (
    ClassificationInputsV1,
    JudgmentOutcome,
    RefutationPayloadV1,
    SolverJudgmentV1,
    UnknownReason,
    classify,
)
from slm_training.formal.objects import (
    FORMAL_OBJECT_SCHEMA,
    export_closure_law,
    export_lean_claim,
)

ROOT = Path(__file__).resolve().parents[2]
SFF_PREFLIGHT = (
    ROOT
    / "src/slm_training/resources/experiments/semantic_factor_frontier"
    / "formal_preflights"
    / "sff_factor-membership-roundtrip.json"
)
SCHEMA_PATH = ROOT / "src/slm_training/resources/formal_authority.schema.json"


def test_json_schema_resource_matches_constant() -> None:
    payload = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    assert payload["$id"] == FORMAL_AUTHORITY_SCHEMA


def test_formal_object_round_trip_upgrade_downgrade() -> None:
    obj = export_lean_claim("Judgment.timeout_never_refuted")
    judgment = classify(
        ClassificationInputsV1(
            vss_verdict="supported",
            has_witness_digest=True,
            replay_checked=True,
            replay_ok=True,
        ),
        checked=True,
    )
    authority = from_formal_object_v1(
        obj,
        judgment=judgment,
        checker_results=(
            checker_result_from_backend(backend="python_structural", ok=True),
            checker_result_from_backend(backend="python_reference", ok=True),
        ),
    )
    assert authority.schema == FORMAL_AUTHORITY_SCHEMA
    assert authority.surface == "formal_object"
    assert authority.authority_class == "checked_semantic"
    assert authority.v1_content_digest == obj.content_digest

    rebuilt = FormalAuthorityV2.from_dict(authority.to_dict())
    assert rebuilt.content_digest == authority.content_digest
    assert to_formal_object_v1(rebuilt) == obj

    write = prefer_write_payload(authority)
    assert write["schema"] == FORMAL_AUTHORITY_SCHEMA
    assert detect_payload_schema(write) == FORMAL_AUTHORITY_SCHEMA


def test_formal_preflight_v1_remains_readable_and_round_trips() -> None:
    raw = json.loads(SFF_PREFLIGHT.read_text(encoding="utf-8"))
    preflight = migrate_formal_preflight_v1(raw)
    authority = from_formal_preflight_v1(preflight)
    assert authority.surface == "formal_preflight"
    assert authority.v1_content_digest == formal_preflight_sha256(preflight)
    assert authority.authority_class == "checked_semantic"
    assert authority.toolchain.lean_version == preflight.lean_version

    downgraded = to_formal_preflight_v1(authority)
    assert formal_preflight_payload(downgraded) == formal_preflight_payload(preflight)

    loaded = load_formal_authority(raw)
    assert loaded.v1_content_digest == authority.v1_content_digest


def test_unknown_preflight_never_checked_semantic_refutation() -> None:
    raw = json.loads(SFF_PREFLIGHT.read_text(encoding="utf-8"))
    raw["status"] = "unknown"
    raw.pop("counterexample", None)
    preflight = migrate_formal_preflight_v1(raw)
    authority = from_formal_preflight_v1(preflight)
    assert authority.judgment.outcome is JudgmentOutcome.UNKNOWN
    assert authority.judgment.checked is False
    assert authority.authority_class in {"unchecked", "empirical_remainder"}
    assert not authority.judgment.authorizes_semantic_conclusion()
    assert not authority.judgment.authorizes_removal()


def test_skipped_checker_blocks_checked_refutation() -> None:
    judgment = SolverJudgmentV1(
        outcome=JudgmentOutcome.REFUTED,
        payload=RefutationPayloadV1(
            refutation_digest="ab" * 32,
            source="counterexample",
        ),
        checked=True,
    )
    obj = export_closure_law("removable_only_replay_valid_unsupported")
    with pytest.raises(ValueError, match="checked semantic"):
        from_formal_object_v1(
            obj,
            judgment=judgment,
            checker_results=(
                CheckerResultRefV1(backend="python_structural", status="skipped"),
            ),
        )


def test_contradictory_authority_class_rejected() -> None:
    judgment = classify(
        ClassificationInputsV1(timed_out=True),
        checked=False,
    )
    assert judgment.outcome is JudgmentOutcome.UNKNOWN
    obj = export_lean_claim("Judgment.timeout_never_refuted")
    with pytest.raises(ValueError, match="authority_class"):
        build_formal_authority_v2(
            claim_id=obj.object_id,
            surface="formal_object",
            judgment=judgment,
            v1_artifact=obj.to_dict(),
            v1_content_digest=obj.content_digest,
            authority_class="checked_semantic",
        )


def test_empirical_remainder_recorded_separately() -> None:
    judgment = SolverJudgmentV1(
        outcome=JudgmentOutcome.UNKNOWN,
        payload=UnknownReason.INCONCLUSIVE,
        checked=False,
    )
    obj = export_lean_claim("Judgment.timeout_never_refuted")
    authority = from_formal_object_v1(
        obj,
        judgment=judgment,
        empirical_remainder_claim_ids=("wall_clock_latency",),
        empirical_remainder_notes="runtime remainder",
    )
    assert authority.authority_class == "empirical_remainder"
    assert authority.empirical_remainder_claim_ids == ("wall_clock_latency",)
    assert not authority.judgment.authorizes_semantic_conclusion()


def test_version_negotiation() -> None:
    assert negotiate_schema(None) == FORMAL_AUTHORITY_SCHEMA
    assert negotiate_schema(FORMAL_OBJECT_SCHEMA) == FORMAL_OBJECT_SCHEMA
    assert negotiate_schema("FormalPreflightV1") == "FormalPreflightV1"
    with pytest.raises(ValueError, match="unsupported formal authority schema"):
        negotiate_schema("formal_authority/v9")


def test_load_formal_object_requires_explicit_judgment() -> None:
    obj = export_lean_claim("Judgment.unchecked_never_authorizes")
    assert obj.schema == FORMAL_OBJECT_SCHEMA
    with pytest.raises(ValueError, match="explicit"):
        load_formal_authority(obj.to_dict())
