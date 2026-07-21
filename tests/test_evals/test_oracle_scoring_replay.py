"""Tests for the SLM-260 oracle scoring replay harness."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from slm_training.data.contract import GenerationRequest
from slm_training.dsl.schema import ExampleRecord
from slm_training.evals.oracle_scoring_replay import (
    SCHEMA_VERSION,
    VARIANT_KINDS,
    OracleScoringReplayV1,
    build_fixture_records,
    build_replay_manifest,
    build_variant_rows,
    score_prediction,
    score_rows,
)


@pytest.fixture
def fixture_records() -> list[ExampleRecord]:
    return build_fixture_records()


@pytest.fixture
def replay_rows(fixture_records: list[ExampleRecord]) -> list[OracleScoringReplayV1]:
    return build_variant_rows(fixture_records)


@pytest.fixture
def scored_rows(replay_rows: list[OracleScoringReplayV1]) -> list[dict[str, object]]:
    return score_rows(replay_rows)


def test_schema_version_and_variant_kinds() -> None:
    assert SCHEMA_VERSION == "oracle_scoring_replay/v1"
    assert VARIANT_KINDS == (
        "exact_gold",
        "canonical_roundtrip",
        "alpha_renamed_equivalent",
        "egraph_equivalent",
        "unbound_reference",
        "wrong_component_or_property_role",
        "wrong_placeholder_identity",
        "prompt_contract_omission",
        "prompt_incompatible_but_valid",
        "duplicate_or_filler_gaming",
        "unreachable_or_dead_content",
    )


def test_fixture_records_match_archetypes(fixture_records: list[ExampleRecord]) -> None:
    assert [r.id for r in fixture_records] == [
        "card",
        "slider",
        "switch",
        "tabs",
        "callout",
        "image_block",
    ]
    for record in fixture_records:
        assert record.split == "adversarial"
        assert record.source == "oracle_scoring_replay_fixture"
        assert record.target_kind == "document"


def test_variant_rows_count_and_coverage(
    fixture_records: list[ExampleRecord],
    replay_rows: list[OracleScoringReplayV1],
) -> None:
    assert len(replay_rows) == len(fixture_records) * len(VARIANT_KINDS)
    kinds = {row.variant_kind for row in replay_rows}
    assert kinds == set(VARIANT_KINDS)
    record_ids = {row.record_id for row in replay_rows}
    assert record_ids == {r.id for r in fixture_records}


def test_expected_verdicts_by_variant_kind(replay_rows: list[OracleScoringReplayV1]) -> None:
    positive_kinds = {
        "exact_gold",
        "canonical_roundtrip",
        "alpha_renamed_equivalent",
        "egraph_equivalent",
    }
    for row in replay_rows:
        if row.variant_kind in positive_kinds:
            assert row.expected_verdict is True
        else:
            assert row.expected_verdict is False


def test_oracle_row_round_trip(replay_rows: list[OracleScoringReplayV1]) -> None:
    for original in replay_rows:
        payload = original.to_dict()
        restored = OracleScoringReplayV1.from_dict(payload)
        assert restored == original
        assert restored.slot_contract == tuple(original.slot_contract)


def test_score_prediction_keys_and_production_path(
    fixture_records: list[ExampleRecord],
) -> None:
    record = fixture_records[0]
    pred = record.openui
    request = GenerationRequest.from_record(record)
    detail = score_prediction(record, pred, request=request)

    expected_keys = {
        "parse_ok",
        "meaningful_program_v1",
        "binding_aware_meaningful_v2",
        "semantic_meaning_report_v2",
        "syntax_parse_valid",
        "raw_syntax_valid",
        "error",
        "placeholder_fidelity",
        "placeholder_fidelity_normalized",
        "placeholder_validity",
        "contract_precision",
        "contract_recall",
        "exact_match",
        "structural_similarity",
        "tree_edit_similarity",
        "component_type_recall",
        "reward_score",
        "prediction",
        "prediction_sha256",
        "generation_request",
        "source_record_sha256",
        "serialized",
    }
    assert set(detail) == expected_keys
    assert detail["parse_ok"] is True
    assert detail["meaningful_program_v1"] is True
    assert detail["binding_aware_meaningful_v2"] is True
    assert detail["syntax_parse_valid"] is True
    assert detail["prediction"] == pred
    assert detail["generation_request"] == request.to_dict()
    assert isinstance(detail["prediction_sha256"], str)
    assert isinstance(detail["source_record_sha256"], str)


def test_score_prediction_uses_request_or_builds_one(
    fixture_records: list[ExampleRecord],
) -> None:
    record = fixture_records[0]
    pred = record.openui
    detail_without_request = score_prediction(record, pred)
    detail_with_request = score_prediction(
        record, pred, request=GenerationRequest.from_record(record)
    )
    assert detail_without_request["binding_aware_meaningful_v2"] == detail_with_request[
        "binding_aware_meaningful_v2"
    ]


def test_positives_pass_and_negatives_fail(scored_rows: list[dict[str, object]]) -> None:
    """Oracle replay verdicts must match their expected polarity.

    Positive rows must pass both ``meaningful_program_v1`` and
    ``binding_aware_meaningful_v2``.  Negative rows must fail the stricter v2
    judge.
    """
    mismatches: list[str] = []
    for scored in scored_rows:
        row = OracleScoringReplayV1.from_dict(scored["row"])  # type: ignore[arg-type]
        detail = scored["detail"]  # type: ignore[index]
        expected = row.expected_verdict
        if expected is None:
            continue
        v1 = detail["meaningful_program_v1"]  # type: ignore[index]
        v2 = detail["binding_aware_meaningful_v2"]  # type: ignore[index]
        reasons = detail["semantic_meaning_report_v2"]["reason_codes"]  # type: ignore[index]

        if expected:
            if not v1 or not v2:
                mismatches.append(
                    f"{row.row_id}: expected positive, v1={v1}, v2={v2}, reasons={reasons}"
                )
        elif v2:
            mismatches.append(
                f"{row.row_id}: expected negative, v2={v2}, reasons={reasons}"
            )
    assert not mismatches, "\n".join(mismatches)


def test_manifest_structure(replay_rows: list[OracleScoringReplayV1]) -> None:
    scored = score_rows(replay_rows)
    manifest = build_replay_manifest(replay_rows, scored)
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["suite"] == "oracle_replay"
    assert manifest["n"] == len(replay_rows)
    assert set(manifest["variant_counts"]) == set(VARIANT_KINDS)
    assert 0.0 <= manifest["mean_parse_ok"] <= 1.0
    assert 0.0 <= manifest["meaningful_program_v1_rate"] <= 1.0
    assert 0.0 <= manifest["binding_aware_meaningful_v2_rate_strict"] <= 1.0
    assert "task_scoreboard" in manifest
    assert manifest["task_scoreboard"]["n"] == len(replay_rows)
    assert manifest["version_stamp"]["stamp_schema"] == "version_stamp/v1"
    assert "evals.scoring" in manifest["version_stamp"]["components"]
    assert "harness.oracle_scoring_replay" in manifest["version_stamp"]["components"]
    assert len(manifest["details"]) == len(replay_rows)


def test_cli_runs_and_writes_valid_json(tmp_path: Path) -> None:
    output = tmp_path / "replay.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.audit_gold_scoring",
            "--output",
            str(output),
            "--run-dir",
            str(tmp_path / "run"),
        ],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
    )
    status = json.loads(result.stdout)
    assert status["status"] == "ok"
    assert Path(status["output"]).exists()
    manifest = json.loads(output.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["n"] == len(VARIANT_KINDS) * len(build_fixture_records())
