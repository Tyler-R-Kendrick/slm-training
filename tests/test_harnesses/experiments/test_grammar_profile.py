"""Regression tests for CAP5-01 grammar/profile manifest (SLM-100)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.dsl.analysis.arity import DecisionDifficulty
from slm_training.harnesses.experiments.grammar_profile import (
    MANIFEST_SCHEMA,
    GrammarProfile,
    build_grammar_profile,
    build_grammar_profile_manifest,
    validate_grammar_profile_manifest,
)


def _difficulty(
    *,
    state: str = "fp",
    arity: int,
    entropy: float | None,
    source_hash: str = "abcd1234",
) -> DecisionDifficulty:
    import math

    return DecisionDifficulty(
        state_fingerprint=state,
        live_legal_action_count=arity,
        log2_live_legal_action_count=math.log2(max(arity, 1)),
        posterior_entropy_bits=entropy,
        top1_margin=None,
        completion_support_size_exact=None,
        source_hash=source_hash,
    )


def test_build_grammar_profile_empty() -> None:
    profile = build_grammar_profile([], profile_id="empty", signature="none")
    assert profile.profile_id == "empty"
    assert profile.decision_count == 0
    assert profile.mean_arity == 0.0
    assert profile.max_arity == 0
    assert profile.mean_entropy_bits is None
    assert profile.max_entropy_bits is None


def test_build_grammar_profile_aggregates() -> None:
    difficulties = [
        _difficulty(arity=2, entropy=1.0),
        _difficulty(arity=4, entropy=2.0),
        _difficulty(arity=4, entropy=2.0),
    ]
    profile = build_grammar_profile(
        difficulties, profile_id="p1", signature="test"
    )
    assert profile.decision_count == 3
    assert profile.mean_arity == pytest.approx(10 / 3)
    assert profile.max_arity == 4
    assert profile.mean_entropy_bits == pytest.approx(5 / 3)
    assert profile.max_entropy_bits == pytest.approx(2.0)
    assert profile.source_hash is not None
    assert len(profile.source_hash) == 16


def test_build_grammar_profile_honors_missing_entropy() -> None:
    difficulties = [
        _difficulty(arity=2, entropy=None),
        _difficulty(arity=2, entropy=1.0),
    ]
    profile = build_grammar_profile(
        difficulties, profile_id="p2", signature="partial"
    )
    assert profile.mean_entropy_bits == pytest.approx(1.0)
    assert profile.max_entropy_bits == pytest.approx(1.0)


def test_grammar_profile_round_trip_dict() -> None:
    profile = GrammarProfile(
        profile_id="p",
        signature="s",
        decision_count=2,
        mean_arity=2.5,
        max_arity=3,
        mean_entropy_bits=1.0,
        max_entropy_bits=1.5,
        source_hash="deadbeef",
    )
    data = profile.to_dict()
    assert data["schema_version"] == MANIFEST_SCHEMA
    restored = GrammarProfile.from_dict(data)
    assert restored == profile


def test_build_manifest_hashes_and_validates() -> None:
    profile = build_grammar_profile(
        [_difficulty(arity=2, entropy=1.0)],
        profile_id="p3",
        signature="sig",
    )
    manifest = build_grammar_profile_manifest(
        [profile],
        run_id="r1",
        source_manifest_sha="sha123",
        note="test",
    )
    assert manifest["schema_version"] == MANIFEST_SCHEMA
    assert manifest["run_id"] == "r1"
    assert manifest["profile_count"] == 1
    assert manifest["manifest_hash"] is not None
    assert len(manifest["manifest_hash"]) == 16
    assert validate_grammar_profile_manifest(manifest) == []


def test_validate_manifest_catches_errors() -> None:
    assert validate_grammar_profile_manifest({}) != []
    assert validate_grammar_profile_manifest({"schema_version": "wrong"}) != []
    assert validate_grammar_profile_manifest(
        {"schema_version": MANIFEST_SCHEMA, "run_id": "r", "profiles": [{}]}
    ) != []


def test_cli_builds_manifest(tmp_path: Path) -> None:
    from scripts.build_grammar_profile_manifest import main

    diffs = [
        _difficulty(arity=2, entropy=1.0).to_dict(),
        _difficulty(arity=4, entropy=2.0).to_dict(),
    ]
    in_file = tmp_path / "diffs.jsonl"
    out_file = tmp_path / "manifest.json"
    with in_file.open("w", encoding="utf-8") as f:
        for d in diffs:
            f.write(json.dumps(d) + "\n")

    rc = main(
        [
            "--decision-difficulties",
            str(in_file),
            "--run-id",
            "cap5-01-cli",
            "--profile-id",
            "profile_a",
            "--signature",
            "kind=bind/legal=2,4",
            "--out",
            str(out_file),
        ]
    )
    assert rc == 0
    assert out_file.is_file()
    manifest = json.loads(out_file.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == MANIFEST_SCHEMA
    assert manifest["run_id"] == "cap5-01-cli"
    assert manifest["profile_count"] == 1
    assert manifest["profiles"][0]["profile_id"] == "profile_a"
    assert manifest["profiles"][0]["max_arity"] == 4
