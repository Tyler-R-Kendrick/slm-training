"""reasoning/revmath harness parity certificate must fail on real regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import verify_revmath_harness_parity as verifier
from slm_training.harnesses.reasoning.profiles import (
    DEFAULT_PROFILE_ID,
    REVMATH_PROFILE_ID,
    ReasoningProfileError,
    assert_default_unchanged,
    get_profile,
    list_profile_ids,
    resolve_profile,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_passes_on_the_real_repository():
    result = verifier.certify()
    assert result["schema"] == "revmath_harness_parity/v1"
    assert result["issue_key"] == "HARN-01"
    assert result["linear_id"] == "SLM-520"
    assert result["registration"]["profile_id"] == REVMATH_PROFILE_ID
    assert result["registration"]["default_profile_id"] == DEFAULT_PROFILE_ID
    assert "HARN-12" in result["absorbed_predecessors"]
    assert result["obligations"]["blockers"] == []
    assert "docs.skills_and_agent_surfaces" in result["obligations"]["present"]
    assert "commands" in result["obligations"]["categories"]


def test_profile_discovery_keeps_default():
    assert resolve_profile(None).profile_id == DEFAULT_PROFILE_ID
    assert resolve_profile("").profile_id == DEFAULT_PROFILE_ID
    assert get_profile(REVMATH_PROFILE_ID).opt_in is True
    assert get_profile(REVMATH_PROFILE_ID).task_semantics_ready is True
    assert REVMATH_PROFILE_ID in list_profile_ids()
    assert_default_unchanged()
    with pytest.raises(ReasoningProfileError, match="unknown reasoning profile"):
        get_profile("reasoning/does-not-exist")


def test_catches_silent_missing_obligation():
    doc = json.loads((REPO_ROOT / verifier.PARITY_PATH).read_text(encoding="utf-8"))
    doc["obligations"].append(
        {
            "id": "schemas.silent_gap",
            "category": "schemas",
            "required_owner": None,
            "status": "blocker",
            "blocker_issue": "",
            "notes": "must fail",
        }
    )
    with pytest.raises(verifier.RevmathHarnessParityError, match="without blocker_issue"):
        verifier.check_obligations(doc)


def test_catches_self_healing_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(verifier, "ROOT", tmp_path)
    doc = json.loads((REPO_ROOT / verifier.PARITY_PATH).read_text(encoding="utf-8"))
    doc["self_healing"]["must_freeze"].append(doc["self_healing"]["may_modify"][0])
    with pytest.raises(verifier.RevmathHarnessParityError, match="overlap"):
        verifier.check_self_healing(doc)
