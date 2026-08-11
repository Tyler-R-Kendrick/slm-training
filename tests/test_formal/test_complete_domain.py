"""KERN-02 / SLM-517: proof-bearing complete-domain Lean↔Python parity tests."""

from __future__ import annotations

import pytest

from slm_training.formal.complete_domain import (
    CompleteDomainEvidenceV1,
    build_complete_domain,
    finite_search_authorized,
    from_legacy_coverage_flag,
    legacy_flag_authorizes_completeness,
)
from slm_training.formal.structural import (
    check_law,
    is_singleton_bypass_state,
    law_proved_complete_singleton,
)


def test_forged_coverage_complete_has_zero_authority() -> None:
    assert legacy_flag_authorizes_completeness(True) is False
    assert legacy_flag_authorizes_completeness(False) is False
    evidence = from_legacy_coverage_flag("state-a", [7], coverage_complete=True)
    assert evidence.legacy_coverage_complete is True
    assert evidence.is_proved_complete() is False
    assert evidence.authorizes_pruning() is False
    assert evidence.authorizes_forced_emission() is False
    assert evidence.singleton_token() is None
    assert evidence.authority_name() == "partial_coverage"


def test_boolean_singleton_law_demoted() -> None:
    assert is_singleton_bypass_state([7], True) is False
    # Table path: every Boolean-only row expects False (KERN-02 demotion).
    assert check_law("singleton_bypass") == []
    assert check_law("proved_complete_singleton") == []


def test_proved_singleton_authorizes_forced_emission() -> None:
    evidence = build_complete_domain("s1", [7], [7])
    assert evidence.is_proved_complete() is True
    assert evidence.authorizes_forced_emission(expected_state_id="s1") is True
    assert evidence.singleton_token(expected_state_id="s1") == 7
    assert law_proved_complete_singleton("s1", [7], [7], "s1") is True


def test_missing_candidate_rejects_completeness() -> None:
    evidence = build_complete_domain("s1", [7], [7, 8])
    assert evidence.checked_sound is True
    assert evidence.checked_complete is False
    assert evidence.is_proved_complete() is False
    assert evidence.authorizes_pruning() is False


def test_illegal_candidate_rejects_soundness() -> None:
    evidence = build_complete_domain("s1", [7, 9], [7, 8])
    assert evidence.checked_sound is False
    assert evidence.is_proved_complete() is False


def test_duplicate_candidates_reject() -> None:
    evidence = build_complete_domain("s1", [7, 7], [7])
    assert evidence.has_duplicate_candidates() is True
    assert evidence.checked_sound is False
    assert evidence.checked_complete is False
    assert evidence.is_proved_complete() is False


def test_stale_state_identity_rejects() -> None:
    evidence = build_complete_domain("s1", [7], [7])
    assert evidence.binds_state("s1") is True
    assert evidence.binds_state("other") is False
    assert evidence.authorizes_forced_emission(expected_state_id="other") is False
    assert finite_search_authorized(evidence, "other") is False
    assert finite_search_authorized(evidence, "s1") is True


def test_partial_coverage_non_destructive() -> None:
    evidence = from_legacy_coverage_flag("s1", [1, 2, 3], coverage_complete=False)
    assert evidence.authority_name() == "unknown_coverage"
    assert evidence.authorizes_pruning() is False
    assert evidence.authorizes_forced_emission() is False


def test_round_trip_rejects_forged_checked_bits() -> None:
    good = build_complete_domain("s1", [7], [7])
    payload = good.to_dict()
    payload["checked_sound"] = True
    payload["checked_complete"] = True
    payload["legal_members"] = [7, 8]
    with pytest.raises(ValueError, match="disagree with legality"):
        CompleteDomainEvidenceV1.from_dict(payload)


def test_evidence_forbids_bool_coercion() -> None:
    evidence = build_complete_domain("s1", [7], [7])
    with pytest.raises(TypeError, match="forbids truthy/falsy"):
        bool(evidence)


def test_empty_proved_domain_authorizes_pruning_not_emission() -> None:
    evidence = build_complete_domain("s1", [], [])
    assert evidence.is_proved_complete() is True
    assert evidence.authorizes_pruning(expected_state_id="s1") is True
    assert evidence.authorizes_forced_emission(expected_state_id="s1") is False
