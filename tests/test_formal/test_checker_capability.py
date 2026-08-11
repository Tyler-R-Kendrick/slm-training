"""EVID-08 / SLM-535: checker capabilities, trust domains, acceptance policies."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.formal.checker_capability import (
    CAPABILITY_VOCABULARY,
    CHECKER_CAPABILITY_SCHEMA,
    CONFORMANCE_POLICY,
    SEMANTIC_AUTHORITY_POLICY,
    TRUST_DOMAIN_LEAN4_KERNEL,
    TRUST_DOMAIN_PYTHON_STRUCTURAL_FAMILY,
    AcceptancePolicyV1,
    CheckerRunView,
    advertisements_from_registry,
    build_registry_payload,
    capability_satisfied_by,
    evaluate_acceptance_policy,
    load_registry,
    semantic_authority_requires_distinct_trust_domains,
    trust_domain_for,
)
from slm_training.formal.checkers import (
    check_python_reference,
    check_python_structural,
)
from slm_training.formal.objects import FormalObjectV1, export_lean_claim

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = (
    ROOT
    / "src/slm_training/resources/formal/evid08_capability_fixtures.v1.json"
)


def test_registry_matches_defaults_and_vocabulary() -> None:
    payload = load_registry(root=ROOT)
    assert payload["schema"] == CHECKER_CAPABILITY_SCHEMA
    assert set(payload["capability_vocabulary"]) == CAPABILITY_VOCABULARY
    built = build_registry_payload()
    assert built["checkers"] == payload["checkers"]
    assert built["policies"] == payload["policies"]
    ads = advertisements_from_registry(root=ROOT)
    assert ads["python_structural"].role == "redundant_conformance"
    assert ads["python_reference"].trust_domain == TRUST_DOMAIN_PYTHON_STRUCTURAL_FAMILY
    assert ads["lean_kernel"].trust_domain == TRUST_DOMAIN_LEAN4_KERNEL


def test_unadvertised_capability_cannot_be_satisfied() -> None:
    assert not capability_satisfied_by(
        "runtime_refinement", backend="python_structural"
    )
    report = evaluate_acceptance_policy(
        (CheckerRunView(backend="python_structural", ok=True),),
        AcceptancePolicyV1(
            policy_id="probe/v1",
            required_capabilities=frozenset({"runtime_refinement"}),
            min_distinct_trust_domains=1,
            require_independent_verification=False,
        ),
        claimed_capabilities=("runtime_refinement",),
    )
    assert report.accepted is False
    assert "runtime_refinement" in report.missing_capabilities


def test_skipped_is_not_successful_semantic_check() -> None:
    report = evaluate_acceptance_policy(
        (
            CheckerRunView(backend="python_structural", ok=True),
            CheckerRunView(
                backend="python_replay", ok=True, skipped=True, status="skipped"
            ),
        ),
        SEMANTIC_AUTHORITY_POLICY,
    )
    assert report.accepted is False
    assert report.verification_kind == "incomplete"
    assert "python_replay" in report.skipped_backends


def test_shared_family_is_redundant_not_independent() -> None:
    assert not semantic_authority_requires_distinct_trust_domains(
        ("python_structural", "python_reference")
    )
    assert semantic_authority_requires_distinct_trust_domains(
        ("python_structural", "lean_kernel")
    )
    report = evaluate_acceptance_policy(
        (
            CheckerRunView(backend="python_structural", ok=True),
            CheckerRunView(backend="python_reference", ok=True),
        ),
        SEMANTIC_AUTHORITY_POLICY,
    )
    assert report.accepted is False
    assert report.verification_kind == "redundant_conformance"
    assert report.redundant_backends == ("python_structural", "python_reference")
    assert report.independent_backends == ()


def test_conformance_policy_allows_redundant_pair() -> None:
    report = evaluate_acceptance_policy(
        (
            CheckerRunView(backend="python_structural", ok=True),
            CheckerRunView(backend="python_reference", ok=True),
        ),
        CONFORMANCE_POLICY,
    )
    assert report.accepted is True
    assert report.verification_kind == "redundant_conformance"


def test_independent_pair_satisfies_semantic_policy() -> None:
    report = evaluate_acceptance_policy(
        (
            CheckerRunView(backend="python_structural", ok=True),
            CheckerRunView(backend="lean_kernel", ok=True),
        ),
        SEMANTIC_AUTHORITY_POLICY,
    )
    assert report.accepted is True
    assert report.verification_kind == "independent_verification"
    assert TRUST_DOMAIN_LEAN4_KERNEL in report.trust_domains


def test_shared_serialization_fixtures() -> None:
    fixtures = json.loads(FIXTURES.read_text(encoding="utf-8"))
    assert fixtures["schema"] == "evid08_capability_fixtures/v1"
    obj = export_lean_claim("Judgment.timeout_never_refuted")
    raw = obj.to_dict()
    digest = raw["content_digest"]
    raw["content_digest"] = ("0" if digest[0] != "0" else "1") + digest[1:]
    with pytest.raises(ValueError):
        FormalObjectV1.from_dict(raw)
    assert check_python_structural(obj).ok
    assert check_python_reference(obj).ok
    assert trust_domain_for("python_structural") == trust_domain_for(
        "python_reference"
    )
