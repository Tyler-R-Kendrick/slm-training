"""KERN-11 / SLM-539: fixture-level proof-trace refinement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.formal.authority import FORMAL_AUTHORITY_SCHEMA
from slm_training.formal.objects import lean_claim_catalog
from slm_training.formal.trace_refinement import (
    CERTIFICATE_SCHEMA,
    DEFAULT_ASSUMPTIONS,
    SCHEMA,
    SUPPORTED_FIXTURE_IDS,
    THEOREM_FIXTURE_REFINES,
    THEOREM_ILLEGAL_COMMIT,
    THEOREM_SCOPE,
    bind_refinement_authority,
    build_supported_fixture_trace,
    check_trace_refinement,
    emit_refinement_certificate,
    frozen_refinement_fixture_rows,
    inject_fabricated_completeness,
    inject_fabricated_refutation,
    inject_illegal_commit,
    inject_mismatched_identity,
    inject_omitted_model_forwards,
    inject_reordered_events,
    inject_stale_state_domain,
    load_fixture_payload,
    reseal_trace,
)

_FIXTURES = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "slm_training"
    / "resources"
    / "formal"
    / "trace_refinement_fixtures.v1.json"
)


def test_valid_frozen_fixture_refines_and_binds_v2() -> None:
    trace = build_supported_fixture_trace()
    result = check_trace_refinement(
        trace,
        fixture_id="bundle_stats_mechanism",
        expected_cost=28,
        expected_forwards=11,
    )
    assert result.accepted is True
    assert result.reject_reason is None
    assert result.assumptions.fixture_subset_scoped is True
    assert result.assumptions.claims_full_pytorch_semantics is False
    assert result.assumptions.claims_physical_latency_equivalence is False

    cert = emit_refinement_certificate(
        trace,
        fixture_id="bundle_stats_mechanism",
        expected_cost=28,
        expected_forwards=11,
    )
    assert cert.schema == CERTIFICATE_SCHEMA
    authority = bind_refinement_authority(cert)
    assert authority.schema == FORMAL_AUTHORITY_SCHEMA
    assert authority.claim_id.endswith(THEOREM_FIXTURE_REFINES)
    assert authority.judgment.outcome.value == "witnessed"
    assert authority.replay_digest == trace.trace_hash


@pytest.mark.parametrize(
    ("injector", "reason"),
    [
        (inject_illegal_commit, "illegal_commit"),
        (inject_fabricated_completeness, "fabricated_completeness"),
        (inject_fabricated_refutation, "fabricated_refutation"),
        (inject_reordered_events, "reordered_events"),
        (inject_omitted_model_forwards, "omitted_model_forwards"),
        (inject_mismatched_identity, "mismatched_identity"),
        (inject_stale_state_domain, "stale_state_domain"),
    ],
)
def test_injected_invalid_traces_fail_semantically(injector, reason: str) -> None:
    trace = build_supported_fixture_trace()
    mutated = reseal_trace(injector(trace))
    result = check_trace_refinement(
        mutated,
        fixture_id="bundle_stats_mechanism",
        expected_cost=28,
        expected_forwards=11,
    )
    assert result.accepted is False
    assert result.reject_reason == reason


def test_unsealed_injection_fails_integrity() -> None:
    trace = build_supported_fixture_trace()
    mutated = inject_reordered_events(trace)
    result = check_trace_refinement(
        mutated,
        fixture_id="bundle_stats_mechanism",
        expected_cost=28,
        expected_forwards=11,
    )
    assert result.accepted is False
    assert result.reject_reason == "integrity"


def test_unsupported_fixture_id_rejected() -> None:
    trace = build_supported_fixture_trace()
    result = check_trace_refinement(trace, fixture_id="not_a_supported_fixture")
    assert result.accepted is False
    assert result.reject_reason == "unsupported_fixture"


def test_frozen_fixture_rows_match_resource() -> None:
    rows = frozen_refinement_fixture_rows()
    payload = load_fixture_payload()
    assert payload["schema"] == SCHEMA
    assert set(payload["supported_fixture_ids"]) == set(SUPPORTED_FIXTURE_IDS)
    assert payload["assumptions"] == DEFAULT_ASSUMPTIONS.to_dict()
    assert payload["cases"] == rows
    assert _FIXTURES.is_file()
    disk = json.loads(_FIXTURES.read_text(encoding="utf-8"))
    assert disk == payload


def test_lean_claim_catalog_registers_kern11_theorems() -> None:
    catalog = lean_claim_catalog()
    for theorem_id in (
        THEOREM_FIXTURE_REFINES,
        THEOREM_ILLEGAL_COMMIT,
        THEOREM_SCOPE,
    ):
        assert theorem_id in catalog
        assert catalog[theorem_id]["module"] == "LeverProofLean.ProofTraceRefinement"
