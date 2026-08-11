"""Tests for INTEG-01 / SLM-528 canonical proof-trace projection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.formal.proof_trace import (
    CANONICAL_FIELDS,
    PROOF_TRACE_SCHEMA,
    CanonicalProofTraceV1,
    frozen_fixture_rows,
    match_proof_trace_to_evidence,
    mutate_omit_event,
    mutate_reorder_events,
    mutate_stale_state_fingerprint,
    mutation_rejects_omit,
    mutation_rejects_reorder,
    mutation_rejects_stale_state,
    project_replay_bundle,
    project_runtime_evidence,
    proof_trace_integrity_ok,
    replay_proof_trace,
)
from slm_training.models.decode_stats import (
    DecodeIdentityV1,
    DecodeStats,
    build_decode_stats_record,
    build_mechanism_activation,
)
from slm_training.runtime.telemetry.replay_bundle import build_replay_bundle

_FIXTURES = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "slm_training"
    / "resources"
    / "formal"
    / "proof_trace_fixtures.v1.json"
)


def _identity(**overrides: object) -> DecodeIdentityV1:
    base: dict[str, object] = dict(
        contract_version=2,
        pack_id="openui",
        tokenizer_id="dsl-native/v5",
        artifact_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        evaluator_version="eval/v1",
        evaluator_hash="c" * 64,
        code_commit="deadbeef",
    )
    base.update(overrides)
    return DecodeIdentityV1(**base)  # type: ignore[arg-type]


def _bundle(**overrides: object):
    kwargs: dict[str, object] = dict(
        request_id="req-integ01",
        identity=_identity(),
        seed=11,
        decoder_choices=(
            {
                "kind": "solver_state",
                "state_fingerprint": "fp0",
                "decision_level": 0,
                "domain": {},
            },
            {"kind": "decoder_choice", "action": "select", "token_id": 9},
        ),
        legal_domain_size=3,
        legal_domain_status="complete",
        legal_domain_digest="d" * 64,
        artifact_state={"cache": "warm"},
    )
    kwargs.update(overrides)
    return build_replay_bundle(**kwargs)  # type: ignore[arg-type]


def test_projection_binds_identity_and_preserves_unobserved() -> None:
    stats = DecodeStats(tokens_emitted=2, forwards_count=4, dfa_sync_count=1)
    activation = build_mechanism_activation(
        mechanism_id="goal_support",
        eligible=True,
        invoked=False,
        abstained=True,
        evidence_class="fixture",
    )
    trace = project_runtime_evidence(
        bundle=_bundle(), stats=stats, activations=(activation,)
    )
    assert trace.schema_version == PROOF_TRACE_SCHEMA
    assert trace.identity["checkpoint_sha256"] == "b" * 64
    assert trace.identity["tokenizer_id"] == "dsl-native/v5"
    assert trace.identity["evaluator_hash"] == "c" * 64
    assert trace.claims_abstract_refinement is False
    assert trace.kern11_deferred is True
    assert proof_trace_integrity_ok(trace)
    fields = {event.field for event in trace.events if event.observed}
    assert "state_input_identity" in fields
    assert "legal_domain" in fields
    assert "state_transition" in fields
    assert "chosen_action" in fields
    assert "ranker_model_invocation" in fields
    assert "observed_work" in fields
    assert "cache_mechanism" in fields
    # No verifier/certificate evidence supplied → field stays unobserved.
    assert "verifier_solver_certificate" in trace.unobserved_fields
    assert trace.cost_projection is not None
    assert trace.cost_projection["abstract_cost"] == 2 + 1 + 4
    assert trace.source_digests["replay_bundle_hash"]


def test_stats_only_projection_leaves_bundle_fields_unobserved() -> None:
    stats = DecodeStats(forwards_count=1, tokens_emitted=1)
    record = build_decode_stats_record(
        stats,
        measurement_stage="steady_state",
        identity=_identity(),
    )
    trace = project_runtime_evidence(stats=record)
    assert trace.identity["checkpoint_sha256"] == "b" * 64
    assert "state_transition" in trace.unobserved_fields
    assert "legal_domain" in trace.unobserved_fields
    assert "chosen_action" in trace.unobserved_fields
    assert trace.source_digests["decode_record_digest"] == record.record_digest
    assert match_proof_trace_to_evidence(trace, stats=record)


def test_replay_and_match_round_trip() -> None:
    bundle = _bundle()
    stats = DecodeStats(tokens_emitted=1, forwards_count=2)
    trace = project_replay_bundle(bundle, stats=stats)
    replayed = replay_proof_trace(trace)
    assert replayed == trace
    assert match_proof_trace_to_evidence(trace, bundle=bundle, stats=stats)
    # Divergent evidence must not match.
    other = _bundle(request_id="other-req")
    assert not match_proof_trace_to_evidence(trace, bundle=other, stats=stats)


def test_cannot_claim_abstract_refinement() -> None:
    trace = project_replay_bundle(_bundle())
    payload = trace.to_dict()
    payload["claims_abstract_refinement"] = True
    # Keep hash so the refinement guard fires before (or instead of) hash check.
    with pytest.raises(ValueError, match="abstract refinement"):
        CanonicalProofTraceV1.from_dict(payload)


def test_mutation_reorder_omit_stale_state() -> None:
    trace = project_replay_bundle(_bundle())
    assert mutation_rejects_reorder(trace)
    assert mutation_rejects_omit(trace)
    assert mutation_rejects_stale_state(trace)
    assert not proof_trace_integrity_ok(mutate_reorder_events(trace))
    assert not proof_trace_integrity_ok(mutate_omit_event(trace))
    assert not proof_trace_integrity_ok(mutate_stale_state_fingerprint(trace))
    # Mutated traces also fail evidence match.
    mutated = mutate_reorder_events(trace)
    assert not match_proof_trace_to_evidence(mutated, bundle=_bundle())


def test_no_fabrication_when_legal_domain_absent() -> None:
    bundle = _bundle(
        legal_domain_size=None,
        legal_domain_status="unknown",
        legal_domain_digest=None,
        decoder_choices=(),
        artifact_state={},
    )
    trace = project_replay_bundle(bundle)
    legal = next(e for e in trace.events if e.field == "legal_domain")
    assert legal.observed is True  # explicit unknown status is observed
    assert legal.refs.get("legal_domain_size") is None
    assert "legal_domain_digest" not in legal.refs
    assert set(CANONICAL_FIELDS) >= set(trace.unobserved_fields)


def test_frozen_fixture_file() -> None:
    payload = json.loads(_FIXTURES.read_text(encoding="utf-8"))
    assert payload["schema"] == "proof_trace_fixtures/v1"
    assert payload["claims_abstract_refinement"] is False
    live = {row["id"]: row for row in frozen_fixture_rows()}
    for row in payload["cases"]:
        got = live[row["id"]]
        assert got["trace_hash"] == row["trace_hash"]
        assert got["event_count"] == row["event_count"]
        assert got["abstract_cost"] == row["abstract_cost"]
        assert got["claims_abstract_refinement"] is False
        assert got["kern11_deferred"] is True
