"""KERN-08 / SLM-531: mechanism trigger / no-effect / dominance certificates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slm_training.formal.mechanism_trigger import (
    CACHE_REUSE,
    CERTIFICATE_SCHEMA,
    CLOSURE_REMOVAL,
    DOMINANCE_SCHEMA,
    FIXTURE_SCHEMA,
    FORCED_SPAN,
    NEURAL_FORWARD_ONLY,
    SINGLETON_BYPASS,
    THEOREM_UNSAFE,
    UNCONSTRAINED_RERANK,
    CertificateError,
    TriggerEvidence,
    certified_fixture_rows,
    dominates_locally,
    emit_dominance_certificate,
    emit_no_effect_certificate,
    necessary_trigger,
    observation_from_cache_reuse,
    observation_from_closure_removal,
    observation_from_forced_span,
    observation_from_singleton_bypass,
    outputs_equal,
    trigger_evidence_complete,
    unconstrained_rerank_counterexample,
    unknown_activation_never_no_effect,
)
from slm_training.formal.objects import lean_claim_catalog

_FIXTURES = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "slm_training"
    / "resources"
    / "formal"
    / "mechanism_trigger_fixtures.v1.json"
)


def test_supported_mechanisms_necessary_and_absent_no_effect() -> None:
    singleton = observation_from_singleton_bypass(
        input_id="t/singleton",
        is_singleton=False,
        baseline_output=42,
        enabled_output=42,
        baseline_cost=3,
        enabled_cost=3,
    )
    closure = observation_from_closure_removal(
        input_id="t/closure",
        removable_nonempty=False,
        baseline_output=7,
        enabled_output=7,
        baseline_cost=5,
        enabled_cost=5,
    )
    cache = observation_from_cache_reuse(
        input_id="t/cache",
        cache_hit=False,
        baseline_output=11,
        enabled_output=11,
        baseline_cost=9,
        enabled_cost=9,
    )
    forced = observation_from_forced_span(
        input_id="t/forced",
        forced_nonempty=False,
        baseline_output=13,
        enabled_output=13,
        baseline_cost=4,
        enabled_cost=4,
    )
    for obs in (singleton, closure, cache, forced):
        assert trigger_evidence_complete(obs.trigger)
        assert necessary_trigger(obs)
        assert outputs_equal(obs)
        assert obs.trigger is TriggerEvidence.ABSENT


def test_singleton_change_requires_present_trigger() -> None:
    changed = observation_from_singleton_bypass(
        input_id="t/singleton_present",
        is_singleton=True,
        baseline_output=1,
        enabled_output=99,
        baseline_cost=4,
        enabled_cost=0,
    )
    assert changed.trigger is TriggerEvidence.PRESENT
    assert necessary_trigger(changed)
    with pytest.raises(ValueError, match="well-formedness"):
        observation_from_singleton_bypass(
            input_id="bad",
            is_singleton=False,
            baseline_output=1,
            enabled_output=2,
            baseline_cost=0,
            enabled_cost=0,
        )


def test_no_effect_certificate_requires_complete_absent_evidence() -> None:
    obs = observation_from_singleton_bypass(
        input_id="cert/ok",
        is_singleton=False,
        baseline_output=1,
        enabled_output=1,
        baseline_cost=2,
        enabled_cost=2,
    )
    cert = emit_no_effect_certificate(
        mechanism=SINGLETON_BYPASS,
        cost_model=NEURAL_FORWARD_ONLY,
        corpus_id="kern08/test",
        corpus=[obs],
    )
    assert cert.schema == CERTIFICATE_SCHEMA
    assert cert.generalizes_beyond_corpus is False
    assert cert.to_dict()["all_outputs_equal"] is True


def test_dominance_certificate() -> None:
    obs = observation_from_singleton_bypass(
        input_id="dom/ok",
        is_singleton=False,
        baseline_output=1,
        enabled_output=1,
        baseline_cost=2,
        enabled_cost=5,
    )
    assert dominates_locally(obs)
    dom = emit_dominance_certificate(
        mechanism=SINGLETON_BYPASS,
        cost_model=NEURAL_FORWARD_ONLY,
        corpus_id="kern08/dom",
        corpus=[obs],
    )
    assert dom.to_dict()["schema"] == DOMINANCE_SCHEMA
    assert dom.to_dict()["dominates_locally"] is True
    cheap = observation_from_singleton_bypass(
        input_id="dom/cheap",
        is_singleton=False,
        baseline_output=1,
        enabled_output=1,
        baseline_cost=5,
        enabled_cost=2,
    )
    with pytest.raises(CertificateError, match="dominance"):
        emit_dominance_certificate(
            mechanism=SINGLETON_BYPASS,
            cost_model=NEURAL_FORWARD_ONLY,
            corpus_id="kern08/dom_fail",
            corpus=[cheap],
        )


def test_unknown_activation_never_no_effect() -> None:
    assert unknown_activation_never_no_effect()


def test_unconstrained_rerank_has_no_safe_theorem() -> None:
    assert UNCONSTRAINED_RERANK.has_safe_trigger_theorem is False
    counter = unconstrained_rerank_counterexample()
    assert counter.enabled_output != counter.baseline_output
    with pytest.raises(CertificateError, match="no safe trigger theorem"):
        emit_no_effect_certificate(
            mechanism=UNCONSTRAINED_RERANK,
            cost_model=NEURAL_FORWARD_ONLY,
            corpus_id="counterexample",
            corpus=[
                observation_from_singleton_bypass(
                    input_id="unused",
                    is_singleton=False,
                    baseline_output=0,
                    enabled_output=0,
                    baseline_cost=0,
                    enabled_cost=0,
                )
            ],
        )


def test_frozen_fixtures() -> None:
    payload = json.loads(_FIXTURES.read_text(encoding="utf-8"))
    assert payload["schema"] == FIXTURE_SCHEMA
    assert payload["generalizes_beyond_corpus"] is False
    live = {row["id"]: row for row in certified_fixture_rows()}
    assert "singleton_bypass_absent_no_effect" in live
    assert live["singleton_bypass_absent_no_effect"]["certificate"]["schema"] == (
        CERTIFICATE_SCHEMA
    )
    assert live["singleton_bypass_dominated"]["certificate"]["dominates_locally"] is True
    assert live["unknown_activation_rejected"]["certificate_error"] is True
    assert live["unconstrained_rerank_no_safe_theorem"]["certificate_error"] is True
    assert all(
        live[mid]["mechanism"]["has_safe_trigger_theorem"]
        for mid in (
            "singleton_bypass_absent_no_effect",
            "closure_removal_absent_no_effect",
            "cache_reuse_absent_no_effect",
            "forced_span_absent_no_effect",
        )
    )
    assert CLOSURE_REMOVAL.has_safe_trigger_theorem
    assert CACHE_REUSE.has_safe_trigger_theorem
    assert FORCED_SPAN.has_safe_trigger_theorem


def test_lean_claim_catalog_entries() -> None:
    catalog = lean_claim_catalog()
    for theorem_id in (
        "MechanismTrigger.no_effect_of_necessary_and_absent",
        "MechanismTrigger.singleton_bypass_output_change_implies_trigger",
        "MechanismTrigger.closure_removal_output_change_implies_trigger",
        "MechanismTrigger.cache_reuse_output_change_implies_trigger",
        "MechanismTrigger.forced_span_output_change_implies_trigger",
        "MechanismTrigger.dominanceCertificate_dominates",
        "MechanismTrigger.unknown_activation_blocks_no_effect_certificate",
        THEOREM_UNSAFE,
    ):
        assert theorem_id in catalog
        assert catalog[theorem_id]["module"] == "LeverProofLean.MechanismTrigger"
