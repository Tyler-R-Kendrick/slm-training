"""KERN-07 / SLM-530: singleton zero-neural policy + decode telemetry refinement."""

from __future__ import annotations

import json
from pathlib import Path

from slm_training.formal.event_trace import (
    NEURAL_FORWARD_ONLY_MODEL,
    Event,
    EventKind,
    project_decode_stats,
    resource_cost_model_id,
    trace_cost,
)
from slm_training.formal.singleton_neural_policy import (
    FIXTURE_SCHEMA,
    THEOREM_REF,
    SingletonProofV1,
    certified_singleton_fixture_rows,
    mutation_rejects_incomplete_singleton_domain,
    mutation_rejects_mismatched_forced_token,
    observe_singleton_runtime,
    policy_neural_cost,
    singleton_bypass_neural_trace,
    singleton_policy_neural_cost_zero,
    singleton_proof_valid,
)
from slm_training.models.decode_stats import DecodeStats, proves_zero_neural_work

_FIXTURES = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "slm_training"
    / "resources"
    / "formal"
    / "singleton_neural_policy_fixtures.v1.json"
)


def test_valid_singleton_forced_equals_sole_candidate() -> None:
    proof = SingletonProofV1(status="complete", candidates=((9,),), forced_token=9)
    assert singleton_proof_valid(proof)
    assert singleton_policy_neural_cost_zero(proof)
    assert policy_neural_cost(singleton_bypass_neural_trace(proof)) == 0


def test_mutation_rejects_mismatched_forced_token() -> None:
    assert mutation_rejects_mismatched_forced_token()
    proof = SingletonProofV1(status="complete", candidates=((0,),), forced_token=1)
    assert not singleton_proof_valid(proof)
    assert not singleton_policy_neural_cost_zero(proof)


def test_mutation_rejects_incomplete_singleton_domain() -> None:
    assert mutation_rejects_incomplete_singleton_domain()
    proof = SingletonProofV1(status="incomplete", candidates=((0,),), forced_token=0)
    assert not singleton_proof_valid(proof)


def test_named_model_ignores_non_forward_events() -> None:
    # Grammar / certificate / memory stay outside NeuralForwardOnlyModel.
    mixed = (
        Event(EventKind.GRAMMAR_TRANSITION, 5),
        Event(EventKind.CERTIFICATE_CHECK, 3),
        Event(EventKind.MEMORY_READ, 100),
        Event(EventKind.NEURAL_FORWARD, 0),
    )
    assert trace_cost(NEURAL_FORWARD_ONLY_MODEL, mixed) == 0
    assert resource_cost_model_id(NEURAL_FORWARD_ONLY_MODEL) == "NeuralForwardOnlyModel.v1"


def test_runtime_projection_zero_forwards_on_certified_fixture() -> None:
    proof = SingletonProofV1(status="complete", candidates=((7,),), forced_token=7)
    stats = DecodeStats(
        forwards_count=0,
        tokens_emitted=1,
        dfa_sync_count=2,
        solver_verifier_calls=1,
        total_ms=4.5,  # wall-clock remainder — not charged by the theorem
    )
    observed = observe_singleton_runtime(proof, stats)
    assert observed.proof_valid
    assert observed.observed_forwards_count == 0
    assert observed.projected_neural_cost == 0
    assert observed.policy_neural_cost == 0
    assert observed.proves_zero_neural_work
    assert proves_zero_neural_work(stats)
    assert observed.wall_clock_claim is False
    assert observed.grammar_certificate_memory_wall_clock_outside_theorem
    assert observed.theorem_ref == THEOREM_REF
    # Unit-work projection still sees grammar/cert counts; neural-only model does not.
    unit = project_decode_stats(stats)  # default DecodeUnitWorkModel
    assert unit.abstract_cost == 1 + 2 + 1  # tokenize + grammar + cert; forwards 0
    neural = project_decode_stats(stats, model=NEURAL_FORWARD_ONLY_MODEL)
    assert neural.abstract_cost == 0


def test_frozen_fixture_file_agrees_without_stronger_theorem() -> None:
    payload = json.loads(_FIXTURES.read_text(encoding="utf-8"))
    assert payload["schema"] == FIXTURE_SCHEMA
    assert payload["cost_model"] == "NeuralForwardOnlyModel"
    assert payload["wall_clock_claim"] is False
    assert payload["implementation_refinement_status"] == "empirical_remainder"
    live_rows = certified_singleton_fixture_rows()
    by_id = {row["id"]: row for row in live_rows}
    for case in payload["cases"]:
        live = by_id[case["id"]]["live"]
        assert live["proof_valid"] is case["expected_proof_valid"]
        assert live["observed_forwards_count"] == case["expected_observed_forwards"]
        assert live["projected_neural_cost"] == case["expected_projected_neural_cost"]
        assert live["implementation_refinement_status"] == "empirical_remainder"
        assert live["wall_clock_claim"] is False
        if case["expected_proof_valid"]:
            assert live["proves_zero_neural_work"] is True
            assert live["policy_neural_cost"] == 0
