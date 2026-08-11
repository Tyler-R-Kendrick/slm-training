"""INTEG-08 / SLM-562: corpus-local dominance skip-decision evidence."""

from __future__ import annotations

from slm_training.autoresearch.hillclimb import ExhaustedKnobLedger
from slm_training.autoresearch.preflight import has_block, run_preflight
from slm_training.autoresearch.preflight import mechanism_dominance as md
from slm_training.formal.mechanism_trigger import (
    DOMINANCE_SCHEMA,
    NEURAL_FORWARD_ONLY,
    SINGLETON_BYPASS,
    THEOREM_NECESSARY,
    observation_from_singleton_bypass,
)


def _dominated_row():
    return observation_from_singleton_bypass(
        input_id="fixture/singleton_dominated",
        is_singleton=False,
        baseline_output=42,
        enabled_output=42,
        baseline_cost=3,
        enabled_cost=5,
    )


def _cheaper_no_effect_row():
    return observation_from_singleton_bypass(
        input_id="fixture/singleton_cheaper",
        is_singleton=False,
        baseline_output=7,
        enabled_output=7,
        baseline_cost=9,
        enabled_cost=2,
    )


def _present_effect_row():
    return observation_from_singleton_bypass(
        input_id="fixture/singleton_present",
        is_singleton=True,
        baseline_output=1,
        enabled_output=2,
        baseline_cost=4,
        enabled_cost=4,
    )


def _base_kwargs(**overrides):
    payload = {
        "mechanism_id": SINGLETON_BYPASS.id,
        "baseline_id": "baseline/v1",
        "treatment_id": "treatment/singleton_bypass",
        "corpus_id": "kern08/singleton_bypass",
        "corpus_version": "v1",
        "model_id": "fixture-model/v1",
        "observations": [_dominated_row()],
        "scan_complete": True,
        "cost_model": NEURAL_FORWARD_ONLY,
        "theorem_ref": THEOREM_NECESSARY,
        "claim_class": "fixture",
        "campaign_id": "camp.integ08",
        "experiment_id": "exp.integ08",
        "manifest_sha256": "b" * 64,
    }
    payload.update(overrides)
    return payload


def test_discovery_includes_mechanism_dominance() -> None:
    assert md.CHECK_ID in {v.check_id for v in run_preflight({})}


def test_idle_without_dominance_identity() -> None:
    verdict = md.CHECK.run({"mechanism_id": SINGLETON_BYPASS.id})
    assert verdict.verdict == "pass"
    assert "idle" in verdict.reasons[0].lower() or "incomplete" in verdict.reasons[0]


def test_skip_dominated_emits_certificate_and_blocks() -> None:
    admission = md.admit_mechanism_dominance(**_base_kwargs())
    assert admission.disposition == "skip_dominated"
    assert admission.evidence.certificate is not None
    assert admission.evidence.certificate["schema"] == DOMINANCE_SCHEMA
    assert admission.evidence.aggregate["relation"] == "dominated"
    assert admission.evidence.empirical_remainder is None
    assert admission.evidence.generalizes_beyond_corpus is False
    assert admission.evidence.output_equivalence["complete"] is True

    sealed = admission.evidence.to_dict()
    assert md.evidence_integrity_ok(sealed)
    assert md.replay_dominance_decision(sealed) == "skip_dominated"

    verdict = md.CHECK.run(
        {
            "mechanism_id": SINGLETON_BYPASS.id,
            "baseline_id": "baseline/v1",
            "treatment_id": "treatment/singleton_bypass",
            "corpus_id": "kern08/singleton_bypass",
            "corpus_version": "v1",
            "model_id": "fixture-model/v1",
            "observations": [_dominated_row().to_dict()],
            "scan_complete": True,
            "cost_model": NEURAL_FORWARD_ONLY.id,
        }
    )
    assert verdict.verdict == "block"
    assert verdict.data["disposition"] == "skip_dominated"
    assert has_block([verdict])


def test_no_effect_but_cheaper_is_not_dominated() -> None:
    admission = md.admit_mechanism_dominance(
        **_base_kwargs(observations=[_cheaper_no_effect_row()])
    )
    assert admission.disposition == "no_effect_cheaper"
    assert admission.evidence.certificate is None
    assert admission.evidence.no_effect_certificate is not None
    assert admission.evidence.aggregate["relation"] == "cheaper"
    assert admission.evidence.empirical_remainder == "cost_improvement_blocks_dominance"
    assert md.replay_dominance_decision(admission.evidence.to_dict()) == "no_effect_cheaper"

    verdict = md.CHECK.run(
        {
            "mechanism_id": SINGLETON_BYPASS.id,
            "baseline_id": "baseline/v1",
            "treatment_id": "treatment/cheaper",
            "corpus_id": "kern08/cheaper",
            "corpus_version": "v1",
            "model_id": "fixture-model/v1",
            "observations": [_cheaper_no_effect_row().to_dict()],
            "scan_complete": True,
        }
    )
    assert verdict.verdict == "pass"
    assert verdict.data["disposition"] == "no_effect_cheaper"


def test_unknown_when_trigger_present_or_incomplete() -> None:
    present = md.admit_mechanism_dominance(
        **_base_kwargs(observations=[_present_effect_row()])
    )
    assert present.disposition == "unknown"
    assert present.evidence.certificate is None

    incomplete = md.admit_mechanism_dominance(
        **_base_kwargs(scan_complete=False)
    )
    assert incomplete.disposition == "unknown"
    assert incomplete.evidence.empirical_remainder == "incomplete_scan"


def test_identity_mutation_invalidates_certificate() -> None:
    evidence = md.emit_dominance_evidence(**_base_kwargs())
    sealed = evidence.to_dict()
    assert md.scope_matches_evidence(
        sealed,
        baseline_id="baseline/v1",
        treatment_id="treatment/singleton_bypass",
        corpus_id="kern08/singleton_bypass",
        corpus_version="v1",
        model_id="fixture-model/v1",
        cost_model=NEURAL_FORWARD_ONLY,
        mechanism_id=SINGLETON_BYPASS.id,
    )

    for field, value in (
        ("corpus_id", "kern08/mutated"),
        ("corpus_version", "v2"),
        ("model_id", "other-model"),
        ("baseline_id", "baseline/v9"),
        ("treatment_id", "treatment/other"),
        ("mechanism_id", "cache_reuse"),
    ):
        kwargs = dict(
            baseline_id="baseline/v1",
            treatment_id="treatment/singleton_bypass",
            corpus_id="kern08/singleton_bypass",
            corpus_version="v1",
            model_id="fixture-model/v1",
            cost_model=NEURAL_FORWARD_ONLY,
            mechanism_id=SINGLETON_BYPASS.id,
        )
        kwargs[field] = value
        assert md.scope_matches_evidence(sealed, **kwargs) is False

    # Cost-model identity mutation.
    assert (
        md.scope_matches_evidence(
            sealed,
            baseline_id="baseline/v1",
            treatment_id="treatment/singleton_bypass",
            corpus_id="kern08/singleton_bypass",
            corpus_version="v1",
            model_id="fixture-model/v1",
            cost_model="DecodeUnitWorkModel.v1",
            mechanism_id=SINGLETON_BYPASS.id,
        )
        is False
    )

    tampered = dict(sealed)
    tampered["scope"] = dict(sealed["scope"])
    tampered["scope"]["corpus_version"] = "mutated"
    assert md.evidence_integrity_ok(tampered) is False
    assert md.replay_dominance_decision(tampered) == "unknown"


def test_record_dominated_skip_into_exhausted_ledger() -> None:
    evidence = md.emit_dominance_evidence(**_base_kwargs())
    ledger = ExhaustedKnobLedger()
    entry = md.record_dominated_skip(
        evidence,
        ledger,
        knobs={
            "mechanism_id": SINGLETON_BYPASS.id,
            "treatment_id": "treatment/singleton_bypass",
        },
    )
    assert entry is not None
    assert entry.reason == md.DOMINANCE_SKIP_REASON
    assert ledger.is_exhausted(
        knob_signature_sha256=entry.knob_signature_sha256,
        data_eval_identity=entry.data_eval_identity,
        claim_class="fixture",
    )

    # Cheaper no-effect must not exhaust the knob.
    cheaper = md.emit_dominance_evidence(
        **_base_kwargs(observations=[_cheaper_no_effect_row()])
    )
    assert md.record_dominated_skip(cheaper, ledger, knobs={"x": 1}) is None


def test_feed_dominance_to_disposition_distinguishes_cases() -> None:
    dominated = md.feed_dominance_to_disposition(
        md.emit_dominance_evidence(**_base_kwargs())
    )
    assert dominated["dominance_disposition"] == "skip_dominated"
    assert dominated["disposition_record"]["disposition"] == "reject"

    cheaper = md.feed_dominance_to_disposition(
        md.emit_dominance_evidence(
            **_base_kwargs(observations=[_cheaper_no_effect_row()])
        )
    )
    assert cheaper["dominance_disposition"] == "no_effect_cheaper"
    assert cheaper["disposition_record"]["disposition"] == "retain_diagnostic"

    unknown = md.feed_dominance_to_disposition(
        md.emit_dominance_evidence(**_base_kwargs(scan_complete=False))
    )
    assert unknown["dominance_disposition"] == "unknown"
    assert unknown["disposition_record"]["disposition"] == "inconclusive"
