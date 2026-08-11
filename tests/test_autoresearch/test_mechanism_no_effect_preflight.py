"""INTEG-02 / SLM-555: theorem-backed mechanism no-effect admission."""

from __future__ import annotations

from types import SimpleNamespace

from slm_training.autoresearch.preflight import has_block, run_preflight
from slm_training.autoresearch.preflight import mechanism_no_effect as mno
from slm_training.formal.mechanism_trigger import (
    CERTIFICATE_SCHEMA,
    NEURAL_FORWARD_ONLY,
    SINGLETON_BYPASS,
    THEOREM_NECESSARY,
    load_fixture_payload,
    observation_from_singleton_bypass,
)


def _absent_row():
    return observation_from_singleton_bypass(
        input_id="fixture/singleton_absent",
        is_singleton=False,
        baseline_output=42,
        enabled_output=42,
        baseline_cost=3,
        enabled_cost=3,
    )


def _present_success_row():
    """Historical successful treatment: trigger present + output change."""

    return observation_from_singleton_bypass(
        input_id="fixture/singleton_present_effect",
        is_singleton=True,
        baseline_output=1,
        enabled_output=2,
        baseline_cost=4,
        enabled_cost=1,
    )


def test_discovery_includes_mechanism_no_effect() -> None:
    verdicts = run_preflight({})
    assert mno.CHECK_ID in {v.check_id for v in verdicts}


def test_idle_without_mechanism_id() -> None:
    verdict = mno.CHECK.run({"slug": "unrelated"})
    assert verdict.verdict == "pass"
    assert "idle" in verdict.reasons[0]


def test_skip_when_complete_absent_evidence() -> None:
    admission = mno.admit_mechanism_treatment(
        mechanism_id=SINGLETON_BYPASS.id,
        corpus_id="kern08/singleton_bypass",
        observations=[_absent_row()],
        scan_complete=True,
        campaign_id="camp.integ02",
        experiment_id="exp.integ02",
        manifest_sha256="a" * 64,
        cost_model=NEURAL_FORWARD_ONLY,
        theorem_ref=THEOREM_NECESSARY,
    )
    assert admission.disposition == "skip_no_effect"
    assert admission.certificate is not None
    assert admission.certificate["schema"] == CERTIFICATE_SCHEMA
    assert admission.campaign_id == "camp.integ02"
    assert admission.evidence_lineage["certificate_sha256"]
    assert admission.content_sha256

    verdict = mno.CHECK.run(
        {
            "mechanism_id": SINGLETON_BYPASS.id,
            "corpus_id": "kern08/singleton_bypass",
            "observations": [_absent_row().to_dict()],
            "scan_complete": True,
            "campaign_id": "camp.integ02",
            "experiment_id": "exp.integ02",
            "manifest_sha256": "a" * 64,
        }
    )
    assert verdict.verdict == "block"
    assert verdict.data["disposition"] == "skip_no_effect"
    assert has_block([verdict])


def test_run_when_necessary_trigger_present() -> None:
    admission = mno.admit_mechanism_treatment(
        mechanism_id=SINGLETON_BYPASS.id,
        corpus_id="kern08/singleton_bypass_active",
        observations=[_present_success_row()],
        scan_complete=True,
    )
    assert admission.disposition == "run"
    assert admission.certificate is None
    assert "present" in admission.reasons[0]


def test_run_when_unknown_evidence() -> None:
    row = {
        "input_id": "counterexample/unknown_activation",
        "baseline_output": 1,
        "enabled_output": 1,
        "baseline_cost": 1,
        "enabled_cost": 1,
        "trigger": "unknown",
    }
    admission = mno.admit_mechanism_treatment(
        mechanism_id=SINGLETON_BYPASS.id,
        corpus_id="counterexample/unknown",
        observations=[row],
        scan_complete=True,
    )
    assert admission.disposition == "run"
    assert "unknown" in admission.reasons[0].lower()


def test_incomplete_scan_never_skips_even_if_all_absent() -> None:
    admission = mno.admit_mechanism_treatment(
        mechanism_id=SINGLETON_BYPASS.id,
        corpus_id="partial/scan",
        observations=[_absent_row()],
        scan_complete=False,
    )
    assert admission.disposition == "run"
    assert "incomplete" in admission.reasons[0]


def test_omitted_scan_complete_treated_as_incomplete() -> None:
    verdict = mno.CHECK.run(
        {
            "mechanism_id": SINGLETON_BYPASS.id,
            "corpus_id": "partial/scan",
            "observations": [_absent_row().to_dict()],
        }
    )
    assert verdict.verdict == "pass"
    assert verdict.data["disposition"] == "run"
    assert "incomplete" in " ".join(verdict.reasons).lower()


def test_fixture_no_effect_rows_skip_deterministically() -> None:
    payload = load_fixture_payload()
    for case in payload["cases"]:
        if case["kind"] != "no_effect":
            continue
        admission = mno.admit_mechanism_treatment(
            mechanism_id=case["mechanism_id"],
            corpus_id=case["corpus_id"],
            observations=[case["observation"]],
            scan_complete=True,
            cost_model=NEURAL_FORWARD_ONLY
            if case.get("cost_model") == "NeuralForwardOnlyModel"
            else None,
            theorem_ref=case.get("theorem_ref", THEOREM_NECESSARY),
            claim_class="fixture",
        )
        assert admission.disposition == "skip_no_effect", case["id"]


def test_admit_for_campaign_preserves_identity() -> None:
    manifest = SimpleNamespace(
        campaign_id="camp.bound",
        experiment_id="exp.bound",
        claim_class="wiring",
    )
    admission = mno.admit_for_campaign(
        manifest,
        mechanism_id=SINGLETON_BYPASS.id,
        corpus_id="kern08/singleton_bypass",
        observations=[_absent_row()],
        scan_complete=True,
        manifest_sha256="b" * 64,
    )
    assert admission.disposition == "skip_no_effect"
    assert admission.campaign_id == "camp.bound"
    assert admission.experiment_id == "exp.bound"
    assert admission.manifest_sha256 == "b" * 64
    assert admission.claim_class == "wiring"


def test_unsafe_mechanism_always_runs() -> None:
    admission = mno.admit_mechanism_treatment(
        mechanism_id="unconstrained_rerank",
        corpus_id="counterexample",
        observations=[_absent_row()],
        scan_complete=True,
    )
    assert admission.disposition == "run"
    assert "no safe trigger theorem" in admission.reasons[0]


def test_replayable_digest_stable() -> None:
    a = mno.admit_mechanism_treatment(
        mechanism_id=SINGLETON_BYPASS.id,
        corpus_id="kern08/singleton_bypass",
        observations=[_absent_row()],
        scan_complete=True,
        campaign_id="camp.integ02",
        experiment_id="exp.integ02",
        manifest_sha256="a" * 64,
    )
    b = mno.admit_mechanism_treatment(
        mechanism_id=SINGLETON_BYPASS.id,
        corpus_id="kern08/singleton_bypass",
        observations=[_absent_row()],
        scan_complete=True,
        campaign_id="camp.integ02",
        experiment_id="exp.integ02",
        manifest_sha256="a" * 64,
    )
    assert a.content_sha256 == b.content_sha256
    assert a.to_dict() == b.to_dict()
