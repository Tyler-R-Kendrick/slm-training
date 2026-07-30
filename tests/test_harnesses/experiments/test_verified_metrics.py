from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.casefiles import case_values

from slm_training.autoresearch.experiment_campaign import (
    CampaignResultV1,
    campaign_manifest_sha256,
)
from slm_training.harnesses.experiments import verified_metrics
from slm_training.harnesses.experiments.promotion import (
    register_promoted_checkpoint,
)
from slm_training.harnesses.experiments.verified_metrics import (
    VerifiedMetricError,
    build_metric_evidence,
    optimum_feedback,
    verify_metric_certificate,
)
from slm_training.harnesses.experiments.slm183_power_protocol import (
    build_experiment_campaign,
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def _evidence(tmp_path: Path) -> dict:
    campaign = build_experiment_campaign(seeds=(0,))
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(campaign.model_dump_json(), encoding="utf-8")
    return build_metric_evidence(
        run_id="run-1",
        evidence_bundle_path=_write(tmp_path / "bundle.json", b"bundle"),
        feature_flags_path=_write(tmp_path / "flags.json", b"flags"),
        campaign_manifest_path=campaign_path,
        cold_requests=1,
        warm_requests=9,
        candidates=[
            {
                "id": "candidate",
                "hardware": "cpu",
                "lever_snapshot_sha256": "a" * 64,
                "cold_ns": [100],
                "warm_ns": [20, 21],
                "input_units": [512],
                "passes": [2],
                "successes": 3,
                "quality_failures": 0,
                "trainable_params": 100,
                "energy_uj": [5],
                "cost_micro_usd": [1],
            }
        ],
    )


def _certificate(evidence: dict) -> dict:
    return {
        "schema": "metric_certificate/v1",
        "checker": "leverproof-lean/v1",
        "verified": True,
        "assurance": "observed_raw_samples",
        "run_id": evidence["run_id"],
        "evidence_sha256": evidence["source"]["evidence_sha256"],
        "feature_flags_sha256": evidence["source"]["feature_flags_sha256"],
        "campaign_manifest_sha256": evidence["source"]["campaign_manifest_sha256"],
        "selected_candidate": "candidate",
        "candidates": [{"id": "candidate"}],
    }


def _band_certificate(
    evidence: dict,
    *,
    authority: str = "assumption_backed",
    relation: str = "in_band",
) -> dict:
    return {
        **_certificate(evidence),
        "schema": "metric_certificate/v2",
        "checker": "leverproof-lean/v2",
        "assurance": "calculated_bands_and_observed_raw_samples",
        "metric_expectations_sha256": evidence["source"][
            "metric_expectations_sha256"
        ],
        "assessments": [
            {
                "metric_id": "tokens_per_step",
                "authority": authority,
                "relation": relation,
            }
        ],
    }


def _band_evidence(tmp_path: Path) -> dict:
    expectations = {
        "schema": "metric_expectations/v1",
        "metrics": [
            {
                "metric_id": "tokens_per_step",
                "unit": "tokens",
                "authority": "assumption_backed",
                "dependencies": [
                    {
                        "name": "batch",
                        "value": {
                            "lower": {"numerator": 8, "denominator": 1},
                            "upper": {"numerator": 16, "denominator": 1},
                        },
                    }
                ],
                "program": [{"op": "variable", "name": "batch"}],
            }
        ],
    }
    observations = {
        "schema": "metric_observations/v1",
        "metrics": {"tokens_per_step": [8, 12, 16]},
    }
    expectations_path = tmp_path / "expectations.json"
    observations_path = tmp_path / "observations.json"
    expectations_path.write_text(json.dumps(expectations), encoding="utf-8")
    observations_path.write_text(json.dumps(observations), encoding="utf-8")
    campaign = build_experiment_campaign(seeds=(0,))
    campaign_path = tmp_path / "band-campaign.json"
    campaign_path.write_text(campaign.model_dump_json(), encoding="utf-8")
    return build_metric_evidence(
        run_id="run-1",
        evidence_bundle_path=_write(tmp_path / "band-bundle.json", b"bundle"),
        feature_flags_path=_write(tmp_path / "band-flags.json", b"flags"),
        campaign_manifest_path=campaign_path,
        cold_requests=1,
        warm_requests=9,
        candidates=_evidence(tmp_path)["candidates"],
        expectation_manifest_path=expectations_path,
        observations_path=observations_path,
    )


def test_build_metric_evidence_hashes_provenance(tmp_path: Path) -> None:
    evidence = _evidence(tmp_path)

    assert evidence["source"]["evidence_sha256"] == _digest(b"bundle")
    assert evidence["source"]["feature_flags_sha256"] == _digest(b"flags")
    assert evidence["candidates"][0]["warm_ns"] == [20, 21]


def test_verify_metric_certificate_replays_checker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _evidence(tmp_path)
    certificate = _certificate(evidence)
    evidence_path = tmp_path / "metric-evidence.json"
    certificate_path = tmp_path / "metric-certificate.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    certificate_path.write_text(json.dumps(certificate), encoding="utf-8")
    checker = _write(tmp_path / "leverproof", b"")
    monkeypatch.setattr(
        verified_metrics.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0], returncode=0, stdout="verified\n", stderr=""
        ),
    )

    actual = verify_metric_certificate(
        evidence_path=evidence_path,
        certificate_path=certificate_path,
        expected_campaign_manifest_sha256=evidence["source"][
            "campaign_manifest_sha256"
        ],
        expected_selected_candidate="candidate",
        checker=checker,
    )

    assert actual["verified"] is True


def test_band_evidence_binds_expectations_and_observations(tmp_path: Path) -> None:
    evidence = _band_evidence(tmp_path)

    assert evidence["schema"] == "metric_evidence/v2"
    assert len(evidence["source"]["metric_expectations_sha256"]) == 64
    assert evidence["expectations"][0]["observations"] == [8, 12, 16]


@pytest.mark.parametrize(
    ("authority", "relation", "policy"),
    case_values(__file__, "test_optimum_feedback_uses_tiered_policy"),
)
def test_optimum_feedback_uses_tiered_policy(
    tmp_path: Path, authority: str, relation: str, policy: str
) -> None:
    evidence = _band_evidence(tmp_path)

    feedback = optimum_feedback(
        _band_certificate(evidence, authority=authority, relation=relation)
    )

    assert feedback["policy"] == policy
    assert bool(feedback["diagnosis_lanes"]) == (relation != "in_band")


def test_new_promotion_rejects_assumption_backed_band_miss(
    tmp_path: Path,
) -> None:
    evidence = _band_evidence(tmp_path)
    evidence_path = tmp_path / "metric-evidence.json"
    certificate_path = tmp_path / "metric-certificate.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    certificate_path.write_text(
        json.dumps(_band_certificate(evidence, relation="above")),
        encoding="utf-8",
    )

    with pytest.raises(VerifiedMetricError, match="assumption-backed"):
        verify_metric_certificate(
            evidence_path=evidence_path,
            certificate_path=certificate_path,
            expected_metric_expectations_sha256=evidence["source"][
                "metric_expectations_sha256"
            ],
            require_band_certificate=True,
        )


def test_verify_metric_certificate_rejects_wrong_selected_candidate(
    tmp_path: Path,
) -> None:
    evidence = _evidence(tmp_path)
    evidence_path = tmp_path / "metric-evidence.json"
    certificate_path = tmp_path / "metric-certificate.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    certificate_path.write_text(json.dumps(_certificate(evidence)), encoding="utf-8")

    with pytest.raises(VerifiedMetricError, match="promoted candidate"):
        verify_metric_certificate(
            evidence_path=evidence_path,
            certificate_path=certificate_path,
            expected_selected_candidate="different",
        )


def test_build_metric_evidence_rejects_aggregate_only_input(tmp_path: Path) -> None:
    candidate = {
        "id": "aggregate",
        "hardware": "cpu",
        "lever_snapshot_sha256": "a" * 64,
        "cold_ns": 100,
    }
    with pytest.raises(VerifiedMetricError, match="cold_ns"):
        build_metric_evidence(
            run_id="run-1",
            evidence_bundle_path=_write(tmp_path / "bundle.json", b"bundle"),
            feature_flags_path=_write(tmp_path / "flags.json", b"flags"),
            campaign_manifest_path=None,
            cold_requests=1,
            warm_requests=0,
            candidates=[candidate],
        )


def test_checkpoint_registration_fails_closed_without_certificate(
    tmp_path: Path,
) -> None:
    manifest = build_experiment_campaign(seeds=(0,)).model_copy(
        update={"metric_expectations_sha256": "d" * 64}
    )
    manifest_sha = campaign_manifest_sha256(manifest)
    result = CampaignResultV1(
        campaign_id=manifest.campaign_id,
        experiment_id=manifest.experiment_id,
        manifest_sha256=manifest_sha,
        claim_class="promotion_candidate",
    )
    promotion = {
        "promotable": True,
        "checks": {
            "campaign_governance": {
                "pass": True,
                "manifest_sha256": manifest_sha,
            }
        },
    }
    store = SimpleNamespace(validate_campaign_result=lambda *args, **kwargs: ())

    with pytest.raises(ValueError, match="verified resource metrics"):
        register_promoted_checkpoint(
            tmp_path / "checkpoints",
            promotion_result=promotion,
            campaign_manifest=manifest,
            campaign_result=result,
            campaign_store=store,
            artifact_root=tmp_path,
            promoted_candidate_id="candidate",
        )
