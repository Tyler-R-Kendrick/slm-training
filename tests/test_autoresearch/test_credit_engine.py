"""Authoritative credit engine + promotion integration tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from slm_training.autoresearch.credit_engine import (
    CreditEngineError,
    SealedScoreLedger,
    analysis_plan_from_mapping,
    assert_score_once,
    compute_credit_report,
    credit_defaults_digest,
    load_credit_defaults,
    observation_table_from_mapping,
    seal_identity_from_table,
    verify_credit_against_caller,
)
from slm_training.autoresearch.experiment_campaign import validate_result_claim
from slm_training.harnesses.experiments.promotion import evaluate_promotion
from tests.test_autoresearch.test_experiment_campaign import (
    _complete_result,
    _manifest,
)


def _small_table_plan(
    *, delta: float = 0.1, n_pairs: int = 4
) -> tuple[dict, dict]:
    rows = []
    for i in range(n_pairs):
        rows.append(
            {
                "arm_id": "control",
                "seed": i % 2,
                "example_id": f"ex-{i}",
                "metric_id": "primary",
                "value": 0.5,
                "direction": "increase",
                "split": "search",
            }
        )
        rows.append(
            {
                "arm_id": "candidate",
                "seed": i % 2,
                "example_id": f"ex-{i}",
                "metric_id": "primary",
                "value": 0.5 + delta,
                "direction": "increase",
                "split": "search",
            }
        )
    table = {
        "schema": "observation_table/v1",
        "version": "v1",
        "campaign_id": "c1",
        "experiment_id": "e1",
        "manifest_sha256": "a" * 64,
        "locked_eval_manifest_sha256": "b" * 64,
        "rows": rows,
    }
    plan = {
        "schema": "analysis_plan/v1",
        "version": "v1",
        "primary_metric": "primary",
        "primary_endpoint_id": "primary",
        "direction": "increase",
        "minimum_effect": 0.01,
        "control_arm_id": "control",
        "candidate_arm_id": "candidate",
        "holm_hypothesis_ids": ["primary"],
        "holm_alpha": 0.05,
        "seal_split": "seal",
        "score_once": True,
        "underpowered_n_below": 2,
    }
    return table, plan


def test_credit_defaults_load_and_digest_changes() -> None:
    defaults = load_credit_defaults()
    assert defaults["schema"] == "authoritative_credit_defaults/v1"
    d1 = credit_defaults_digest(defaults)
    mutated = dict(defaults)
    mutated["default_holm_alpha"] = 0.01
    assert credit_defaults_digest(mutated) != d1


def test_compute_credit_promotable_on_clear_gain() -> None:
    table_raw, plan_raw = _small_table_plan(delta=0.1, n_pairs=6)
    table = observation_table_from_mapping(table_raw)
    plan = analysis_plan_from_mapping(plan_raw)
    report = compute_credit_report(table, plan, promotion_gate_threshold=0.01)
    assert report.paired_effect == pytest.approx(0.1)
    assert report.promotable_empirical is True
    assert report.disposition == "promotable"


def test_compute_credit_inconclusive_when_underpowered() -> None:
    table_raw, plan_raw = _small_table_plan(delta=0.1, n_pairs=1)
    plan_raw["underpowered_n_below"] = 4
    table = observation_table_from_mapping(table_raw)
    plan = analysis_plan_from_mapping(plan_raw)
    report = compute_credit_report(table, plan, promotion_gate_threshold=0.01)
    assert report.underpowered is True
    assert report.promotable_empirical is False
    assert report.disposition == "inconclusive"


def test_caller_holm_mismatch_detected() -> None:
    table_raw, plan_raw = _small_table_plan(delta=0.2, n_pairs=6)
    table = observation_table_from_mapping(table_raw)
    plan = analysis_plan_from_mapping(plan_raw)
    report = compute_credit_report(table, plan)
    failures = verify_credit_against_caller(
        report,
        caller_endpoints={
            plan.primary_endpoint_id: report.paired_effect,
        },
        required_endpoint_ids=(plan.primary_endpoint_id,),
        caller_holm=[
            {
                "hypothesis_id": "primary",
                "raw_p_value": 0.99,
                "adjusted_p_value": 0.99,
                "rejected": False,
            }
        ],
    )
    assert not any("caller_endpoint_mismatch" in f for f in failures)
    assert any("caller_holm_mismatch" in f or "caller_holm_reject_mismatch" in f for f in failures)


def test_caller_invented_endpoint_value_mismatch() -> None:
    """Invented high endpoint_values must not silently pass when true effect is small."""
    table_raw, plan_raw = _small_table_plan(delta=0.05, n_pairs=6)
    table = observation_table_from_mapping(table_raw)
    plan = analysis_plan_from_mapping(plan_raw)
    report = compute_credit_report(table, plan)
    assert report.paired_effect == pytest.approx(0.05)
    failures = verify_credit_against_caller(
        report,
        caller_endpoints={plan.primary_endpoint_id: 0.99},
        required_endpoint_ids=(plan.primary_endpoint_id,),
        caller_holm=None,
    )
    assert any(
        f == f"caller_endpoint_mismatch:{plan.primary_endpoint_id}" for f in failures
    )


def test_kind_only_observation_rejected(tmp_path: Path) -> None:
    from slm_training.autoresearch.credit_engine import load_json_artifact

    path = tmp_path / "observation_table.json"
    path.write_text(json.dumps({"kind": "observation_table"}), encoding="utf-8")
    with pytest.raises(CreditEngineError, match="kind-only|semantic"):
        load_json_artifact(path)


def test_validate_result_claim_accepts_recomputed_credit(tmp_path: Path) -> None:
    manifest = _manifest()
    result = _complete_result(manifest, tmp_path)
    assert validate_result_claim(manifest, result, artifact_root=tmp_path) == ()


def test_validate_result_claim_rejects_summary_only(tmp_path: Path) -> None:
    manifest = _manifest()
    # Build structural result without credit kinds
    artifacts = []
    for kind in (
        "version_stamp",
        "seed_result",
        "paired_examples",
        "endpoint_result",
        "holm_family",
        "agentevals",
        "agentv",
    ):
        path = tmp_path / f"{kind}.json"
        path.write_text(json.dumps({"kind": kind}), encoding="utf-8")
        artifacts.append(
            {
                "kind": kind,
                "uri": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    result = _complete_result(
        manifest,
        tmp_path,
        artifacts=artifacts,
    )
    # _complete_result may re-add credit; force empty credit by rewriting
    from slm_training.autoresearch.experiment_campaign import CampaignResultV1

    payload = result.model_dump(mode="json")
    payload["artifacts"] = artifacts
    result = CampaignResultV1.model_validate(payload)
    failures = validate_result_claim(manifest, result, artifact_root=tmp_path)
    assert "credit_summary_only_rejected" in failures or any(
        f.startswith("missing_credit_artifact") for f in failures
    )


def test_validate_result_claim_rejects_invented_holm(tmp_path: Path) -> None:
    manifest = _manifest()
    result = _complete_result(manifest, tmp_path)
    # Tamper caller Holm away from recomputed values
    bad = result.model_copy(
        update={
            "holm_results": tuple(
                h.model_copy(update={"raw_p_value": 0.99, "adjusted_p_value": 0.99, "rejected": False})
                for h in result.holm_results
            )
        }
    )
    failures = validate_result_claim(manifest, bad, artifact_root=tmp_path)
    assert any("caller_holm" in f for f in failures)


def test_validate_result_claim_rejects_invented_endpoint_values(tmp_path: Path) -> None:
    """Honest credit pass, then invent endpoint_values={'meaning': 0.99} → fail."""
    manifest = _manifest()
    result = _complete_result(manifest, tmp_path)
    assert validate_result_claim(manifest, result, artifact_root=tmp_path) == ()
    true_pe = float(result.endpoint_values["meaning"])
    assert true_pe == pytest.approx(0.05)
    invented = result.model_copy(update={"endpoint_values": {"meaning": 0.99}})
    failures = validate_result_claim(manifest, invented, artifact_root=tmp_path)
    assert "caller_endpoint_mismatch:meaning" in failures
    # Invented high value must not authorize promotion gates either
    assert "promotion_gates_not_passed" not in failures or true_pe >= 0.01
    # Structural gates no longer trust caller: with true effect 0.05, gates still
    # pass via recompute; the decisive failure is the caller mismatch.
    assert any("caller_endpoint_mismatch" in f for f in failures)


def test_validate_result_claim_gates_ignore_caller_endpoint_inflation(
    tmp_path: Path,
) -> None:
    """Caller endpoint_values cannot green a gate when recomputed effect fails."""
    from slm_training.autoresearch.experiment_campaign import CampaignResultV1
    from tests.test_autoresearch.test_experiment_campaign import _write_credit_bundle

    manifest = _manifest()
    # Honest structural scaffold first, then replace credit with tiny effect.
    result = _complete_result(manifest, tmp_path)
    _report, _endpoints, holm = _write_credit_bundle(manifest, tmp_path, delta=0.001)
    arts = []
    for a in result.artifacts:
        path = tmp_path / f"{a.kind}.json"
        arts.append(
            {
                "kind": a.kind,
                "uri": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    # Caller invents a huge primary so the old structural gate path would pass.
    payload = result.model_dump(mode="json")
    payload["artifacts"] = arts
    payload["endpoint_values"] = {"meaning": 0.99}
    payload["holm_results"] = holm
    small = CampaignResultV1.model_validate(payload)
    failures = validate_result_claim(manifest, small, artifact_root=tmp_path)
    assert "caller_endpoint_mismatch:meaning" in failures
    assert (
        "promotion_gates_not_passed" in failures
        or "credit_not_promotable_empirical" in failures
    )


def test_evaluate_promotion_governance_cannot_clear_sufficient_evidence(
    tmp_path: Path,
) -> None:
    """Green structural governance + sole sufficient_evidence must not promote."""
    from unittest.mock import MagicMock

    manifest = _manifest()
    result = _complete_result(manifest, tmp_path)
    # validate_result_claim green on full credit
    assert validate_result_claim(manifest, result, artifact_root=tmp_path) == ()
    store = MagicMock()
    store.validate_campaign_result.return_value = ()
    out = evaluate_promotion(
        campaign_manifest=manifest,
        campaign_result=result,
        campaign_store=store,
        artifact_root=tmp_path,
        ship_suites=None,
    )
    assert out["checks"]["campaign_governance"]["pass"] is True
    # Engine has no comparative loss/rank evidence → sole sufficient_evidence
    assert list(out.get("failures") or []) == ["sufficient_evidence"]
    assert out["promotable"] is False
    note = out["checks"].get("governed_campaign_evidence", {})
    assert note.get("pass") is True
    assert "empirical" in str(note.get("note") or "").lower()


def test_evaluate_promotion_with_credit_still_needs_engine_evidence(
    tmp_path: Path,
) -> None:
    """Full credit + governance green does not invent comparative engine metrics."""
    from unittest.mock import MagicMock

    manifest = _manifest()
    result = _complete_result(manifest, tmp_path)
    store = MagicMock()
    store.validate_campaign_result.return_value = ()
    out = evaluate_promotion(
        campaign_manifest=manifest,
        campaign_result=result,
        campaign_store=store,
        artifact_root=tmp_path,
    )
    assert out["checks"]["campaign_governance"]["pass"] is True
    assert out["promotable"] is False
    assert out.get("failures") == ["sufficient_evidence"]
    note = out["checks"].get("governed_campaign_evidence", {})
    assert note.get("pass") is True


def test_sealed_score_once_denies_reuse() -> None:
    table_raw, plan_raw = _small_table_plan()
    table = observation_table_from_mapping(table_raw)
    plan = analysis_plan_from_mapping(plan_raw)
    identity = seal_identity_from_table(table, plan)
    ledger = SealedScoreLedger()
    assert_score_once(ledger, seal_identity=identity, record=True)
    with pytest.raises(CreditEngineError, match="score-once"):
        assert_score_once(ledger, seal_identity=identity, record=True)
    # Different identity still allowed
    assert_score_once(ledger, seal_identity=identity + "-other", record=True)
