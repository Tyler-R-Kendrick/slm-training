"""Tests for SLM-184 single-touch confirmation firewall and claim manifests."""

from __future__ import annotations

import json

import pytest

from slm_training.harnesses.experiments.claim_manifest import (
    MATRIX_VERSION,
    AccessDecision,
    ExperimentClaimManifestV1,
    SuiteAccessBroker,
    TouchLedger,
    TouchRecord,
    build_default_manifest,
    classify_iter_artifact,
    freeze_manifest,
    is_frozen,
    validate_manifest,
    with_claim_manifest_guard,
)


def _valid_manifest(**overrides) -> ExperimentClaimManifestV1:
    manifest = build_default_manifest()
    if overrides:
        data = manifest.to_dict()
        data.update(overrides)
        manifest = ExperimentClaimManifestV1.from_dict(data)
    return manifest


def test_build_default_manifest_is_valid() -> None:
    manifest = build_default_manifest()
    assert validate_manifest(manifest) == []
    assert manifest.manifest_version == MATRIX_VERSION


def test_manifest_round_trip() -> None:
    manifest = build_default_manifest()
    reconstructed = ExperimentClaimManifestV1.from_dict(manifest.to_dict())
    assert reconstructed == manifest


def test_validate_manifest_requires_family_id() -> None:
    manifest = _valid_manifest(experiment_family_id="")
    errors = validate_manifest(manifest)
    assert any("experiment_family_id" in e for e in errors)


def test_validate_manifest_requires_positive_mde() -> None:
    manifest = _valid_manifest(mde=0.0)
    errors = validate_manifest(manifest)
    assert any("mde" in e for e in errors)


def test_validate_manifest_requires_alpha_in_unit_interval() -> None:
    manifest = _valid_manifest(alpha=1.0)
    errors = validate_manifest(manifest)
    assert any("alpha" in e for e in errors)


def test_validate_manifest_requires_power_in_unit_interval() -> None:
    manifest = _valid_manifest(power=0.0)
    errors = validate_manifest(manifest)
    assert any("power" in e for e in errors)


def test_validate_manifest_rejects_confirmation_suite_in_dev_list() -> None:
    manifest = _valid_manifest(
        confirmation_suite_id="smoke",
        allowed_dev_suite_ids=("smoke", "held_out"),
    )
    errors = validate_manifest(manifest)
    assert any("allowed_dev_suite_ids" in e for e in errors)


def test_validate_manifest_rejects_overlapping_frozen_tunable() -> None:
    manifest = _valid_manifest(frozen_fields=("seeds",), tunable_fields=("seeds",))
    errors = validate_manifest(manifest)
    assert any("frozen" in e and "tunable" in e for e in errors)


def test_freeze_manifest_writes_file(tmp_path) -> None:
    manifest = build_default_manifest()
    frozen_path = freeze_manifest(manifest, tmp_path)
    assert frozen_path.exists()
    assert is_frozen(frozen_path)
    data = json.loads(frozen_path.read_text())
    assert data["schema"] == "ExperimentClaimManifestV1Frozen"
    assert len(data["manifest_sha256"]) == 64
    assert data["manifest"]["experiment_family_id"] == manifest.experiment_family_id
    assert data["version_stamp"]["stamp_schema"] == "version_stamp/v1"


def test_frozen_manifest_is_create_once_and_detects_mutation(tmp_path) -> None:
    manifest = _valid_manifest()
    frozen_path = freeze_manifest(manifest, tmp_path)
    assert freeze_manifest(manifest, tmp_path) == frozen_path
    payload = json.loads(frozen_path.read_text(encoding="utf-8"))
    payload["manifest"]["primary_hypothesis"] = "changed after observation"
    frozen_path.write_text(json.dumps(payload), encoding="utf-8")
    assert is_frozen(frozen_path) is False
    with pytest.raises(RuntimeError, match="integrity"):
        freeze_manifest(manifest, tmp_path)


def test_is_frozen_false_for_missing_path(tmp_path) -> None:
    assert not is_frozen(tmp_path / "missing.json")


def test_touch_record_round_trip() -> None:
    record = TouchRecord(
        experiment_family_id="fam",
        suite_digest="sha256:abc",
        touch_id="t1",
        touch_kind="confirm",
        timestamp="2026-07-20T00:00:00Z",
        prediction_materialized=True,
        reason="test",
    )
    reconstructed = TouchRecord.from_dict(record.to_dict())
    assert reconstructed == record


def test_touch_ledger_rejects_duplicate_touch_id() -> None:
    ledger = TouchLedger()
    record = TouchRecord(
        experiment_family_id="fam",
        suite_digest="sha256:abc",
        touch_id="t1",
        touch_kind="dev",
        timestamp="2026-07-20T00:00:00Z",
        prediction_materialized=False,
        reason="test",
    )
    ledger.record_touch(record)
    with pytest.raises(ValueError, match="duplicate touch_id"):
        ledger.record_touch(record)


def test_confirmation_touches_for_filters_by_kind() -> None:
    ledger = TouchLedger()
    dev = TouchRecord(
        experiment_family_id="fam",
        suite_digest="sha256:abc",
        touch_id="t1",
        touch_kind="dev",
        timestamp="2026-07-20T00:00:00Z",
        prediction_materialized=False,
        reason="dev",
    )
    confirm = TouchRecord(
        experiment_family_id="fam",
        suite_digest="sha256:abc",
        touch_id="t2",
        touch_kind="confirm",
        timestamp="2026-07-20T00:00:01Z",
        prediction_materialized=True,
        reason="confirm",
    )
    ledger.record_touch(dev)
    ledger.record_touch(confirm)
    assert len(ledger.confirmation_touches_for("fam", "sha256:abc")) == 1
    assert ledger.has_prediction_materialized_touch("fam", "sha256:abc")


def test_broker_dev_access_always_allowed() -> None:
    manifest = build_default_manifest()
    broker = SuiteAccessBroker()
    decision = broker.request_dev_access(
        manifest, "smoke", "sha256:abc", prediction_materialized=False
    )
    assert decision.allowed
    assert decision.touch_record is not None
    assert decision.touch_record.touch_kind == "dev"


def test_broker_confirmation_denied_without_frozen_manifest() -> None:
    manifest = build_default_manifest()
    broker = SuiteAccessBroker()
    decision = broker.request_confirmation_access(
        manifest,
        manifest.confirmation_suite_id,
        manifest.confirmation_suite_digest,
    )
    assert not decision.allowed
    assert "not frozen" in decision.reason


def test_broker_confirmation_denied_for_wrong_suite(tmp_path) -> None:
    manifest = build_default_manifest()
    frozen_path = freeze_manifest(manifest, tmp_path)
    broker = SuiteAccessBroker()
    decision = broker.request_confirmation_access(
        manifest,
        "wrong_suite",
        manifest.confirmation_suite_digest,
        frozen_manifest_path=frozen_path,
        prediction_materialized=True,
    )
    assert not decision.allowed
    assert "confirmation_suite_id" in decision.reason


def test_broker_confirmation_lifecycle(tmp_path) -> None:
    manifest = build_default_manifest()
    frozen_path = freeze_manifest(manifest, tmp_path)
    broker = SuiteAccessBroker()

    first = broker.request_confirmation_access(
        manifest,
        manifest.confirmation_suite_id,
        manifest.confirmation_suite_digest,
        frozen_manifest_path=frozen_path,
        prediction_materialized=True,
    )
    assert first.allowed
    assert first.touch_record is not None
    assert first.touch_record.touch_kind == "confirm"

    second = broker.request_confirmation_access(
        manifest,
        manifest.confirmation_suite_id,
        manifest.confirmation_suite_digest,
        frozen_manifest_path=frozen_path,
        prediction_materialized=True,
    )
    assert not second.allowed
    assert "already exists" in second.reason


def test_broker_confirmation_denied_for_digest_mismatch(tmp_path) -> None:
    manifest = build_default_manifest()
    frozen_path = freeze_manifest(manifest, tmp_path)
    broker = SuiteAccessBroker()
    decision = broker.request_confirmation_access(
        manifest,
        manifest.confirmation_suite_id,
        "sha256:mismatch",
        frozen_manifest_path=frozen_path,
        prediction_materialized=True,
    )
    assert not decision.allowed
    assert "digest" in decision.reason


def test_with_claim_manifest_guard_dev_allowed(tmp_path) -> None:
    manifest = build_default_manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()))
    ledger_path = tmp_path / "ledger.json"

    with with_claim_manifest_guard(
        manifest_path, ledger_path, "smoke", "sha256:abc", is_confirmation=False
    ) as decision:
        assert isinstance(decision, AccessDecision)
        assert decision.allowed

    assert ledger_path.exists()
    ledger = TouchLedger.from_dict(json.loads(ledger_path.read_text()))
    assert len(ledger.records) == 1
    assert ledger.records[0].touch_kind == "dev"


def test_with_claim_manifest_guard_confirmation_fail_closed(tmp_path) -> None:
    manifest = build_default_manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()))
    ledger_path = tmp_path / "ledger.json"

    with pytest.raises(PermissionError):
        with with_claim_manifest_guard(
            manifest_path,
            ledger_path,
            manifest.confirmation_suite_id,
            manifest.confirmation_suite_digest,
            is_confirmation=True,
            prediction_materialized=True,
        ):
            pass


def test_with_claim_manifest_guard_confirmation_allowed_when_frozen(tmp_path) -> None:
    manifest = build_default_manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest.to_dict()))
    freeze_manifest(manifest, tmp_path)
    ledger_path = tmp_path / "ledger.json"

    with with_claim_manifest_guard(
        manifest_path,
        ledger_path,
        manifest.confirmation_suite_id,
        manifest.confirmation_suite_digest,
        is_confirmation=True,
        prediction_materialized=True,
    ) as decision:
        assert decision.allowed


def test_classify_iter_artifact_clean_confirmation() -> None:
    data = {
        "status": "confirmatory",
        "claim_class": "confirmatory",
        "suite_role": "confirmation",
        "confirmation_suite_id": "rico_held",
        "source_commit": "abc",
        "version_stamp": {"stamp_schema": "version_stamp/v1", "code_commit": "abc"},
    }
    assert classify_iter_artifact(data) == "clean_confirmation"


def test_classify_iter_artifact_development_only() -> None:
    assert classify_iter_artifact({"status": "dev", "claim_class": "wiring"}) == "development_only"


def test_classify_iter_artifact_reused() -> None:
    data = {
        "status": "fixture",
        "claim_class": "wiring",
        "eval_from_run": "qx_e0_baseline",
        "version_stamp": {"stamp_schema": "version_stamp/v1", "code_commit": "abc"},
    }
    assert classify_iter_artifact(data) == "reused_evaluation_data"


def test_classify_iter_artifact_provenance_incomplete() -> None:
    assert classify_iter_artifact({"status": "fixture", "claim_class": "wiring"}) == "provenance_incomplete"


def test_classify_iter_artifact_not_applicable_fixture() -> None:
    data = {
        "status": "fixture",
        "claim_class": "wiring",
        "version_stamp": {"stamp_schema": "version_stamp/v1", "code_commit": "abc"},
    }
    assert classify_iter_artifact(data) == "not_applicable_fixture"
